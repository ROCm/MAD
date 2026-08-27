###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
import numpy as np
import os
import argparse
import csv
import re
import statistics

# parse arguments
parser = argparse.ArgumentParser(description='Convert pytorch train output format to MAD csv output format')
parser.add_argument("--mode",
                        type=str,
                        help="pretrain or finetune")
parser.add_argument("--model",
                        type=str,
                        help="model name")
parser.add_argument("--quantization",
                        type=str,
                        default="bf16",
                        help="quantization type, e.g. bf16, nanoo_fp8, etc.")
parser.add_argument("--input",
                        type=str,
                        help="path to input file")
parser.add_argument("--output",
                        type=str,
                        help="path to output file")
parser.add_argument("--batch_size",
                        type=str,
                        help="batch size")
parser.add_argument("--seq_len",
                        type=str,
                        help="sequence length")
parser.add_argument("--device",
                        type=str,
                        help="device name")
parser.add_argument("--num_gpus",
                        type=str,
                        help="number of GPUs")
parser.add_argument("--emit-performance-line",
                        action="store_true",
                        help="Print the madengine-scraped 'performance:' line and fail when the "
                             "median is unavailable. OFF by default: madengine scrapes the FIRST "
                             "match, so emitting it unconditionally would silently replace the "
                             "single-node cards' long-standing 'performance: 1 pass' with a "
                             "different metric, and make them fail on a short run. Multi-node "
                             "callers pass it because there the median IS the measurement.")
parser.add_argument("--worker-rank",
                        action="store_true",
                        help="This rank is not the one that reports. MaxText emits step metrics "
                             "only from process 0, and madengine already exempts worker nodes "
                             "from producing performance (MAD_COLLECT_METRICS=false), so a log "
                             "with no step markers is the expected case here rather than a "
                             "failure - without this the worker died on a traceback and took "
                             "the whole multi-node run down with it.")
parser.add_argument("--warmup_steps",
                        type=int,
                        default=10,
                        help="steps to drop before taking the median step time; the first "
                             "steps include XLA compilation and are not comparable between runs")

# read arguments
args = parser.parse_args()
input_file = args.input
output_file = args.output
quantization = args.quantization
print("Input file path: ", input_file)
print("Output file path: ", output_file)
print("Quantization: ", quantization)

# MaxText emits one line per step, e.g.
#   completed step: 12, seconds: 2.518, TFLOP/s/device: 271.3, Tokens/s/device: 12048.3, ...
STEP_RE = re.compile(r"completed step:\s*(\d+)\s*,\s*seconds:\s*([0-9.]+)")


def last_run_lines(log_path):
    """Lines belonging to the final training run in the log.

    Defence in depth for logs produced elsewhere or appended to; the harness itself
    truncates per invocation, which is the primary measure. A run boundary is a step number
    that DECREASES - so this cannot help a run that emitted no step records at all.

    Strictly less-than, not <=: a repeated step number means a duplicated line, not a new
    run. Treating equality as a boundary made every duplicate a restart, and since `start`
    keeps moving, a log with each line written twice - two ranks tee'd into one file, a
    doubled logging handler - collapsed to a single line. The two averages then reported one
    step as if it were ten, and the median silently dropped out of the CSV for want of
    samples, at exit 0.
    """
    with open(log_path, errors="ignore") as fh:
        lines = fh.readlines()
    start, prev = 0, None
    for i, line in enumerate(lines):
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if prev is not None and step < prev:
            start = i          # numbering restarted: a new run begins here
        prev = step
    return lines[start:]



def median_steady_step_seconds(log_path, warmup_steps=10):
    """Median wall time of steady-state steps.

    This is the metric ROCM-27881 is stated in, and it is far more sensitive to a
    collective-path regression than TFLOP/s averages are.

    Warmup steps are dropped because the first steps include XLA compilation, which
    is unrelated to the collective path and varies between runs.

    Dropped by POSITION, not by step number. Filtering on `step >= warmup_steps` assumed a
    run always starts at 0: a run resumed from a checkpoint at step 10 or later dropped
    nothing and folded compile time into the median, while `steps=12` - one word in
    MAXTEXT_EXTRA_ARGS - left 2 samples and failed the run as unmeasurable.

    Raises ValueError when too few steady-state steps are present. Returning a
    sentinel such as 0.0 instead would let a truncated or crashed run become a
    plausible-looking row in the A/B table and silently corrupt the conclusion.
    """
    observed = []
    for line in last_run_lines(log_path):
            m = STEP_RE.search(line)
            if m:
                observed.append(float(m.group(2)))
    steady = observed[warmup_steps:]
    if len(steady) < 3:
        raise ValueError(
            "only %d steady-state steps found in %s (need >=3 after dropping %d warmup "
            "steps); the run may have crashed, been truncated, or the log format changed"
            % (len(steady), log_path, warmup_steps))
    return statistics.median(steady)


def find_match(path, search_string, num_iters):
    content = "".join(last_run_lines(path))
    pattern = fr"{re.escape(search_string)}\s*(\d+\.\d+|\d+)"
    matches = re.findall(pattern, content)
    perf_nums = [float(num) for num in matches][-num_iters:]
    # np.average([]) is nan, not an error: a log with no markers would otherwise produce a
    # structurally valid CSV row reading "nan".
    if not perf_nums:
        raise ValueError(
            "no %r values found in %s - the run produced no steps, or MaxText changed this "
            "log marker" % (search_string, path))
    avg = np.average(perf_nums)
    return str("{:.2f}".format(avg))

SUPPORTED_MODELS = {
    "Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B", "Llama-3.3-70B",
    "Llama-2-7B", "Llama-2-70B", "DeepSeek-V2-lite", "Mixtral-8x7B",
    "Qwen3-14B", "Qwen3-30B-A3B",
}

if args.model not in SUPPORTED_MODELS:
    raise SystemExit(
        "unsupported model %r; add it to SUPPORTED_MODELS. (Previously this fell through "
        "to an undefined `data` and died with NameError further down.)" % (args.model,))

# Remove the destination BEFORE parsing: otherwise a refusal to report leaves an earlier
# run's CSV in place, to be collected as if it were this run's result.
if os.path.exists(output_file):
    print("removing stale %s from a previous run" % output_file)
    os.remove(output_file)

try:
    tok_per_s_per_gpu = find_match(input_file, "Tokens/s/device:", 10)
    TFLOPS_per_gpu = find_match(input_file, "TFLOP/s/device:", 10)
except ValueError as exc:
    if not args.worker_rank:
        raise
    # A worker rank with no step markers is the normal case, not a failure. Exit clean and
    # write nothing: there is no measurement here to record, and madengine's log scan would
    # read a Python traceback as an error and fail a node that it otherwise exempts.
    print("worker rank has no step metrics, as expected: %s" % exc)
    raise SystemExit(0)


def _row(perf, metric):
    return {'model': args.model, 'performance': perf, 'metric': metric, 'mode': args.mode,
            'precision': args.quantization, 'batch_size': args.batch_size,
            'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}


data = [
    _row(tok_per_s_per_gpu, 'tok_per_s_per_gpu'),
    _row(TFLOPS_per_gpu, 'TFLOPS_per_gpu'),
]

# Median steady-state step time - the primary metric for the RCCL FAULT_INJECTION A/B.
# Computed defensively: a short run must still write the tok/s and TFLOPS rows that were
# written unconditionally before this change. MAXTEXT_EXTRA_ARGS makes short runs easy to
# ask for (e.g. steps=12), so this path is reachable in normal use.
median_step_s = None
try:
    median_step_s = median_steady_step_seconds(input_file, args.warmup_steps)
    data.append(_row("{:.4f}".format(median_step_s), 'seconds_per_step'))
except ValueError as exc:
    print("WARNING: primary metric unavailable: %s" % exc)

# Refuse before writing anything. Emitting the secondary rows first and only then exiting
# non-zero leaves an artifact under the exact multiple_results name for the collector to
# pick up, which contradicts the refusal. Single-node keeps its legacy rows: there the
# median is not the measurement.
if args.emit_performance_line and median_step_s is None:
    raise SystemExit(
        "FATAL: no median seconds_per_step could be computed from %s; refusing to report "
        "this run as a measurement, and writing no CSV" % input_file)

with open(output_file, mode='w', newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric','mode','precision','batch_size','seq_len','device','num_gpus'])
    writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")

# Gated on --emit-performance-line so that adding this metric does not change what the
# existing single-node cards report.
if args.emit_performance_line:
    # Scraped by madengine (deployment/base.py PERFORMANCE_LOG_PATTERN). The missing-median
    # case exited above, before the CSV was created.
    print("performance: {:.4f} seconds_per_step".format(median_step_s))
elif median_step_s is not None:
    print("median seconds_per_step (not scraped): {:.4f}".format(median_step_s))
