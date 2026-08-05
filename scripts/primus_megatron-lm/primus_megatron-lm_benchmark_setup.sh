#!/bin/bash
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
export HF_HOME=/workspace/huggingface

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m) MODEL_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

echo "[INFO] Primus setup script starting in directory $(pwd)"

cd /workspace/Primus

# Metrics parser patch for primus-cli-direct.sh. Kept in a single variable so
# the --check probe and the actual apply can never drift apart.
read -r -d '' METRICS_PARSER_PATCH <<'PATCH_EOF' || true
diff --git a/runner/primus-cli-direct.sh b/runner/primus-cli-direct.sh
index b58a1527..2d9882d6 100755
--- a/runner/primus-cli-direct.sh
+++ b/runner/primus-cli-direct.sh
@@ -551,4 +551,87 @@ else
     LOG_INFO "[direct] torchrun finished successfully (code 0)"
 fi
 
+###############################################################################
+# STEP 12: Parse training metrics from log
+###############################################################################
+TRAIN_LOG="${direct_config[log_file]:-}"
+
+# Detect framework from the experiment config YAML passed via --config
+FRAMEWORK=""
+prev_arg=""
+for i in "$@"; do
+    if [[ "$prev_arg" == "--config" && -f "$i" ]]; then
+        FRAMEWORK=$(grep -m1 'framework:' "$i" 2>/dev/null | sed -E 's/.*framework:[[:space:]]*//' | tr -d '[:space:]')
+        break
+    fi
+    prev_arg="$i"
+done
+
+# Both read "<iteration> <value>" pairs on stdin and skip the warmup iterations.
+_metric_hmean() { awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }'; }
+_metric_amean() { awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }'; }
+
+if [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]]; then
+    if [[ "$FRAMEWORK" == "megatron" ]]; then
+        LOG_INFO "[direct] Using Megatron log parser"
+
+        num_warmup=$(grep 'lr_warmup_iters' "${TRAIN_LOG}" | sed -En 's/.*lr_warmup_iters[^:]*:[[:space:]]*([0-9,]+).*/\1/p' | tr -d ',' 2>/dev/null)
+        num_warmup="${num_warmup:-0}"
+        echo "Num warmup: $num_warmup"
+
+        # Primus 26.5 renamed both fields on the iteration line:
+        #   "tokens per GPU (tokens/s/GPU): X"    -> "tokens/s/GPU inst/harmonic mean: X/Y"
+        #   "throughput per GPU (TFLOP/s/GPU): X" -> "compute per GPU (TFLOP/s/GPU): X (avg Y)"
+        # It still emits the 26.4 shape for the first iterations (before the
+        # running means exist), so a single 26.5 log carries BOTH shapes and
+        # picking just one would average the wrong subset. Each shape is a second
+        # -e expression: sed tries it only on lines the first one did not already
+        # rewrite, so every iteration line yields exactly one "<iter> <value>"
+        # pair and a 26.4-only log parses byte-for-byte as before.
+        avg_tps=$(sed -En \
+          -e '/iteration  *[0-9].*tokens per GPU/s/.*iteration  *([0-9]+).*tokens per GPU \(tokens\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' \
+          -e '/iteration  *[0-9].*tokens\/s\/GPU inst/s/.*iteration  *([0-9]+).*tokens\/s\/GPU inst\/harmonic mean:  *([0-9.]+)\/.*/\1 \2/p' \
+          "${TRAIN_LOG}" | _metric_hmean)
+
+        avg_tflops=$(sed -En \
+          -e '/iteration  *[0-9].*throughput per GPU/s/.*iteration  *([0-9]+).*throughput per GPU \(TFLOP\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' \
+          -e '/iteration  *[0-9].*compute per GPU/s/.*iteration  *([0-9]+).*compute per GPU \(TFLOP\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' \
+          "${TRAIN_LOG}" | _metric_hmean)
+
+        avg_mem_pct=$(sed -En '/iteration  *[0-9].*usage_ratio/s/.*iteration  *([0-9]+).*usage_ratio:  *[^/]*\/[^/]*\/[^/]*\/([0-9.]+)%.*/\1 \2/p' "${TRAIN_LOG}" | _metric_amean)
+        avg_elapsed_time=$(sed -En '/iteration  *[0-9].*elapsed time per iteration/s/.*iteration  *([0-9]+).*elapsed time per iteration \(ms\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" | _metric_amean)
+
+        echo "Harmonic mean of TPS (excluding first $num_warmup steps): $avg_tps" | tee -a "${TRAIN_LOG}"
+        echo "Harmonic mean of TFLOPS (excluding first $num_warmup steps): $avg_tflops" | tee -a "${TRAIN_LOG}"
+        echo "Arithmetic mean of memory percentage (excluding first $num_warmup steps): $avg_mem_pct" | tee -a "${TRAIN_LOG}"
+        echo "Arithmetic mean of elapsed time (ms) (excluding first $num_warmup steps): $avg_elapsed_time" | tee -a "${TRAIN_LOG}"
+
+    elif [[ "$FRAMEWORK" == "torchtitan" ]]; then
+        LOG_INFO "[direct] Using Torchtitan log parser"
+
+        num_warmup=$(grep 'lr_scheduler.warmup_steps' ${TRAIN_LOG} | sed -En 's/.*lr_scheduler.warmup_steps[^:]*:[[:space:]]*([0-9,]+).*/\1/p' | tr -d ',' 2>/dev/null)
+        num_warmup="${num_warmup:-0}"
+        echo "Num warmup (first steps skipped): $num_warmup"
+
+        avg_tps=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*tps:  *([0-9,]+).*/\1 \2/p' | tr -d ',' | _metric_hmean)
+
+        avg_tflops=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*tflops:  *([0-9,.]+).*/\1 \2/p' | tr -d ',' | _metric_hmean)
+
+        avg_mfu=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*mfu:  *([0-9,.]+)%.*/\1 \2/p' | tr -d ',' | _metric_amean)
+
+        avg_mem=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*memory:  *[0-9.,]+GiB\(([0-9.]+)%\).*/\1 \2/p' | _metric_amean)
+
+        echo "Harmonic mean of TPS (excluding first $num_warmup steps): $avg_tps" | tee -a "${TRAIN_LOG}"
+        echo "Harmonic mean of TFLOPS (excluding first $num_warmup steps): $avg_tflops" | tee -a "${TRAIN_LOG}"
+        echo "Arithmetic mean of MFU (excluding first $num_warmup steps): $avg_mfu" | tee -a "${TRAIN_LOG}"
+        echo "Arithmetic mean of memory percentage (excluding first $num_warmup steps): $avg_mem" | tee -a "${TRAIN_LOG}"
+    else
+        LOG_INFO "[direct] Framework '$FRAMEWORK' not recognized for log parsing, skipping metrics summary"
+    fi
+fi
+
 exit "$exit_code"
PATCH_EOF

if printf '%s\n' "$METRICS_PARSER_PATCH" | git apply --check - 2>/dev/null; then
  printf '%s\n' "$METRICS_PARSER_PATCH" | git apply -
  echo "[INFO] Metrics parser patch applied successfully to runner/primus-cli-direct.sh"
else
  echo "[WARN] Metrics parser patch could not be applied (already applied or context mismatch), skipping"
fi

# Zebra-Llama FLOPs: Primus may pass a SimpleNamespace without hybrid_mlp_ratio / hybrid_attention_ratio
# (see primus/configs/models/megatron/zebra_llama_*.yaml — ratio often omitted, defaults to 0 in Megatron).
_ZEBRA_FLOPS="primus/backends/megatron/patches/zebra_llama_flops_patches.py"
if [[ -f "$_ZEBRA_FLOPS" ]]; then
  python3 << 'PY'
from pathlib import Path

path = Path("primus/backends/megatron/patches/zebra_llama_flops_patches.py")
text = path.read_text()
old = """        if args.hybrid_override_pattern:
            counts = {"M": 0, "*": 0, "-": 0}
            for layer_type in args.hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts["*"], counts["M"], counts["-"]
        else:
            num_attn_layers = round(args.num_layers * args.hybrid_attention_ratio)
            num_mlp_layers = round(args.num_layers * args.hybrid_mlp_ratio)"""
new = """        hybrid_override_pattern = getattr(args, "hybrid_override_pattern", None)
        if hybrid_override_pattern:
            counts = {"M": 0, "*": 0, "-": 0}
            for layer_type in hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts["*"], counts["M"], counts["-"]
        else:
            hybrid_ar = getattr(args, "hybrid_attention_ratio", 0.0)
            hybrid_mlp_r = getattr(args, "hybrid_mlp_ratio", 0.0)
            num_attn_layers = round(args.num_layers * hybrid_ar)
            num_mlp_layers = round(args.num_layers * hybrid_mlp_r)"""
if old not in text:
    print("[INFO] Zebra-Llama FLOPs patch: already applied or source mismatch, skipping")
else:
    path.write_text(text.replace(old, new, 1))
    print("[INFO] Zebra-Llama FLOPs patch applied (getattr for hybrid ratios / override pattern)")
PY
else
  echo "[WARN] Zebra-Llama FLOPs patch: $_ZEBRA_FLOPS not found, skipping"
fi
