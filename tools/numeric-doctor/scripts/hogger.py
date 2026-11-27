#!/usr/bin/env python3
"""Occupy GPU #3 for two hours doing nothing in particular.

The point is to hold the device, not to compute: a CUDA/HIP context plus a live
allocation is what actually keeps another process off the GPU, so the loop does the
smallest possible amount of work and sleeps between iterations.  Use it to park a node
(or one GPU of a node) while you set something else up.

    python hogger.py                       # GPU 3, 2 hours
    python hogger.py --hours 0.5           # 30 minutes
    python hogger.py --reserve-gb 64       # also pin 64 GiB of HBM
    python hogger.py --busy                # no sleep -- keep the SIMDs warm too

Note on visibility: HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES renumber the devices torch
can see.  `replay_hipblaslt_log.sbatch` sets HIP_VISIBLE_DEVICES=0, which leaves torch with
exactly one device and no GPU #3 -- unset it, or set it to 3, before running this.
"""

import argparse
import os
import signal
import sys
import time

DEVICE_INDEX = 3          # the skill asks for GPU #3 specifically
DEFAULT_HOURS = 2.0

_stop = False


def _on_signal(signum, _frame):
    global _stop
    _stop = True
    print(f"\n[hogger] caught {signal.Signals(signum).name}, releasing the GPU", flush=True)


def fmt(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                    help=f"how long to hold the GPU (default: {DEFAULT_HOURS})")
    ap.add_argument("--device", type=int, default=DEVICE_INDEX,
                    help=f"GPU index to occupy (default: {DEVICE_INDEX})")
    ap.add_argument("--reserve-gb", type=float, default=0.0,
                    help="also allocate this much HBM, to keep other jobs out of it")
    ap.add_argument("--busy", action="store_true",
                    help="do not sleep between iterations (hotter, still trivial work)")
    ap.add_argument("--sleep", type=float, default=0.1,
                    help="seconds to sleep per iteration unless --busy (default: 0.1)")
    ap.add_argument("--report-every", type=float, default=300.0,
                    help="seconds between progress lines (default: 300)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    import torch

    if not torch.cuda.is_available():
        sys.exit("no GPU visible to torch (need /dev/kfd and /dev/dri in the container)")

    n = torch.cuda.device_count()
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if n <= args.device:
        sys.exit(f"torch sees {n} GPU(s); GPU #{args.device} needs more than {args.device}."
                 + (f"\nHIP/CUDA_VISIBLE_DEVICES={visible} is hiding the rest -- unset it, "
                    f"or set it to {args.device}." if visible else ""))

    torch.cuda.set_device(args.device)
    dev = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(dev)
    print(f"[hogger] pid {os.getpid()} on {os.uname().nodename}")
    print(f"[hogger] torch {torch.__version__}  hip {getattr(torch.version, 'hip', None)}")
    print(f"[hogger] {n} GPU(s) visible"
          + (f" (HIP/CUDA_VISIBLE_DEVICES={visible})" if visible else "")
          + f"; holding cuda:{args.device} = {props.name}, "
            f"{props.total_memory / 2**30:.0f} GiB")

    # The allocation is the part that actually reserves the device; the loop below just
    # keeps the context alive and gives us something to sync on.
    ballast = None
    if args.reserve_gb > 0:
        elems = int(args.reserve_gb * 2**30 // 4)
        ballast = torch.empty(elems, dtype=torch.float32, device=dev)
        ballast.zero_()
        print(f"[hogger] reserved {args.reserve_gb:.1f} GiB of HBM")

    x = torch.zeros(1, device=dev)          # the dummy work: one element, incremented

    total = args.hours * 3600.0
    t0 = time.monotonic()
    deadline = t0 + total
    next_report = t0
    iters = 0

    print(f"[hogger] holding for {fmt(total)} (until "
          f"{time.strftime('%H:%M:%S', time.localtime(time.time() + total))}); "
          f"Ctrl-C or SIGTERM to stop early", flush=True)

    while not _stop:
        now = time.monotonic()
        if now >= deadline:
            break
        x += 1                              # trivial kernel, nothing meaningful computed
        if not args.busy:
            time.sleep(min(args.sleep, deadline - now))
        iters += 1
        if now >= next_report:
            torch.cuda.synchronize(dev)     # prove the device is still answering
            print(f"[hogger] {fmt(now - t0)} elapsed, {fmt(deadline - now)} left, "
                  f"{iters} iters, "
                  f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB allocated",
                  flush=True)
            next_report = now + args.report_every

    torch.cuda.synchronize(dev)
    held = time.monotonic() - t0
    del ballast, x
    torch.cuda.empty_cache()
    print(f"[hogger] done: held cuda:{args.device} for {fmt(held)} over {iters} iters"
          + (" (stopped early)" if _stop else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
