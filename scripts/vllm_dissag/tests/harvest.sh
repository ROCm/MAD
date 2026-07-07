#!/bin/bash
# harvest.sh <RUN_TAG> — print pass/fail + per-cell benchmark summary
RUN_TAG="$1"
D=/shared_inference/${USER}/model_blog_logs/${RUN_TAG}
echo "### $RUN_TAG ###"
[ -d "$D" ] || { echo "  no log dir yet"; exit 0; }
# readiness + errors
pc=$(grep -l 'Application startup complete' $D/prefill_NODE0.log 2>/dev/null | wc -l)
rdma=$(grep -h 'RegisterRdmaMemoryRegion failed' $D/*.log 2>/dev/null | wc -l)
died=$(grep -h 'died unexpectedly' $D/*.log 2>/dev/null | wc -l)
echo "  startup-complete(prefill)=$pc  rdma-fail=$rdma  worker-died=$died"
# benchmark cells
grep -aE 'Maximum request concurrency:|Successful requests:|Failed requests:|Output token throughput|Median TTFT|Median ITL' $D/*CONCURRENCY.log 2>/dev/null \
  | grep -vE "namespace|Namespace" \
  | awk '/Maximum request concurrency:/{c=$4} /Successful requests:/{s=$3} /Failed requests:/{f=$3} /Output token throughput/{o=$5} /Median TTFT/{t=$4} /Median ITL/{print "  con="c" ok="s" fail="f" tok/s="o" TTFT="t" ITL="$4}'
