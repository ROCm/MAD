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

# Apply metrics parser patch to primus-cli-direct.sh
git apply --check - <<'PATCH_EOF' 2>/dev/null && PATCH_APPLICABLE=1 || PATCH_APPLICABLE=0
diff --git a/runner/primus-cli-direct.sh b/runner/primus-cli-direct.sh
index b58a1527..2d9882d6 100755
--- a/runner/primus-cli-direct.sh
+++ b/runner/primus-cli-direct.sh
@@ -551,4 +551,77 @@ else
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
+if [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]]; then
+    if [[ "$FRAMEWORK" == "megatron" ]]; then
+        LOG_INFO "[direct] Using Megatron log parser"
+
+        num_warmup=$(grep -m1 'lr_warmup_iters' "${TRAIN_LOG}" | sed -En 's/.*lr_warmup_iters[^:]*:[[:space:]]*([0-9,]+).*/\1/p' | tr -d ',' 2>/dev/null)
+        num_warmup="${num_warmup:-0}"
+        echo "Num warmup: $num_warmup"
+
+        avg_tps=$(sed -En '/iteration  *[0-9].*tokens per GPU/s/.*iteration  *([0-9]+).*tokens per GPU \(tokens\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_tflops=$(sed -En '/iteration  *[0-9].*throughput per GPU/s/.*iteration  *([0-9]+).*throughput per GPU \(TFLOP\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_mem_pct=$(sed -En '/iteration  *[0-9].*usage_ratio/s/.*iteration  *([0-9]+).*usage_ratio:  *[^/]*\/[^/]*\/[^/]*\/([0-9.]+)%.*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
+
+        avg_elapsed_time=$(sed -En '/iteration  *[0-9].*elapsed time per iteration/s/.*iteration  *([0-9]+).*elapsed time per iteration \(ms\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
+
+        echo "Harmonic mean of TPS (excluding first $num_warmup steps): $avg_tps" | tee -a "${TRAIN_LOG}"
+        echo "Harmonic mean of TFLOPS (excluding first $num_warmup steps): $avg_tflops" | tee -a "${TRAIN_LOG}"
+        echo "Arithmetic mean of memory percentage (excluding first $num_warmup steps): $avg_mem_pct" | tee -a "${TRAIN_LOG}"
+        echo "Arithmetic mean of elapsed time (ms) (excluding first $num_warmup steps): $avg_elapsed_time" | tee -a "${TRAIN_LOG}"
+
+    elif [[ "$FRAMEWORK" == "torchtitan" ]]; then
+        LOG_INFO "[direct] Using Torchtitan log parser"
+
+        num_warmup=$(grep 'lr_scheduler.warmup_steps' "${TRAIN_LOG}" | sed -En 's/.*lr_scheduler.warmup_steps[^:]*:[[:space:]]*([0-9,]+).*/\1/p' | tr -d ',' 2>/dev/null)
+        num_warmup="${num_warmup:-0}"
+        echo "Num warmup (first steps skipped): $num_warmup"
+
+        avg_tps=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*tps:  *([0-9,]+).*/\1 \2/p' | tr -d ',' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_tflops=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*tflops:  *([0-9,.]+).*/\1 \2/p' | tr -d ',' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_mfu=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*mfu:  *([0-9,.]+)%.*/\1 \2/p' | tr -d ',' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
+
+        avg_mem=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*memory:  *[0-9.,]+GiB\(([0-9.]+)%\).*/\1 \2/p' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
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

if [[ "$PATCH_APPLICABLE" -eq 1 ]]; then
  git apply - <<'PATCH_EOF'
diff --git a/runner/primus-cli-direct.sh b/runner/primus-cli-direct.sh
index b58a1527..2d9882d6 100755
--- a/runner/primus-cli-direct.sh
+++ b/runner/primus-cli-direct.sh
@@ -551,4 +551,77 @@ else
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
+if [[ -n "$TRAIN_LOG" && -f "$TRAIN_LOG" ]]; then
+    if [[ "$FRAMEWORK" == "megatron" ]]; then
+        LOG_INFO "[direct] Using Megatron log parser"
+
+        num_warmup=$(grep 'lr_warmup_iters' "${TRAIN_LOG}" | sed -En 's/.*lr_warmup_iters[^:]*:[[:space:]]*([0-9,]+).*/\1/p' | tr -d ',' 2>/dev/null)
+        num_warmup="${num_warmup:-0}"
+        echo "Num warmup: $num_warmup"
+
+        avg_tps=$(sed -En '/iteration  *[0-9].*tokens per GPU/s/.*iteration  *([0-9]+).*tokens per GPU \(tokens\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_tflops=$(sed -En '/iteration  *[0-9].*throughput per GPU/s/.*iteration  *([0-9]+).*throughput per GPU \(TFLOP\/s\/GPU\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_mem_pct=$(sed -En '/iteration  *[0-9].*usage_ratio/s/.*iteration  *([0-9]+).*usage_ratio:  *[^/]*\/[^/]*\/[^/]*\/([0-9.]+)%.*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
+
+        avg_elapsed_time=$(sed -En '/iteration  *[0-9].*elapsed time per iteration/s/.*iteration  *([0-9]+).*elapsed time per iteration \(ms\):  *([0-9.]+).*/\1 \2/p' "${TRAIN_LOG}" \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
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
+          | sed -En 's/.*step:  *([0-9]+).*tps:  *([0-9,]+).*/\1 \2/p' | tr -d ',' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_tflops=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*tflops:  *([0-9,.]+).*/\1 \2/p' | tr -d ',' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += 1/$2; count++ } END { if (count == 0 || sum == 0) print "N/A"; else printf "%.2f", count/sum }')
+
+        avg_mfu=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*mfu:  *([0-9,.]+)%.*/\1 \2/p' | tr -d ',' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
+
+        avg_mem=$(grep 'rank-0.*INFO.*step:' "${TRAIN_LOG}" \
+          | sed -En 's/.*step:  *([0-9]+).*memory:  *[0-9.,]+GiB\(([0-9.]+)%\).*/\1 \2/p' \
+          | awk -v warmup="$num_warmup" '$1+0 > warmup && $2+0 > 0 { sum += $2; count++ } END { if (count == 0) print "N/A"; else printf "%.4f", sum/count }')
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
  echo "[INFO] Metrics parser patch applied successfully to runner/primus-cli-direct.sh"
else
  echo "[WARN] Metrics parser patch could not be applied (already applied or context mismatch), skipping"
fi
