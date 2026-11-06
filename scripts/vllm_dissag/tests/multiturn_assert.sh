#!/bin/bash
# Offline checks for the GLM-5.3 MXFP4 recipe and the warm multi-turn benchmark path.
# No cluster / no GPUs / no transformers. Exits 0 if all assertions hold.
#
# Covers:
#   - GLM-5.3-Quark-MXFP4-AttnFP8 moriio+wideEP argv per role (block 16, KV bytes, MTP-3,
#     MoRI-IO READ, per-role MNBT and MoRI-EP backend, prefix caching on)
#   - MORIIO_READ_MODE stays opt-in: DeepSeek-V3's kv-transfer-config has no read_mode
#   - benchmark_multiturn_retained.sh DRY_RUN plan (presets, exact OSL, session ids, corpus SHA)
#   - session shape math and sticky-id balancing (benchmark_multiturn_retained.py)
#   - multiturn_metrics.py on a synthetic run: round-1 exclusion, tok/s/GPU, goodput gates
#   - KV-transfer probe pass/fail logic; NIAH summary rows
#   - parse_to_csv.py --result-json rows; long_context argv unchanged unless opted in
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pass=0; fail=0
_has()   { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot(){ grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }
_flag()  { # argv-per-line, flag, value, label: the line after <flag> equals <value>
  awk -v f="$2" 'p {print; exit} $0 == f {p=1}' <<<"$1" | grep -qxF -- "$3" \
    && { printf "  PASS  %s\n" "$4"; pass=$((pass+1)); } || { printf "  FAIL  %s (%s != %s)\n" "$4" "$2" "$3"; fail=$((fail+1)); }; }

_argv() { # node_rank model
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK="$1" xP=1 yD=1 CONNECTOR=moriio WIDE_EP=1 EP_BACKEND=mori \
    MODEL_NAME="$2" MODEL_PATH=/m/model MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}

echo "=== GLM-5.3 MXFP4 prefill (NODE_RANK=0) ==="
P="$(_argv 0 GLM-5.3-Quark-MXFP4-AttnFP8)"
_flag   "$P" "--block-size" "16" "block size 16"
_flag   "$P" "--kv-cache-dtype" "auto" "kv-cache-dtype auto (BF16 latent + FP8 indexer)"
_flag   "$P" "--kv-cache-memory-bytes" "80000000000" "80e9 KV bytes per GPU"
_flag   "$P" "--speculative-config" '{"method":"mtp","num_speculative_tokens":3}' "MTP-3"
_flag   "$P" "--max-num-batched-tokens" "16384" "prefill MNBT 16384"
_flag   "$P" "--max-num-seqs" "64" "max-num-seqs 64"
_flag   "$P" "--max-model-len" "560000" "max-model-len 560000"
_flag   "$P" "--all2all-backend" "mori_high_throughput" "prefill MoRI-EP high throughput"
_flag   "$P" "--data-parallel-size" "8" "DP8"
_has    "$P" '"read_mode":"true"' "MoRI-IO READ"
_has    "$P" '"defer_timeout":"1800","recv_abort_timeout":"600","qp_per_transfer":"2","num_workers":"4"}}' "tested MoRIIO extra config"
_has    "$P" '"kv_role":"kv_producer"' "prefill is kv_producer"
_hasnot "$P" "--no-enable-prefix-caching" "prefix caching on"

echo ""
echo "=== GLM-5.3 MXFP4 decode (NODE_RANK=1) ==="
D="$(_argv 1 GLM-5.3-Quark-MXFP4-AttnFP8)"
_flag   "$D" "--max-num-batched-tokens" "256" "decode MNBT 256"
_flag   "$D" "--all2all-backend" "mori_low_latency" "decode MoRI-EP low latency"
_has    "$D" '"cudagraph_mode":"FULL_AND_PIECEWISE"' "decode FULL_AND_PIECEWISE"
_has    "$D" '"read_mode":"true"' "MoRI-IO READ"
_has    "$D" '"kv_role":"kv_consumer"' "decode is kv_consumer"

echo ""
echo "=== READ mode is opt-in (DeepSeek-V3 unchanged) ==="
B="$(_argv 0 DeepSeek-V3)"
_has    "$B" '"notify_port":"61005"}}' "DSV3 kv-transfer-config ends at notify_port"
_hasnot "$B" "read_mode" "DSV3 has no read_mode"

echo ""
echo "=== benchmark_multiturn_retained.sh DRY_RUN plan ==="
_plan() { env -i PATH="$PATH" HOME="$HOME" DRY_RUN=1 MODEL_PATH=/m/model xP=1 yD=1 GPUS_PER_NODE=8 "$@" \
            bash "$DIR/benchmark_multiturn_retained.sh" 2>&1; }
M="$(_plan MT_PRESET=H256-30)"
_has "$M" "--isl 286000 --new-tokens 30000 --turns 3 --conversations 16" "H256-30: isl 256k+30k, 16 sessions"
_has "$M" "--limit-min-tokens 1000 --limit-max-tokens 1000" "exact 1000-token output"
_has "$M" "--send-conversation-id" "sends X-Session-ID"
_has "$M" "--sticky-mod 8" "sessions balanced over 8 ranks"
_has "$M" "--gpus 16" "tok/s/GPU over both roles (16 GPUs)"
_has "$M" "99245b8f3f6b637a3670d37305bb1ba52003f05797b6638571562999fd623f9c" "corpus SHA256 pinned"
_has "$(_plan MT_PRESET=M512-30)" "--isl 542000 --new-tokens 30000 --turns 3 --conversations 8" "M512-30 preset"
_has "$(_plan MT_PRESET=R128-10)" "--isl 138000 --new-tokens 10000 --turns 3 --conversations 32" "R128-10 preset"
_has "$(_plan MT_PRESET=R128-30)" "--isl 158000 --new-tokens 30000 --turns 3 --conversations 32" "R128-30 preset"
_has "$(_plan MT_PRESET=R128-70)" "--isl 198000 --new-tokens 70000 --turns 3 --conversations 16" "R128-70 preset"
_has "$(_plan MT_PRESET=R128-70)" "--ttft-slo-ms 12800" "128k TTFT target 12.8 s (retained / 10k tok/s)"
_has "$M" "--ttft-slo-ms 25600" "256k TTFT target 25.6 s"
_has "$(_plan MT_PRESET=M512-30)" "--ttft-slo-ms 51200" "512k TTFT target 51.2 s"
_has "$(_plan MT_RETAINED_TOKENS=64000 MT_NEW_TOKENS=2000 BENCHMARK_CON=4)" "--isl 66000 --new-tokens 2000 --turns 3 --conversations 4" "explicit shape overrides preset"
_has "$(_plan MT_PRESET=bogus)" "unknown MT_PRESET" "bad preset rejected"

echo ""
echo "=== session shapes and sticky ids ==="
S="$(cd "$DIR" && python3 - <<'PY'
import benchmark_multiturn_retained as g
for isl, n in ((138000, 10000), (158000, 30000), (198000, 70000), (286000, 30000), (326000, 70000), (542000, 30000)):
    s = g.shape(isl, n, 3, 1000)
    # measured turns 2 and 3 average to the target ISL; last turn + OSL fits the model window
    turn2 = s["isl_warm"] + n + 1000
    print("mean_ok" if abs((turn2 + s["isl_last"]) / 2 - isl) <= 1 else "mean_bad",
          "fits" if s["isl_last"] + 1000 <= 560000 else "overflow")
try:
    g.shape(20000, 30000, 3, 1000); print("small_accepted")
except ValueError:
    print("small_rejected")
try:
    g.shape(560000, 30000, 3, 1000); print("long_accepted")
except ValueError:
    print("long_rejected")
ranks = sorted(g.sticky_hash(g.balanced_id(i, 8)) % 8 for i in range(16))
print("ranks", ranks)
print("plain", g.balanced_id(3, 0))
PY
)"
_has "$S" "mean_ok fits" "preset shapes hit the target ISL and fit 560k"
_hasnot "$S" "mean_bad" "no preset misses its target ISL"
_has "$S" "small_rejected" "ISL too small for new tokens rejected"
_has "$S" "long_rejected" "last turn beyond max-model-len rejected"
_has "$S" "ranks [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]" "16 sessions balanced 2 per rank"
_has "$S" "plain CONV_ID_3" "sticky-mod 0 gives plain ids"

echo ""
echo "=== multiturn_metrics.py on a synthetic 2-round run ==="
T="$(mktemp -d)"
python3 - "$T" <<'PY'
import json, os, sys
root = sys.argv[1]
# Decode ITL histogram deltas per round (le seconds -> cumulative count). Round 2 alone has
# p99 ~117 ms; round 1 (cold prefill) is slow, and including it would push p99 to 250 ms.
BUCKETS = {1: {"0.05": 10, "0.1": 20, "0.15": 30, "0.25": 40, "+Inf": 100},
           2: {"0.05": 90, "0.1": 98.5, "0.15": 100, "0.25": 100, "+Inf": 100}}
def prom(path, scale, r):
    b = {le: c * scale for le, c in BUCKETS[r].items()}
    with open(path, "w") as f:
        f.write("# EP role=prefill ep=p:1\nvllm:prefix_cache_queries_total 0\n")
        f.write("# EP role=decode ep=d:1\n")
        for le, c in b.items():
            f.write(f'vllm:inter_token_latency_seconds_bucket{{engine="0",le="{le}"}} {c}\n')
        f.write(f"vllm:prefix_cache_queries_total {1000 * scale}\nvllm:prefix_cache_hits_total {900 * scale}\n")
for r in (1, 2):
    rd = os.path.join(root, f"round-{r}"); os.makedirs(rd)
    prom(os.path.join(rd, "prom-before.prom"), 0, r)
    prom(os.path.join(rd, "prom-after.prom"), 1, r)
    stats = []
    for c in range(2):                       # 2 sessions x 2 measured turns, 1000 tokens in 10 s each
        for t in range(2):
            st = 100.0 * r + 20 * t
            stats.append(dict(conversation_id=f"s{c}", start_time_ms=st * 1e3, latency_ms=10_000,
                              ttft_ms=2000, tpot_ms=8, input_num_tokens=286000, output_num_tokens=1000))
    json.dump(stats, open(os.path.join(rd, "stats.json"), "w"))
PY
R="$(cd "$DIR" && python3 multiturn_metrics.py "$T" --mode multiturn --gpus 16 --isl 286000 --osl 1000 --t-meas 2 --layout pd)"
V="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(' '.join(f'{k}={r[k]}' for k in sys.argv[2:]))" "$T/result.json" \
     completed osl_exact_frac slo100_pass slo150_pass step_itl_p99_ms decode_prefix_cache_hit_rate)"
_has "$R" "MT_RESULT" "reducer prints MT_RESULT"
_has "$V" "completed=8" "all 8 measured requests counted"
_has "$V" "osl_exact_frac=1.0" "exact-OSL share"
_has "$V" "slo150_pass=True" "passes ITL p99 <= 150 ms (round 2 only)"
_has "$V" "slo100_pass=False" "fails ITL p99 <= 100 ms"
_has "$V" "decode_prefix_cache_hit_rate=0.9" "decode prefix-hit rate from counter deltas"
TOK="$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(round(r['out_tok_s_per_gpu'], 3), round(r['slo150_goodput_tok_s_per_gpu'], 3), r['slo100_goodput_tok_s_per_gpu'])" "$T/result.json")"
# steady window per round: first start .. first session done = 30 s; 2 sessions x 2000 tokens
# emitted across both 30 s windows -> 8000 tokens / 60 s / 16 GPUs
_has "$TOK" "8.333 8.333 0.0" "tok/s/GPU = window tokens / window / 16 GPUs; goodput gated"

(cd "$DIR" && python3 multiturn_metrics.py "$T" --mode multiturn --gpus 16 --isl 286000 --osl 1000 \
   --t-meas 2 --ttft-slo-ms 1000 --out "$T/slo.json" >/dev/null)
_has "$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['slo150_pass'], r['slo150_goodput_tok_s_per_gpu'])" "$T/slo.json")" \
     "False 0.0" "TTFT p90 (2 s) above a 1 s SLO zeroes goodput"

P2="$(cd "$DIR" && MAD_SYSTEM_GPU_ARCHITECTURE=gfx950 python3 parse_to_csv.py "$T/result.json" --result-json --perf-csv "$T/perf.csv" --model-name GLM-5.3-Quark-MXFP4-AttnFP8 --workload mt-H256-30 >/dev/null && cat "$T/perf.csv")"
_has "$P2" "output tok/s/GPU (mt-H256-30 isl=286000 osl=1000 gpus=16)" "perf.csv tok/s/GPU row"
_has "$P2" "goodput tok/s/GPU at ITL p99<=150ms" "perf.csv goodput row"
_has "$P2" "gfx950" "perf.csv gpu_architecture from MAD_SYSTEM_GPU_ARCHITECTURE"
rm -rf "$T"

echo ""
echo "=== KV-transfer probe checks and NIAH summaries ==="
RP="$(cd "$DIR" && python3 - <<'PY'
from benchmark_read_probe import evaluate
def d(p_gen, d_local, d_ext, engines):
    out = {("prefill", "p", "vllm:generation_tokens_total", 'engine="0"'): p_gen,
           ("decode", "d", "vllm:prompt_tokens_by_source_total", 'engine="0",source="local_compute"'): d_local,
           ("decode", "d", "vllm:prompt_tokens_by_source_total", 'engine="0",source="external_kv_transfer"'): d_ext}
    for e, v in engines.items():
        out[("decode", "d", "vllm:request_success_total", f'engine="{e}",finished_reason="stop"')] = v
    return out
good = {str(e): 4 for e in range(8)}
print("good", all(evaluate(d(32, 32, 32 * 40, good), 32)[0].values()))
print("local_prefill", evaluate(d(32, 32 * 40, 0, good), 32)[0]["decode_one_local_token_per_request"])
print("idle_rank", evaluate(d(32, 32, 32 * 40, {**good, "7": 0}), 32)[0]["every_decode_engine_served"])
print("long_prefill_leg", evaluate(d(64, 32, 32 * 40, good), 32)[0]["prefill_one_token_per_request"])
PY
)"
_has "$RP" "good True" "probe passes on a clean transfer"
_has "$RP" "local_prefill False" "probe fails when decode prefills locally"
_has "$RP" "idle_rank False" "probe fails when a decode rank served nothing"
_has "$RP" "long_prefill_leg False" "probe fails when prefill generates more than one token"
T2="$(mktemp -d)"
printf '%s\n' "niah_single lengths=65536,262144,524288 found=18/18 errors=0" "niah_mt start=246000 found=7/8 errors=0" > "$T2/niah.log"
NP="$(cd "$DIR" && python3 parse_to_csv.py "$T2/niah.log" --niah-mt --perf-csv "$T2/p.csv" --model-name m >/dev/null && cat "$T2/p.csv")"
_has "$NP" "100.0,\"percent found (niah single lengths=65536,262144,524288 asked=18)\"" "perf.csv NIAH single row"
_has "$NP" "87.5,percent found (niah multi-turn start=246000 asked=8)" "perf.csv multi-turn needle row"
rm -rf "$T2"

echo ""
echo "=== long_context argv unchanged unless opted in ==="
L="$(cat "$DIR/benchmark_long_context.sh")"
_has "$L" 'detail=()' "detailed flags start empty"
_has "$L" 'if [ "${LC_DETAILED:-0}" = "1" ]; then' "detailed output is opt-in"
_has "$L" '[ -n "${NUM_PROMPTS:-}" ] && n_prompts=$NUM_PROMPTS' "NUM_PROMPTS only overrides when set"

echo ""
echo "======================================================"
echo "  multiturn_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
