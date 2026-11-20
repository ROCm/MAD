#!/bin/bash
# The per-node log files are a contract: RCCL CI and other consumers parse
# /run_logs/<job>/<role>_NODE<rank>.log (host: $LOG_PATH/<job>/...). This pins every
# name and where it is written, so a change to one shows up here and is made on purpose.
#   1. From the snapshot cases (scripts/common/tests/snapshot/cases: one per path through the
#      launcher code): on every rank, LOG_PATH is mounted at /run_logs, the container gets
#      SLURM_JOB_ID=<job> and NODE_RANK=<rank>, and the node program tees to
#      /run_logs/<job>/<launcher log>_NODE<rank>.log. snapshot.py --equivalence
#      spur-array / k8s-indexed / --spooled / --spec-file compare every card under those
#      dispatchers with its slurm-srun run, so the names hold for every dispatcher.
#   2. In the role scripts, which run inside the container, the exact expressions
#      that name the role logs, as on ROCm/MAD develop (vllm_dissag, sglang_disagg)
#      and as the colocated launcher was added (vllm_multinode).
# See "Log file names are a contract" in docs/multinode-running.md.
# No cluster, no GPUs, no docker.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pass=0; fail=0
_ok()  { printf "  PASS  %s\n" "$1"; pass=$((pass+1)); }
_bad() { printf "  FAIL  %s\n" "$1"; fail=$((fail+1)); }

echo "=== launcher logs, per rank, from the snapshot cases ==="
out="$(python3 - "$S/common/tests/snapshot/cases" <<'PY'
import glob, json, os, re, sys
# The log each node program tees its container's output to, per catalog.
LAUNCHER_LOG = {"vllm_dissag": "pd_vllm_bench", "sglang_disagg": "pd_sglang_bench_serving.sh",
                "vllm_multinode": "colocated_bench"}
cases = sys.argv[1]
cards, problems = {}, []
for path in sorted(glob.glob(os.path.join(cases, "*", "*.json"))):
    catalog = os.path.basename(os.path.dirname(path))
    d = json.load(open(path))["snapshot"]
    if not d.get("ranks"):
        continue                                   # a card that stops before any node runs
    cards[catalog] = cards.get(catalog, 0) + 1
    for r in d["ranks"]:
        rank, dk, name = str(r["rank"]), r["docker"], f"{catalog}/{d['card']} rank {r['rank']}"
        job = next((e.split("=", 1)[1] for e in dk["env"] if e.startswith("SLURM_JOB_ID=")), None)
        if "<WORK>/logs:/run_logs" not in dk["mounts"]:
            problems.append(f"{name}: LOG_PATH is not mounted at /run_logs")
        if f"NODE_RANK={rank}" not in dk["env"]:
            problems.append(f"{name}: NODE_RANK is not the node's rank")
        want = f"/run_logs/{job}/{LAUNCHER_LOG[catalog]}_NODE{rank}.log"
        logs = set(re.findall(r"/run_logs/[^\s\"']*_NODE\d+\.log", " ".join(dk["command"])))
        if logs != {want}:
            problems.append(f"{name}: tees to {sorted(logs)}, want {want}")
        for b in (r.get("engine") or {}).get("blocks", []):
            h = b.get("header", "")
            if f"NODE_RANK={rank}=" not in h:
                problems.append(f"{name}: role script runs as another rank ({h})")
            m = re.search(r"log_prefix=(\S+)", h)
            if m and m.group(1) not in ("prefill", "decode"):
                problems.append(f"{name}: role log prefix {m.group(1)} (prefill_/decode_NODE<rank>.log)")
for c in LAUNCHER_LOG:
    print(f"cards {c} {cards.get(c, 0)}")
for p in problems:
    print(f"problem {p}")
PY
)"
for c in vllm_dissag sglang_disagg vllm_multinode; do
    n="$(sed -n "s/^cards $c //p" <<<"$out")"
    [ "${n:-0}" -gt 0 ] && _ok "$c: $n card(s) checked" || _bad "$c: no card checked (cases missing?)"
done
if grep -q '^problem ' <<<"$out"; then
    while IFS= read -r p; do _bad "${p#problem }"; done < <(grep '^problem ' <<<"$out")
else
    _ok "every rank: /run_logs mounted, NODE_RANK = rank, <launcher log>_NODE<rank>.log"
fi

echo ""
echo "=== role logs named in the role scripts ==="
_expr() { # file expression
    if grep -qF -- "$2" "$S/$1"; then _ok "$1: $2"; else _bad "$1: $2 (renamed or moved?)"; fi
}
for f in vllm_dissag/node.sh sglang_disagg/node.sh vllm_multinode/node.sh; do
    _expr "$f" ':/run_logs'
    _expr "$f" '-e NODE_RANK=$SLURM_PROCID'
done
_expr vllm_dissag/node.sh            '/run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${SLURM_PROCID}.log'
_expr sglang_disagg/node.sh          '/run_logs/${SLURM_JOB_ID}/pd_sglang_bench_serving.sh_NODE${SLURM_PROCID}.log'
_expr vllm_multinode/node.sh         '/run_logs/${SLURM_JOB_ID}/colocated_bench_NODE${SLURM_PROCID}.log'
for c in vllm_dissag/connectors/moriio.sh vllm_dissag/connectors/rixl.sh; do
    _expr "$c" '/run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log'
done
_expr vllm_dissag/connectors/moriio.sh '/run_logs/${SLURM_JOB_ID}/prefill_NODE0.log'
_expr vllm_dissag/connectors/moriio.sh '/run_logs/${SLURM_JOB_ID}/decode_NODE${xP}.log'
_expr vllm_dissag/vllm_disagg.sh     '/run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log'
_expr vllm_dissag/vllm_disagg.sh     '/run_logs/${SLURM_JOB_ID}/agentic_models_shim_NODE${NODE_RANK}.log'
_expr vllm_dissag/connectors/rixl.sh '/run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log'
for r in prefill decode proxy; do
    _expr sglang_disagg/sglang_disagg_mori_io_ep.sh "/run_logs/\${SLURM_JOB_ID:-0}/${r}_NODE\${NODE_RANK}.log"
done
_expr vllm_multinode/serve_colocated.sh 'RUN_LOGS="${_RUN_LOGS:-/run_logs}"'
_expr vllm_multinode/serve_colocated.sh '${RUN_LOGS}/${SLURM_JOB_ID:-0}/colocated_NODE${NODE_RANK}.log'

echo "======================================================"
echo "  log_names_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
