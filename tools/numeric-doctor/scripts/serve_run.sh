#!/bin/bash
# One trial of a model that runs as a SERVER (`vllm serve`, `python3 -m sglang.launch_server`): start it, wait until
# it answers, send the requests, stop it -- the command a spec line runs (wave.sh, capture_to_repro.sh) or a diff
# wave's CMD (diff_wave.sh), in place of an offline driver (`step1-ask.md` *Servers*).
#
#   bash $ND_REPO/scripts/serve_run.sh --server '<the server command, with {port}>' --requests <requests.jsonl> \
#        [--api openai|sglang] [--port auto|N] [--ready-timeout 900] [-- <serve_client.py options>]
#
# {port} in the server command is replaced by the trial's port: auto = 18000 + 10 x the first visible GPU (+ the
# copy's MASTER_PORT offset), so trials on distinct GPUs, and copies of one multi-node job, never collide. The
# capture / graph / switch shims come in through the trial's PYTHONPATH and act in the server's own processes
# (worker, scheduler) as they do for an offline driver. The server is stopped with SIGINT, then SIGTERM after 60 s,
# then SIGKILL: a capture that stopped the engine has already written its capsules and manifest; a clean server gets
# to shut down.
# The client's lines are the trial's: `Logprobs are not finite at request <i>` (the signal), ND_DIFF_HIT, and the
# last line `SERVE_CLIENT_RESULT ...`. Exit: the client's (1 when every request failed), 3 when the server never
# became ready (its log tail printed), 4 when it died while the client ran.
set -u
server="" requests="" api=openai port=auto ready=900 extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --server) server=$2; shift 2 ;;
    --requests) requests=$2; shift 2 ;;
    --api) api=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --ready-timeout) ready=$2; shift 2 ;;
    --) shift; extra=("$@"); break ;;
    *) echo "serve_run.sh: unknown argument $1"; exit 2 ;;
  esac
done
[ -n "$server" ] && [ -n "$requests" ] || { echo "serve_run.sh: --server and --requests are required"; exit 2; }
if [ "$port" = auto ]; then
  # + the copy's offset: k copies of a multi-node job share every GPU, and wave.sh gives each its own MASTER_PORT
  g=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}; port=$((18000 + 10 * ${g%%,*} + ${MASTER_PORT:-29500} - 29500))
fi
cmd=${server//\{port\}/$port}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
echo "[serve_run] port=$port api=$api server: $cmd"
setsid bash -c "$cmd" &
spid=$!
stop() {
  kill -0 "$spid" 2>/dev/null || return 0
  kill -INT -- -"$spid" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$spid" 2>/dev/null || return 0; sleep 1; done
  kill -TERM -- -"$spid" 2>/dev/null; sleep 10; kill -KILL -- -"$spid" 2>/dev/null
}
trap 'stop; exit 130' TERM INT
t0=$(date +%s)
until python3 - "$port" <<'EOF' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen("http://127.0.0.1:%s/health" % sys.argv[1], timeout=5).read()
EOF
do
  if ! kill -0 "$spid" 2>/dev/null; then echo "[serve_run] the server exited before it was ready"; exit 3; fi
  if [ $(( $(date +%s) - t0 )) -gt "$ready" ]; then echo "[serve_run] not ready after ${ready}s"; stop; exit 3; fi
  sleep 3
done
echo "[serve_run] ready after $(( $(date +%s) - t0 ))s"
python3 "$here/serve_client.py" --url "http://127.0.0.1:$port" --requests "$requests" --api "$api" \
  "${extra[@]+"${extra[@]}"}"
rc=$?
alive=1; kill -0 "$spid" 2>/dev/null || alive=0
stop
wait "$spid" 2>/dev/null
[ "$alive" = 0 ] && { echo "[serve_run] the server died while the client ran"; exit 4; }
exit $rc
