#!/bin/bash
# =============================================================================
# Throughput benchmark for the GLM-5.2 ATOM + mooncake-TCP P/D deployment.
# Drives the atomesh router (OpenAI-compatible) across an ISL/OSL x concurrency
# sweep. Assumes serve_atom.sh prefill+decode+router are already up.
#
# Usage:
#   ./bench_pd.sh                                  # default: 1k/1k @ conc 1,8,32
#   SHAPES="8192,1024" CONCS="8 16" ./bench_pd.sh
#   ROUTER_IP=10.0.0.1 ROUTER_PORT=30000 MODEL=/models/GLM-5.2-MXFP4 ./bench_pd.sh
# =============================================================================
set -uo pipefail
MODEL="${MODEL:-/models/GLM-5.2-MXFP4}"
ROUTER_IP="${ROUTER_IP:-$(ip route get 1.1.1.1 2>/dev/null | awk '/src/{print $7}')}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
SHAPES="${SHAPES:-1024,1024}"       # comma-separated "isl,osl"; space-separate multiple shapes
CONCS="${CONCS:-1 8 32}"
LOG="${LOG:-$PWD/logs/bench}"; mkdir -p "$LOG"; LOG="$(cd "$LOG" && pwd)"

URL="http://$ROUTER_IP:$ROUTER_PORT/v1/chat/completions"
echo "[bench] router=$URL shapes='$SHAPES' concs='$CONCS' $(date)" | tee "$LOG/bench.log"

for shape in $SHAPES; do
  ISL="${shape%,*}"; OSL="${shape#*,}"
  for c in $CONCS; do
    echo "[bench] --- isl=$ISL osl=$OSL conc=$c ---" | tee -a "$LOG/bench.log"
    URL="$URL" MODEL="$MODEL" ISL="$ISL" OSL="$OSL" CONC="$c" \
    RESULT="$LOG/bench_isl${ISL}_osl${OSL}_conc${c}.json" \
    python3 - <<'PY' 2>&1 | tee -a "$LOG/bench.log"
import json, os, time, threading, urllib.request, statistics
URL=os.environ["URL"]; MODEL=os.environ["MODEL"]
ISL=int(os.environ["ISL"]); OSL=int(os.environ["OSL"]); CONC=int(os.environ["CONC"])
# ~ISL-token prompt from a repeated block (~44 tok/block); vary per request id.
BLOCK=("In distributed machine learning inference, disaggregated prefill and decode "
"serving separates the compute-bound prefill stage from the memory-bandwidth-bound "
"decode stage across distinct hardware pools. ")
reps=max(1, ISL//44)
def prompt(rid): return "".join(f"[doc {rid} seg {i}] "+BLOCK for i in range(reps))+f"\n\nSummarize (req {rid})."
res=[]; lock=threading.Lock()
def worker(rid):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":prompt(rid)}],
                     "max_tokens":OSL,"temperature":0}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json"})
    t0=time.time()
    try:
        with urllib.request.urlopen(req,timeout=1200) as r: d=json.loads(r.read())
        u=d.get("usage",{})
        with lock: res.append((time.time()-t0,u.get("prompt_tokens"),u.get("completion_tokens"),u.get("ttft_s"),None))
    except Exception as e:
        with lock: res.append((time.time()-t0,None,None,None,str(e)[:120]))
T0=time.time()
ths=[threading.Thread(target=worker,args=(i,)) for i in range(CONC)]
for t in ths: t.start()
for t in ths: t.join()
WALL=time.time()-T0
ok=[r for r in res if r[4] is None]; err=[r for r in res if r[4] is not None]
out={"isl":ISL,"osl":OSL,"conc":CONC,"wall_s":round(WALL,2),"success":len(ok),"errors":len(err)}
if ok:
    cts=[r[2] for r in ok if r[2]]; pts=[r[1] for r in ok if r[1]]; lats=[r[0] for r in ok]
    ttfts=[r[3] for r in ok if r[3]]
    out.update(output_tok_s=round(sum(cts)/WALL,1), total_tok_s=round((sum(pts)+sum(cts))/WALL,1),
               mean_lat_s=round(statistics.mean(lats),2), p50_lat_s=round(statistics.median(lats),2))
    if ttfts: out["mean_ttft_s"]=round(statistics.mean(ttfts),3)
print(f"Successful: {len(ok)}/{CONC}  output_tok/s={out.get('output_tok_s')}  "
      f"total_tok/s={out.get('total_tok_s')}  mean_ttft={out.get('mean_ttft_s')}s  p50_lat={out.get('p50_lat_s')}s")
if err: print(f"  errors={len(err)} (first: {err[0][4]})")
open(os.environ["RESULT"],"w").write(json.dumps(out,indent=2))
PY
  done
done
echo "[bench] DONE $(date)  results in $LOG" | tee -a "$LOG/bench.log"
