#!/bin/bash
# One perf point: fire CON concurrent ISL/OSL requests, record TTFT/e2e/http as JSON.
# Robust version: build the JSON body ONCE with python -> file, curl -d @file.
# Usage: ROUTER=http://127.0.0.1:30000 bash perf_point.sh ISL OSL CON PLATFORM OUTDIR
set -u
ISL="${1:?isl}"; OSL="${2:?osl}"; CON="${3:?con}"; PLAT="${4:?platform}"; OUT="${5:?outdir}"
ROUTER="${ROUTER:-http://127.0.0.1:30000}"
TAG="${PLAT}_2p2d_isl${ISL}_osl${OSL}_con${CON}"
BODY=/tmp/body_${TAG}.json
python3 - "$ISL" "$OSL" "$BODY" <<'PYEOF'
import sys,json
isl,osl,path=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
words="the quick brown fox jumps over the lazy dog . "
prompt=(words*(isl//9+2))[:isl*4]   # ~isl tokens
body={"model":"kimi-k3","messages":[{"role":"user","content":prompt}],
      "max_tokens":osl,"temperature":0,
      "chat_template_kwargs":{"thinking":False}}
open(path,"w").write(json.dumps(body))
PYEOF
echo "[$TAG] firing $CON @ $(date +%T)"
t0=$(python3 -c "import time;print(time.time())")
: > /tmp/res_${TAG}
for i in $(seq 1 "$CON"); do
  ( curl -s -m 1800 -o /dev/null -w "%{time_starttransfer} %{time_total} %{http_code}\n" \
      "$ROUTER/v1/chat/completions" -H "Content-Type: application/json" -d @"$BODY" >> /tmp/res_${TAG} ) &
done
wait
t1=$(python3 -c "import time;print(time.time())")
python3 - "$ISL" "$OSL" "$CON" "$TAG" "$t0" "$t1" /tmp/res_${TAG} > "$OUT/${TAG}.json" <<'PYEOF'
import sys,json
isl,osl,con,tag,t0,t1,f=sys.argv[1:8]
rows=[l.split() for l in open(f) if l.strip()]
ttft=[float(r[0]) for r in rows if len(r)>=3 and r[2]=="200"]
e2e=[float(r[1]) for r in rows if len(r)>=3 and r[2]=="200"]
codes=[r[2] for r in rows if len(r)>=3]
ok=len(e2e)
def stat(a):
    a=sorted(a); n=len(a)
    return {"min":round(a[0],2),"p50":round(a[n//2],2),"p90":round(a[min(n-1,int(n*0.9))],2),"max":round(a[-1],2),"mean":round(sum(a)/n,2)} if a else {}
wall=float(t1)-float(t0)
tpot=[round((e-t)/max(1,int(osl)),4) for e,t in zip(e2e,ttft)]
print(json.dumps({"tag":tag,"isl":int(isl),"osl":int(osl),"con":int(con),"ok":ok,"total":len(rows),
  "wall_s":round(wall,2),"ttft_s":stat(ttft),"e2e_s":stat(e2e),"tpot_s":stat(tpot),
  "agg_tok_s":round(ok*int(osl)/wall,1) if wall>0 else 0,"http_codes":codes},indent=2))
PYEOF
rm -f /tmp/res_${TAG} "$BODY"
echo "[$TAG] ->"; grep -E '"ok"|"wall_s"|"agg_tok_s"|"p50"' "$OUT/${TAG}.json" | head -6
