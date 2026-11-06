import os
import json,time,urllib.request,sys,threading
from concurrent.futures import ThreadPoolExecutor
URL=os.environ.get("ROUTER_URL","http://127.0.0.1:30000")
WORD="the quick brown fox jumps over lazy dogs and cats near rivers "
def mk_prompt(in_tok):
    words=int(in_tok/0.75); f=(WORD*(words//11+1)); return " ".join(f.split()[:words])
def one(in_tok,out_tok):
    body=json.dumps({"model":"kimi-k3","messages":[{"role":"user","content":mk_prompt(in_tok)}],"max_tokens":out_tok,"temperature":0,"stream":True,"chat_template_kwargs":{"thinking":False}}).encode()
    t0=time.time(); ttft=None; ntok=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=600)
        for line in r:
            line=line.decode().strip()
            if line.startswith("data:") and "[DONE]" not in line:
                if ttft is None: ttft=time.time()-t0
                ntok+=1
        e2e=time.time()-t0
        return (ttft or e2e, e2e, ntok)
    except Exception as e: return (None,time.time()-t0,-1)
def run(in_tok,out_tok,con):
    t0=time.time()
    with ThreadPoolExecutor(max_workers=con) as ex:
        res=list(ex.map(lambda _: one(in_tok,out_tok), range(con)))
    wall=time.time()-t0
    ok=[r for r in res if r[2]>0]
    if not ok: print("  in=%d out=%d con=%d: ALL FAILED"%(in_tok,out_tok,con)); return
    ttfts=sorted(r[0] for r in ok); e2es=sorted(r[1] for r in ok); toks=sum(r[2] for r in ok)
    print("  in=%d/out=%d con=%d: ok=%d/%d wall=%.1fs | TTFT p50=%.1f p99=%.1f | e2e p50=%.1f p99=%.1f | out_tps_agg=%.1f | tot_tps=%.1f"%(
        in_tok,out_tok,con,len(ok),con,wall,ttfts[len(ttfts)//2],ttfts[-1],e2es[len(e2es)//2],e2es[-1],toks/wall,(toks+in_tok*len(ok))/wall)); sys.stdout.flush()
print("=== perf sweep (streaming; TTFT/e2e/throughput) ===")
for (i,o) in [(8000,1000),(16000,1000)]:
    for con in [16,32]:
        run(i,o,con)
