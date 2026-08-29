import json,urllib.request,time,threading
import os
URL=os.environ.get("ENDPOINT","http://127.0.0.1:2322")+"/v1/completions"; M="/models/DeepSeek-V4-Flash-FP8-E4M3"
F="The quick brown fox jumps over the lazy dog. "
def make(isl): 
    reps=isl//9; return (F*reps)[:isl*5]  # approx isl tokens
def one(prompt,osl,res,i):
    t0=time.time()
    data=json.dumps({"model":M,"prompt":prompt,"max_tokens":osl,"temperature":0,"stream":False}).encode()
    r=urllib.request.Request(URL,data=data,headers={"Content-Type":"application/json"})
    try:
        d=json.load(urllib.request.urlopen(r,timeout=300)); dt=time.time()-t0
        ct=d["usage"]["completion_tokens"]; res[i]=(dt,ct)
    except Exception as e: res[i]=(None,str(e)[:40])
def run(isl,osl,con):
    prompt=make(isl)
    res=[None]*con; ths=[]
    t0=time.time()
    for i in range(con):
        th=threading.Thread(target=one,args=(prompt,osl,res,i)); th.start(); ths.append(th)
    for th in ths: th.join()
    wall=time.time()-t0
    ok=[r for r in res if r and r[0]]
    if not ok: print(f"ISL={isl} OSL={osl} CON={con}: ALL FAILED {res[0]}"); return
    lat=sum(r[0] for r in ok)/len(ok); toks=sum(r[1] for r in ok)
    print(f"ISL={isl} OSL={osl} CON={con}: {len(ok)}/{con} ok | mean_latency={lat:.1f}s | out_tok={toks} | wall={wall:.1f}s | out_tps={toks/wall:.0f} tok/s",flush=True)
for isl in [8192,16384]:
    for con in [16,32]:
        run(isl,1024,con)
