import os
import json,time,urllib.request,sys
URL=os.environ.get("ROUTER_URL","http://127.0.0.1:30000")
SIZES=[50000,100000,200000,280000]
DEPTHS=[0.1,0.5,0.9]
NEEDLE="HELIOTROPE-7492"
CHARS_PER_TOK=4
WORD="the quick brown fox jumps over lazy dogs and cats near rivers "
def mk(tok,depth):
    chars=tok*CHARS_PER_TOK
    filler=(WORD*(chars//len(WORD)+1))[:chars]
    cut=int(len(filler)*depth)
    doc=filler[:cut]+f" The magic keyword is {NEEDLE}. "+filler[cut:]
    return f"Read the following document carefully.\n\n{doc}\n\nQuestion: What is the magic keyword? Answer with only the keyword."
def ask(tok,depth):
    body=json.dumps({"model":"kimi-k3","messages":[{"role":"user","content":mk(tok,depth)}],
        "max_tokens":64,"temperature":0,
        "chat_template_kwargs":{"thinking":False}}).encode()
    t0=time.time()
    try:
        r=urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"}),timeout=600)
        j=json.loads(r.read()); m=j["choices"][0]["message"]
        full=(m.get("content") or "")+" "+(m.get("reasoning_content") or "")
        return (NEEDLE in full, round(time.time()-t0,1), j["choices"][0]["finish_reason"])
    except Exception as e: return (False, round(time.time()-t0,1), "ERR:"+str(e)[:30])
print("=== NIAH single-needle thinking=false MT=64 ===")
print("%-8s %-6s %-6s %-8s %s"%("size","depth","recall","secs","finish"))
for tok in SIZES:
    row=[]
    for d in DEPTHS:
        ok,secs,fin=ask(tok,d); row.append(ok)
        print("%-8d %-6.1f %-6s %-8.1f %s"%(tok,d,"PASS" if ok else "FAIL",secs,fin)); sys.stdout.flush()
    print("  -> %dK: %d/3\n"%(tok//1000,sum(row))); sys.stdout.flush()
