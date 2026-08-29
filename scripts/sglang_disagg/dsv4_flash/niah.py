import json, urllib.request
import os
URL=os.environ.get("ENDPOINT","http://127.0.0.1:2322")+"/v1/completions"; M="/models/DeepSeek-V4-Flash-FP8-E4M3"
F="The quick brown fox jumps over the lazy dog. "  # ~9 tokens
def ask(p, mx=5):
    data=json.dumps({"model":M,"prompt":p,"max_tokens":mx,"temperature":0}).encode()
    r=urllib.request.Request(URL,data=data,headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["text"]
# token targets -> filler repeats (~9 tok each)
lengths={"1K":110,"4K":440,"16K":1780,"32K":3560,"100K":11100,"200K":22200}
pass_=tot=0
for name,reps in lengths.items():
    for d in [0.1,0.5,0.9]:
        pre=int(reps*d)
        p=F*pre+"Marie was born in the city of Paris. "+F*(reps-pre)+"Marie was born in the city of"
        try: t=ask(p); ok="paris" in t.lower()
        except Exception as e: t=f"ERR {str(e)[:40]}"; ok=False
        tot+=1; pass_+=ok
        print(f"{name} d={int(d*100)}% -> {t[:28]!r} [{'PASS' if ok else 'FAIL'}]",flush=True)
print(f"NIAH TOTAL: {pass_}/{tot}",flush=True)
