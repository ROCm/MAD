import os, json, time, sys, urllib.request, concurrent.futures as cf

# Concurrency NIAH accuracy probe — validates the con>1 KDA state-recycle fix.
# Each concurrent request carries its OWN distinct needle; we count how many recall
# their own needle. A correct serve returns N/N at every concurrency. Before the fix,
# sustained con=32 fell to ~65% (a recycled KDA/mamba state block handed a finished
# request's state to a new one). See README "Accuracy".
#
# Usage:
#   export ROUTER_URL=http://<PM_IP>:30000
#   python3 niah_con_sweep.py                 # con=1,8,16,32 @ 50K
#   python3 niah_con_sweep.py 6000 1,8,16,32  # custom ctx tokens + concurrency list

URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:30000")
TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
CONS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "8", "16", "32"])]

# Natural, non-repetitive filler. Repetitive filler ("the quick brown fox" x N) is
# adversarial for KDA linear-attention and inflates misses even at con=1 — don't use it
# to judge a serving bug.
SENTS = [
    "The committee reviewed quarterly figures before lunch.",
    "Rainfall in the northern valley exceeded seasonal norms.",
    "Engineers recalibrated the sensor array on Tuesday.",
    "Market analysts revised their growth forecast downward.",
    "The museum acquired three paintings from a private estate.",
    "Volunteers cleared debris along the coastal footpath.",
    "A new bakery opened near the university library.",
    "Researchers published findings on migratory bird patterns.",
    "The orchestra rehearsed the symphony's final movement.",
    "Farmers rotated crops to preserve soil nitrogen levels.",
    "The lighthouse keeper logged an unusual tide at dawn.",
    "Students debated the ethics of autonomous vehicles.",
]

def mk(needle, depth, toks):
    body, i = [], 0
    target = toks * 4
    while len(" ".join(body)) < target:
        body.append(SENTS[i % len(SENTS)] + " (" + str(i) + ")")
        i += 1
    full = " ".join(body)
    cut = int(len(full) * depth)
    return ("Read the document.\n" + full[:cut]
            + " IMPORTANT: The magic keyword for slot " + needle
            + " is ZEBRA-" + needle + ". " + full[cut:]
            + "\nQuestion: What is the magic keyword for slot " + needle
            + "? Answer with only the keyword.")

def one(i, base):
    needle = str(base + i)
    depth = [0.15, 0.35, 0.55, 0.75][i % 4]  # spread; skip 0.9 end-needle (benign miss)
    body = json.dumps({
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": mk(needle, depth, TOK)}],
        "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            URL + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}), timeout=900)
        j = json.loads(r.read()); m = j["choices"][0]["message"]
        full = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
        return ("ZEBRA-" + needle) in full, round(time.time() - t0, 1)
    except Exception as e:
        return False, "ERR:" + str(e)[:30]

def run(con, base):
    t0 = time.time()
    if con == 1:
        res = [one(0, base)]
    else:
        with cf.ThreadPoolExecutor(max_workers=con) as ex:
            res = list(ex.map(lambda i: one(i, base), range(con)))
    ok = sum(1 for r in res if r[0] is True)
    print("con=%-3d @%-6dtok: %2d/%-2d = %3.0f%%  wall=%.0fs"
          % (con, TOK, ok, con, 100 * ok / con, time.time() - t0)); sys.stdout.flush()
    return ok, con

if __name__ == "__main__":
    print("=== NIAH distinct-needle concurrency accuracy @%d tok (thinking=false) ===" % TOK)
    base, tot_ok, tot = 20000, 0, 0
    for con in CONS:
        ok, n = run(con, base); tot_ok += ok; tot += n; base += 100
    print("TOTAL: %d/%d = %.0f%%" % (tot_ok, tot, 100 * tot_ok / tot))
