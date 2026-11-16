#!/usr/bin/env python3
"""Offline NIAH scorer tests. No cluster / no GPUs.

Usage: python3 tests/test_niah_score.py
"""
import hashlib
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import niah_score
from niah_score import ANIMALS, junk_reason, make_haystack, score_answer


def _check(name, cond):
    if not cond:
        raise SystemExit("FAIL: %s" % name)
    print("ok  %s" % name)


def _md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _reload_with_band(band):
    """niah_score reads NIAH_NEEDLE_BAND at import, so re-import to change it."""
    os.environ["NIAH_NEEDLE_BAND"] = band
    return importlib.reload(niah_score)


def main():
    dump = (
        "0 ```  ## Benchmark  ```bash $ hyperfine --warmup 3 "
        "'./target/release/word-search-generator -i data/words.txt "
        "-o /dev/null -r 1000'"
    )
    s = score_answer(dump)
    _check("16k hyperfine answer is junk", s["junk"] == "hyperfine")
    _check("16k hyperfine verdict INVALID", s["verdict"] == "INVALID")
    _check("16k hyperfine found empty", s["found"] == [])
    _check("junk_reason matches", junk_reason(dump) == "hyperfine")

    clean = (
        "elephant (appears once), giraffe (appears once), kangaroo (appears once), "
        "penguin (appears once), dolphin (appears once), tiger, rhinoceros, "
        "octopus, crocodile, panda"
    )
    s = score_answer(clean)
    _check("clean 10/10 RETRIEVAL", s["verdict"] == "RETRIEVAL")
    _check("clean found 10", len(s["found"]) == 10)
    _check("clean product_pass", s["product_pass"] is True)
    _check("clean streak 10", s["streak"] == 10)
    _check("clean not junk", s["junk"] is None)

    order_fail = (
        "dolphin, elephant, giraffe, kangaroo, penguin, tiger, octopus, "
        "panda, hippopotamus"
    )
    s = score_answer(order_fail)
    _check("16k-shape RETRIEVAL bag", s["verdict"] == "RETRIEVAL")
    _check("16k-shape walk True", s["retrieval_walk"] is True)
    _check("16k-shape product FAIL", s["product_pass"] is False)
    _check("16k-shape one decoy", s["decoys"] == ["hippopotamus"])

    phase2 = (
        "1. dolphin 2. tiger 3. penguin 4. elephant 5. giraffe 6. rabbit "
        "7. horse 8. lion 9. bear 10. monkey 11. kangaroo"
    )
    s = score_answer(phase2)
    _check("phase2 GUESSING", s["verdict"].startswith("GUESSING"))
    _check("phase2 streak 5", s["streak"] == 5)
    _check("phase2 prefix RETRIEVAL-PARTIAL", s["prefix_verdict"] == "RETRIEVAL-PARTIAL")
    _check("phase2 product FAIL (dolphin first)", s["product_pass"] is False)

    denied = "1. **elephant** (not found) 2. **kangaroo** (appears once) 3. penguin"
    s = score_answer(denied)
    _check("negation drops elephant from streak", "elephant" in s["negated"])
    _check("bag still counts denied elephant", "elephant" in s["found"])
    _check("product_pass k then p, no dolphin", s["product_pass"] is True)

    eight_k = "elephant, giraffe, kangaroo, penguin, dolphin, tiger, rhinoceros, octopus, panda"
    s = score_answer(eight_k)
    _check("8k truncated RETRIEVAL", s["verdict"] == "RETRIEVAL")
    _check("8k product_pass without crocodile", s["product_pass"] is True)

    hs = make_haystack(2000, 0).split()
    _check("2k haystack length", len(hs) == 2000)
    _check("elephant on 1/11 grid", hs[181] == "elephant")
    _check("panda not in last 128", 1810 < 2000 - 128)

    # NIAH_NEEDLE_BAND is the indexer diagnostic. It MUST leave
    # the product haystack untouched when unset -- these md5s are the ones the
    # Track A / Track B byte-identity check was established on.
    _check("default 2k md5", _md5(make_haystack(2000, 0)) == "196fcd78d9c6")
    _check("default 8k md5", _md5(make_haystack(8000, 0)) == "8a05d202608c")
    _check("default 16k md5", _md5(make_haystack(16000, 0)) == "f8b141c2c633")
    _check("default 35k md5", _md5(make_haystack(35000, 0)) == "eecbdef8abfc")

    ns = _reload_with_band("0.45-0.55")
    hs = ns.make_haystack(8000, 0).split()
    pos = [i for i, w in enumerate(hs) if w in ns.ANIMALS]
    _check("band keeps all ten needles", len(pos) == 10)
    _check("band preserves needle order", [hs[i] for i in pos] == ns.ANIMALS)
    _check("band confines to 0.45-0.55", all(0.45 <= i / 8000 < 0.55 for i in pos))
    _check("band changes the haystack", _md5(ns.make_haystack(8000, 0)) != "8a05d202608c")

    for bad in ("", "garbage", "0.5-0.2", "abc-def", "1.0-0.0"):
        ns = _reload_with_band(bad)
        _check("band %r falls back to product layout" % bad,
               _md5(ns.make_haystack(8000, 0)) == "8a05d202608c")

    # r@10 vs p@10. A 29k draw repeated ONE needle ten times; guesses()
    # de-duplicates, so p@10 divided 1 by 1 and read a perfect 1.00. r@10 divides
    # by k and is the metric upper-rung rows must be compared on.
    loop = ("1. penguin 2. penguin 3. penguin 4. penguin 5. penguin "
            "6. penguin 7. penguin 8. penguin 9. penguin 10. penguin")
    s = score_answer(loop)
    _check("loop: one distinct needle", len(s["found"]) == 1)
    _check("loop: p@10 is inflated to 1.00", abs(s["p_at_k"] - 1.00) < 1e-9)
    _check("loop: r@10 sees through it", abs(s["r_at_k"] - 0.10) < 1e-9)
    _check("loop: streak is 1 not 10", s["streak"] == 1)
    _check("loop: prefix_verdict not RETRIEVAL",
           s["prefix_verdict"] not in ("RETRIEVAL", "RETRIEVAL-PARTIAL"))

    # Ten distinct needles: the two metrics must agree at the top of the scale.
    s = score_answer(", ".join(ANIMALS))
    _check("perfect: r@10 == p@10 == 1.00",
           abs(s["r_at_k"] - 1.0) < 1e-9 and abs(s["p_at_k"] - 1.0) < 1e-9)

    # 30k: four needles then six prior animals. r@10 must read 0.40.
    mixed = ("1. kangaroo 2. elephant 3. giraffe 4. tiger 5. zebra "
             "6. lion 7. bear 8. wolf 9. fox 10. hippo")
    s = score_answer(mixed)
    _check("mixed: r@10 is 0.40", abs(s["r_at_k"] - 0.40) < 1e-9)

    s = score_answer("")
    _check("empty: r@10 is 0.00", abs(s["r_at_k"] - 0.0) < 1e-9)

    import benchmark_niah as _niah
    _check("all None is a failed ladder",
           _niah.no_scored_answers({2000: [None, None], 8000: [None]}))
    _check("a 0/10 score is not a failed ladder",
           not _niah.no_scored_answers({2000: [(0, 0, False)]}))
    _check("mixed score+timeout is not a failed ladder",
           not _niah.no_scored_answers({2000: [(10, 0, False)], 8000: [None]}))

    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
