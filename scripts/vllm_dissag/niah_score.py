# Shared NIAH haystack + scoring. Used by benchmark_niah.py (live) and
# regrade_niah.py (offline). Word lists live here so the two cannot drift.
#
# Live bag verdict (found= / decoys= / verdict=) stays presence-based so new
# jobs remain comparable to earlier product rows. Extra fields
# (streak, p@10, product_pass, junk) are the HMA=0 upper-rung lens.
import os
import random
import re

FILLER = (
    "table chair window bottle pencil garden river mountain coffee planet "
    "engine guitar pillow ticket basket candle market silver button orange "
    "rocket napkin ladder pepper carpet helmet jacket mirror anchor pocket "
    "branch copper saddle tunnel violin wallet zipper meadow cactus pebble"
).split()
ANIMALS = [
    "elephant", "giraffe", "kangaroo", "penguin", "dolphin",
    "tiger", "rhinoceros", "octopus", "crocodile", "panda",
]
DECOYS = [
    "camel", "dog", "cat", "bird", "fish", "rabbit", "horse", "lion", "bear",
    "monkey", "snake", "frog", "turtle", "whale", "shark", "starfish",
    "jellyfish", "seahorse", "butterfly", "dragonfly", "ladybug", "koala",
    "zebra", "gorilla", "cow", "pig", "sheep", "goat", "deer", "fox", "wolf",
    "worm", "snail", "bee", "lizard", "hippo", "hippopotamus", "leopard",
    "cheetah", "squirrel", "mouse", "rat", "duck", "goose", "chicken", "eagle",
    "owl", "parrot", "crab", "lobster", "spider", "ant", "beetle", "moth",
    "donkey", "buffalo", "otter", "seal", "walrus", "raccoon", "badger",
]
MAX_DECOYS = int(os.environ.get("NIAH_MAX_DECOYS", "2"))
_ANIMAL_SET = set(ANIMALS)
_FILLER_PAT = {w: re.compile(r"\b%s(?:e?s)?\b" % re.escape(w)) for w in FILLER}
_ITEM_SPLIT = re.compile(r"[,;\n]+|\s+\d+[.)]\s*")

# Prefer a stable tag over whichever substring appears first in the completion.
_JUNK_RULES = (
    ("hyperfine", re.compile(r"hyperfine", re.I)),
    ("bash-fence", re.compile(r"```(?:bash|sh|shell)\b", re.I)),
    ("/dev/null", re.compile(r"/dev/null")),
    ("word-search-generator", re.compile(r"word-search-generator", re.I)),
    ("target/release", re.compile(r"target/release")),
    ("shebang", re.compile(r"#!/bin/")),
    ("benchmark-serving", re.compile(r"Benchmark Serving")),
)

_WORD_RE = {}
_PAT = {
    w: re.compile(r"\b%s(?:e?s)?\b" % re.escape(w)) for w in ANIMALS + DECOYS
}
_NEG_CUE = re.compile(
    r"\b(?:not|no|isn't|aren't|didn't|don't|without|except|other than|"
    r"rather than|instead of)\b",
    re.I,
)
_CLAUSE_BREAK = re.compile(r"[.!?;\n]")
_NEG_LOOKBACK = 80
_NEG_AFTER = re.compile(
    r"^[\s*_`:,\-]*\((?:[^)]{0,40}?)\b(?:not|isn't|no)\b", re.I
)


def mentions(text, word):
    """Whole-word match, tolerating a plural."""
    pat = _WORD_RE.get(word)
    if pat is None:
        pat = _WORD_RE[word] = re.compile(r"\b%s(?:e?s)?\b" % re.escape(word))
    return pat.search(text) is not None


# NIAH_NEEDLE_BAND="lo-hi" confines all ten needles to that fraction of the
# haystack, e.g. "0.0-0.1" puts them in the first 10%. Unset (default) keeps the
# historical even spread and is byte-identical to every scored row.
#
# This exists for the indexer diagnostic, NOT for scoring. Earlier runs measured
# where the lightning indexer selects, bucketed into ten equal depth bins, and
# Flash (fails 8k) and Pro (retrieves 8k) came out INDISTINGUISHABLE — both flat
# at ~10% per bin. The reason is this function: needles land at (i+1)/11, i.e.
# 9%, 18% ... 91%, exactly one per bin. A perfect needle-only retriever would
# therefore also produce a flat ten-bin histogram, so the measurement cannot
# separate perfect retrieval from uniform random selection.
#
# Clustering the needles breaks that degeneracy: a content-driven indexer must
# concentrate its selections in the band, while a position-driven one stays flat
# regardless of where the needles are.
#
# Any run with this set is a DIAGNOSTIC. Do not compare its found=/decoys= to
# product rows -- it is a different haystack.
_NEEDLE_BAND = os.environ.get("NIAH_NEEDLE_BAND", "").strip()


def _band_bounds():
    """(lo, hi) fractions from NIAH_NEEDLE_BAND, or None when unset/invalid."""
    if not _NEEDLE_BAND:
        return None
    try:
        lo_s, hi_s = _NEEDLE_BAND.split("-", 1)
        lo, hi = float(lo_s), float(hi_s)
    except ValueError:
        return None
    lo, hi = max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))
    return (lo, hi) if hi > lo else None


def make_haystack(n_words, seed=0):
    rng = random.Random(seed)
    words = [rng.choice(FILLER) for _ in range(n_words)]
    band = _band_bounds()
    if band is None:
        step = max(n_words // (len(ANIMALS) + 1), 1)
        for i, animal in enumerate(ANIMALS):
            words[min((i + 1) * step, len(words) - 1)] = animal
        return " ".join(words)
    # Same (i+1)*step layout, rescaled into [lo, hi) so needle ORDER and relative
    # spacing are preserved -- retrieval_walk and product_pass still mean what
    # they meant, just over a shorter stretch of text.
    lo, hi = band
    start = int(lo * n_words)
    width = max(int((hi - lo) * n_words), len(ANIMALS) + 1)
    step = max(width // (len(ANIMALS) + 1), 1)
    for i, animal in enumerate(ANIMALS):
        words[min(start + (i + 1) * step, len(words) - 1)] = animal
    return " ".join(words)


def junk_reason(text):
    """Why this completion is not a NIAH answer, or None.

    A 16k answer scored NONE 0/10 on a hyperfine README the model emitted.
    That is generation, not a missing WRITE, and not tee mixing bash into SSE.
    """
    if not text or not str(text).strip():
        return None
    for name, pat in _JUNK_RULES:
        if pat.search(text):
            return name
    return None

def _negated(text, start, end):
    window = text[max(0, start - _NEG_LOOKBACK):start]
    breaks = list(_CLAUSE_BREAK.finditer(window))
    if breaks:
        window = window[breaks[-1].end():]
    if _NEG_CUE.search(window):
        return True
    return bool(_NEG_AFTER.match(text[end:end + 56]))


def guesses(text):
    """Ordered, de-duplicated (word, kind). kind: hit / decoy / neg."""
    found = []
    for w, pat in _PAT.items():
        for m in pat.finditer(text):
            found.append((m.start(), m.end(), w))
    found.sort()
    out, seen = [], set()
    for start, end, w in found:
        if w in seen:
            continue
        seen.add(w)
        if _negated(text, start, end):
            out.append((w, "neg"))
        elif w in _ANIMAL_SET:
            out.append((w, "hit"))
        else:
            out.append((w, "decoy"))
    return out


def prefix_score(text, k=10):
    """Order-aware metrics. Denials are dropped before streak / p@k."""
    all_g = guesses(text)
    seq = [g for g in all_g if g[1] != "neg"]
    negated = [w for w, kind in all_g if kind == "neg"]
    streak = 0
    for _w, kind in seq:
        if kind != "hit":
            break
        streak += 1
    head = seq[:k]
    k_hit = sum(1 for _w, kind in head if kind == "hit")
    k_dec = len(head) - k_hit
    # p_at_k divides by the number of names ACTUALLY emitted, so it is precision
    # over the answer, not precision@k. A short answer therefore scores 1.00:
    # A 29k draw was ' penguin 2. penguin 3. penguin ...', which guesses()
    # de-duplicates to a single name, giving head=1 and p_at_k=1/1=1.00 on ONE
    # retrieved needle. Kept unchanged for continuity with rows already reported.
    p_at_k = (k_hit / len(head)) if head else 0.0
    # r_at_k is recall@k and is what upper-rung comparisons need: it divides by k,
    # so it cannot be inflated by answering with one name, and it is directly
    # comparable across answer lengths. 29k penguin-loop -> 0.10, not 1.00.
    r_at_k = k_hit / float(k) if k else 0.0
    hits = [w for w, kind in seq if kind == "hit"]
    decoys = [w for w, kind in seq if kind == "decoy"]
    full = (len(hits) / len(seq)) if seq else 0.0
    names = [w for w, _kind in seq]
    def _idx(name):
        try:
            return names.index(name)
        except ValueError:
            return None
    k_i, p_i, d_i = _idx("kangaroo"), _idx("penguin"), _idx("dolphin")
    if k_i is None or p_i is None:
        product = False
    elif d_i is None:
        product = True
    else:
        product = k_i < d_i and p_i < d_i
    if streak >= 8:
        prefix_verdict = "RETRIEVAL"
    elif streak >= 4 and p_at_k >= 0.5:
        prefix_verdict = "RETRIEVAL-PARTIAL"
    elif k_hit >= 5 and p_at_k >= 0.5:
        prefix_verdict = "PARTIAL"
    elif k_hit >= 2:
        prefix_verdict = "WEAK"
    else:
        prefix_verdict = "NONE"
    return {
        "streak": streak,
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
        "k_hit": k_hit,
        "k_dec": k_dec,
        "hits": hits,
        "decoys": decoys,
        "negated": negated,
        "full": full,
        "order": names,
        "product_pass": product,
        "prefix_verdict": prefix_verdict,
    }


def dump_score(text):
    """Count non-answer content. The decoy guard cannot see any of this.

    `decoys` only counts fabricated ANIMAL names, and streak / p@k skip
    non-animals entirely, so an answer made mostly of haystack filler grades
    perfectly. A Pro 2k answer was 44 items, 10 of them animals, and scored
    found=10/10 decoys=0 precision=1.00 p@10=1.00 streak=10 product_pass=True
    verdict=RETRIEVAL -- every field clean. Ravi's recall-only grader is fooled
    the same way. Another was worse (49 items, 39 filler) and also read 10/10.

    filler is exact, not heuristic: the FILLER vocabulary is right here, so a
    dump is counted rather than inferred. list_precision is item-based and only
    meaningful for list-shaped answers -- prose scores low on it by shape alone,
    which is why it annotates rather than gates.
    """
    filler = sorted(w for w in FILLER if _FILLER_PAT[w].search(text))
    items = [i.strip(" \t*_`.-:") for i in _ITEM_SPLIT.split(text)]
    items = [i for i in items if i]
    answer_items = [
        i for i in items
        if any(_PAT[a].search(i) for a in ANIMALS)
    ]
    extra = len(items) - len(answer_items)
    return {
        "filler": filler,
        "items": len(items),
        "extra": extra,
        "list_precision": (len(answer_items) / len(items)) if items else 0.0,
    }


def bag_verdict(n_found, n_decoys, junk=None):
    """Presence bag. junk overrides to INVALID (the 16k hyperfine class)."""
    if junk:
        return "INVALID"
    if n_decoys > MAX_DECOYS:
        return "GUESSING (%d fabricated)" % n_decoys
    if n_found >= 8:
        return "RETRIEVAL"
    if n_found >= 5:
        return "PARTIAL"
    return "NONE"


def score_answer(raw):
    """Score one completion. ``raw`` is the model text (not lowercased)."""
    text = (raw or "").lower()
    junk = junk_reason(raw or "")
    found = sorted(a for a in ANIMALS if mentions(text, a))
    decoys = sorted(d for d in DECOYS if mentions(text, d))
    denom = len(found) + len(decoys)
    precision = (len(found) / denom) if denom else 0.0
    clean = len(decoys) <= MAX_DECOYS
    walk = mentions(text, "kangaroo") and mentions(text, "penguin") and clean
    pref = prefix_score(text)
    dump = dump_score(text)
    verdict = bag_verdict(len(found), len(decoys), junk)
    return {
        "found": found,
        "decoys": decoys,
        "precision": precision,
        "filler": dump["filler"],
        "items": dump["items"],
        "extra": dump["extra"],
        "list_precision": dump["list_precision"],
        "clean": clean,
        "retrieval_walk": walk,
        "verdict": verdict,
        "junk": junk,
        "streak": pref["streak"],
        "p_at_k": pref["p_at_k"],
        "r_at_k": pref["r_at_k"],
        "k_hit": pref["k_hit"],
        "k_dec": pref["k_dec"],
        "negated": pref["negated"],
        "full": pref["full"],
        "order": pref["order"],
        "product_pass": pref["product_pass"],
        "prefix_verdict": pref["prefix_verdict"],
        "kangaroo": mentions(text, "kangaroo"),
        "penguin": mentions(text, "penguin"),
        "dolphin": mentions(text, "dolphin"),
    }
