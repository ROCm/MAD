#!/usr/bin/env python3
"""
Kimi-K3 (1.5T MXFP4) capacity model — WIDE-EP framing.

Corrected to match the measured MI300X sizing analysis (Confluence 1830010189):
  - The 896 routed experts are ALWAYS EP16-sharded (84.2 GiB/GPU) — that axis is FIXED
    by the wide-EP design. It does NOT depend on the attention TP/DP split.
  - The only free axis is how ATTENTION (the 106.5 GiB bf16 remainder) is parallelized:
    it is replicated per DP rank and divided by TP. So per-GPU weights = 84.2 + 106.5/TP.
  - Non-weight overhead (MoRI-EP heap + activations/cudagraph/reserve/frag) is
    EMPIRICALLY ANCHORED from the live TP8/DP2 decode pool reading 158.8 GiB/GPU.

Weight bytes are ground-truth (parsed from safetensors headers; total 1453.7 GiB / 1.42 TiB).
Overhead terms are measured, not modeled — re-validate per platform via rocm-smi at bring-up.

NOTE: ideally this whole model is replaced by real per-node measurement (rocm-smi after
load) for each (platform, TP/DP) cell. Until those runs land, these are the corrected
theoretical figures anchored to the one live data point we have (TP8/DP2 = 158.8 GiB).
"""

GiB = 1024**3

# ---- Measured weight footprint (safetensors headers; total 1453.7 GiB / 1.42 TiB) ----
EXPERTS_MXFP4_GIB = 1347.1      # 896 routed experts, MXFP4 U8-packed + U8 scales
REPLICATED_BF16_GIB = 106.5     # the bf16 "tax": replicated per DP rank, divided by TP
#   breakdown of the 106.5: MLA attn 67.4, shared experts 22.6, embed+lm_head 4.4,
#   KDA+norms+dense-L0 11.3  (KDA holds FIXED recurrent state, not per-token KV)
WEIGHTS_TOTAL = EXPERTS_MXFP4_GIB + REPLICATED_BF16_GIB   # 1453.7 GiB

# ---- Wide-EP invariant ----
EP = 16                                  # 896 experts / 16 = 56 experts/GPU, always
EXPERTS_PER_GPU = EXPERTS_MXFP4_GIB / EP # 84.2 GiB/GPU — CONSTANT across all TP/DP splits

# ---- Empirically-anchored non-weight overhead (from live TP8/DP2 = 158.8 GiB) ----
MORI_EP_HEAP_GIB = 16.0    # MoRI-EP shmem heap, reserved BEFORE the vLLM snapshot
OVERHEAD_GIB     = 28.0    # activations + cudagraph + reserved + fragmentation
# check: TP8 weights = 84.2 + 106.5/8 = 97.5 ; +16 +28 = 141.5 ; +KV -> ~158.8 live (≈17 GiB KV)

# ---- K3 shape (config) ----
N_LAYERS, MLA_LAYERS, KDA_LAYERS = 93, 24, 69
HIDDEN, KV_LORA, QK_ROPE = 7168, 512, 64
# MLA KV per token per full-attn layer, fp8 KV cache (kv_lora + rope), ~13.5 KiB/token total
# across 24 layers per the sizing doc. KDA layers hold FIXED recurrent state, not per-token KV.
KV_KIB_PER_TOKEN = 13.5    # fp8, all 24 MLA layers combined (from sizing doc)

def weights_per_gpu(TP):
    return EXPERTS_PER_GPU + REPLICATED_BF16_GIB / TP

def kv_room_gib(TP, cap):
    return cap - (weights_per_gpu(TP) + MORI_EP_HEAP_GIB + OVERHEAD_GIB)

def kv_tokens(TP, cap):
    room = kv_room_gib(TP, cap)
    return room * GiB / (KV_KIB_PER_TOKEN * 1024) if room > 0 else 0.0

GPUS = {"MI325X": 256.0, "MI300X": 192.0}   # GiB HBM/GPU (MI325X 255.9 usable)

# TP/DP splits at EP16 (TP*DP = 16 per pool)
SPLITS = [("TP1/DP16",1,16), ("TP2/DP8",2,8), ("TP4/DP4",4,4), ("TP8/DP2",8,2)]

def table(gpu_name):
    cap = GPUS[gpu_name]
    rows=[]
    for name,TP,DP in SPLITS:
        w = weights_per_gpu(TP)
        base = w + MORI_EP_HEAP_GIB + OVERHEAD_GIB     # everything except KV
        room = cap - base
        rows.append(dict(split=name, TP=TP, DP=DP, w=w, base=base, room=room,
                         toks=kv_tokens(TP,cap), fit=("OK" if room>0 else "OOM")))
    return cap, rows

if __name__ == "__main__":
    print(f"weights total = {WEIGHTS_TOTAL:.1f} GiB  |  experts/GPU (EP16, fixed) = {EXPERTS_PER_GPU:.1f} GiB")
    print(f"replicated bf16 tax = {REPLICATED_BF16_GIB:.1f} GiB (÷TP)  |  heap {MORI_EP_HEAP_GIB} + overhead {OVERHEAD_GIB} GiB")
    print(f"anchor: TP8/DP2 base = {weights_per_gpu(8)+MORI_EP_HEAP_GIB+OVERHEAD_GIB:.1f} GiB (+KV ≈ live 158.8)")
    for g in GPUS:
        cap,rows = table(g)
        print(f"\n=== {g} ({cap:.0f} GiB/GPU) ===")
        print(f"{'split':12s}{'W/GPU':>8s}{'base(+heap+oh)':>16s}{'KV room':>10s}{'~KV tokens':>13s}  fit")
        for r in rows:
            tk = f"{r['toks']/1000:.0f}k" if r['toks']>0 else "-"
            print(f"{r['split']:12s}{r['w']:8.1f}{r['base']:16.1f}{r['room']:10.1f}{tk:>13s}  {r['fit']}")
