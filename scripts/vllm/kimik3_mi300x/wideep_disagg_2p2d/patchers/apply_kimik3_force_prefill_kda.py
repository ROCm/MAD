#!/usr/bin/env python3
"""DIAGNOSTIC/FIX: route non-spec GDN tokens through the PREFILL kda kernel.

Disagg boundary bug: the last prompt token (query_len=1) is classified as a
DECODE on the decode instance's first step, so its KDA output is computed by
fused_recurrent_kda_packed_decode using the TRANSFERRED conv/recurrent state.
That single-step decode kernel, fed a remote-transferred state, does not
reproduce prefill's chunk_kda output for that token -> the boundary token's
hidden state (hence its MLA retrieval QUERY) is wrong -> exact recall fails
while parametric knowledge/reasoning survive. vLLM's own GDN builder already
reclassifies decodes->prefills in the spec case ("the prefill kernel handles
1-token sequences with initial state correctly, producing identical results").

This forces num_decodes=0 (all non-spec -> prefill path) when env
K3_FORCE_PREFILL_KDA=1, so the boundary token uses the SAME chunk_kda kernel
prefill used. Diagnostic first: if recall becomes correct, the fix direction
(boundary-token-as-prefill) is confirmed; then we refine to boundary-only.

Gated (default off) => byte-identical unless K3_FORCE_PREFILL_KDA=1.
Idempotent, anchor-based, two-pass, py_compile-checked.
Usage: apply_kimik3_force_prefill_kda.py <vllm_install_dir>
"""
import os
import sys

MARK = "k3-force-prefill-kda"
REL = "v1/attention/backends/gdn_attn.py"

OLD = (
    "            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()\n"
    "            # Exclude zero-length padded sequences from prefill count.\n"
    "            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()\n"
    "            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len\n"
    "            num_decode_tokens = num_decodes\n"
)
NEW = (
    "            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()\n"
    "            # Exclude zero-length padded sequences from prefill count.\n"
    "            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()\n"
    "            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len\n"
    "            num_decode_tokens = num_decodes\n"
    "            import os as _k3fpk_os  # " + MARK + "\n"
    "            if _k3fpk_os.environ.get('K3_FORCE_PREFILL_KDA', '0') == '1' and num_decodes > 0:\n"
    "                # " + MARK + ": route ALL non-spec 1-token seqs through the prefill\n"
    "                # chunk_kda kernel (with initial_state) so the disagg boundary token\n"
    "                # is bit-consistent with prefill. num_zero_len stays excluded.\n"
    "                num_prefills = non_spec_query_lens_cpu.size(0) - num_zero_len\n"
    "                num_decodes = 0\n"
    "                num_decode_tokens = 0\n"
)


def main():
    if len(sys.argv) < 2:
        print(f"[{MARK}] usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 1
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0
    if OLD not in src:
        print(f"[{MARK}] ANCHOR MISSING", file=sys.stderr)
        return 1
    src = src.replace(OLD, NEW, 1)
    open(path, "w").write(src)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        print(f"[{MARK}] compile FAIL {e}", file=sys.stderr)
        return 1
    print(f"[{MARK}] applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
