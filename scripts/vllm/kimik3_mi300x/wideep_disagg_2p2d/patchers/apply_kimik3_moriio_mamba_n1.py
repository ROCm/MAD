#!/usr/bin/env python3
"""ROOT-CAUSE FIX (the last one): mamba/KDA N-vs-N-1 prefill/decode boundary.

Kimi-K3 is hybrid MLA + KDA(mamba). vLLM's own reference disagg connectors
(nixl/base_scheduler.py, mooncake) handle hybrid/mamba PD with a SYMMETRIC pair:
  - P-side _truncate_mamba_request_for_prefill: drop the LAST prompt token so the
    prefiller computes recurrent state h(N-1), not h(N). (max_tokens=1, guarded.)
  - D-side _get_remote_prefill_token_count: return N-1 for mamba, so the decoder
    recomputes the last prompt token and derives h(N) itself from h(N-1).
Comment verbatim from nixl base_scheduler.py:368: "Returns N-1 for Mamba models
since the decoder always recomputes the last token and must start from h(N-1)."

The MoRIIO connector NEVER ported this. Result (proven by K3_INPUTS_PROBE):
  PREFILL computes all N tokens (positions 0..N-1) -> KDA state seated AFTER token
  N-1 (all N applied). DECODE gets num_computed=N-1, RECOMPUTES the last prompt
  token, and re-applies it to the already-N-advanced recurrent state -> the last
  token is DOUBLE-COUNTED in the KDA recurrence -> wrong logits from token 1 ->
  decode echoes the last prompt token ("gold is"->"is"). Attention tolerates the
  mismatch (paged, position-indexed); mamba's monolithic recurrent state does not.
This is why transferring correct KDA state is WORSE than zero state, and why the
byte-perfect transport still yields wrong output.

THIS PATCH ports the nixl fix into MoRIIOConnectorScheduler:
  H1  __init__: compute self._has_mamba from the model's hybrid config
      (Kimi-K3 text_config.linear_attn_config.kda_layers) -> True for K3.
  H2  add _truncate_mamba_request_for_prefill + _get_remote_prefill_token_count
      (verbatim semantics from nixl base_scheduler).
  H3  get_num_new_matched_tokens:
        * D-side (consumer) WRITE: return N-1 (was N) for mamba so decode recomputes
          the last token from h(N-1).  [line ~507]
        * P-side (producer) do_remote_decode + _has_mamba: truncate the prompt to
          N-1 so prefill computes h(N-1).  [producer branch]

Attention path unchanged. Non-mamba models: _has_mamba False -> zero behavior
change (all N-1 -> N, no truncation). Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_moriio_mamba_n1.py <vllm_install_dir>
"""
import os
import sys

CONN = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARK = "k3-mamba-n1"


def main():
    base = sys.argv[1]
    path = os.path.join(base, CONN)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0

    # --- H1: detect hybrid/mamba in MoRIIOConnectorScheduler.__init__ ---
    h1_old = (
        "        self.is_producer = self.kv_transfer_config.kv_role == \"kv_producer\"\n"
    )
    h1_new = (
        "        self.is_producer = self.kv_transfer_config.kv_role == \"kv_producer\"\n"
        "        # " + MARK + ": hybrid (mamba/KDA) detection. Kimi-K3 carries\n"
        "        # text_config.linear_attn_config.kda_layers; any linear/mamba/kda\n"
        "        # marker => the recurrent-state N-vs-N-1 boundary applies.\n"
        "        self._has_mamba = False\n"
        "        try:\n"
        "            _mc = getattr(self.vllm_config, 'model_config', None)\n"
        "            _hf = getattr(_mc, 'hf_config', None) if _mc is not None else None\n"
        "            _tc = getattr(_hf, 'text_config', None) or _hf\n"
        "            _la = getattr(_tc, 'linear_attn_config', None)\n"
        "            if _la is None and isinstance(getattr(_tc, '__dict__', None), dict):\n"
        "                _la = _tc.__dict__.get('linear_attn_config')\n"
        "            if _la:\n"
        "                self._has_mamba = True\n"
        "        except Exception:\n"
        "            self._has_mamba = False\n"
        "        import os as _k3n1os\n"
        "        if _k3n1os.environ.get('K3_MAMBA_N1_FORCE', '') in ('0', '1'):\n"
        "            self._has_mamba = (_k3n1os.environ['K3_MAMBA_N1_FORCE'] == '1')\n"
        "        logger.info('[" + MARK + "] _has_mamba=%s (mamba N-1 boundary %s)',\n"
        "                    self._has_mamba, 'ON' if self._has_mamba else 'off')\n"
    )

    # --- H2: helper methods (insert right before get_num_new_matched_tokens of the
    #     scheduler class). Anchor on the scheduler's get_num_new_matched_tokens def
    #     (the one taking (self, request, num_computed_tokens) inside the Scheduler).
    #     There are two defs with that name; the scheduler's is the 2nd (line ~481).
    h2_anchor = (
        "    def get_num_new_matched_tokens(\n"
        "        self,\n"
        "        request: \"Request\",\n"
        "        num_computed_tokens: int,\n"
        "    ) -> tuple[int, bool]:\n"
    )
    h2_new = (
        "    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:\n"
        "        # " + MARK + ": D-side. Mamba decoder recomputes the last prompt\n"
        "        # token and must start from h(N-1), so it pulls only N-1 tokens.\n"
        "        if getattr(self, '_has_mamba', False) and num_prompt_tokens > 1:\n"
        "            return num_prompt_tokens - 1\n"
        "        return num_prompt_tokens\n"
        "\n"
        "    def _truncate_mamba_request_for_prefill(self, request: \"Request\") -> None:\n"
        "        # " + MARK + ": P-side. Drop the last prompt token so the prefiller\n"
        "        # computes h(N-1) not h(N); the decoder recomputes token N to get\n"
        "        # h(N). Guarded against repeated truncation on preempt/reschedule.\n"
        "        params = request.kv_transfer_params\n"
        "        if (\n"
        "            params is not None\n"
        "            and not params.get('_p_side_truncated')\n"
        "            and request.num_prompt_tokens > 1\n"
        "        ):\n"
        "            if request.prompt_token_ids is not None:\n"
        "                request.prompt_token_ids.pop()\n"
        "            elif getattr(request, 'prompt_embeds', None) is not None:\n"
        "                request.prompt_embeds = request.prompt_embeds[:-1]\n"
        "            else:\n"
        "                return\n"
        "            request._all_token_ids.pop()\n"
        "            request.num_prompt_tokens -= 1\n"
        "            request.max_tokens = 1\n"
        "            params['_p_side_truncated'] = True\n"
        "            logger.info('[" + MARK + "] P-side truncated req %s to N-1 for mamba prefill',\n"
        "                        request.request_id)\n"
        "\n"
        + h2_anchor
    )

    # --- H3a: P-side truncation must run BEFORE the producer early-return
    #     (`if self.is_producer: return 0, False`). In MoRIIO WRITE the producer
    #     is the PREFILL leg (do_remote_decode) and returns early, so put the
    #     mamba truncation ahead of that guard.
    h3a_old = (
        "        if self.is_producer:\n"
        "            return 0, False\n"
        "\n"
        "        token_ids = request.prompt_token_ids or []\n"
    )
    h3a_new = (
        "        # " + MARK + ": P-side prompt truncation for mamba runs BEFORE the\n"
        "        # producer early-return. In MoRIIO WRITE the producer is the prefill\n"
        "        # leg (do_remote_decode); drop its last prompt token so it computes\n"
        "        # h(N-1). The decoder (consumer) recomputes token N to derive h(N).\n"
        "        _k3_params = request.kv_transfer_params\n"
        "        if (\n"
        "            getattr(self, '_has_mamba', False)\n"
        "            and _k3_params is not None\n"
        "            and _k3_params.get('do_remote_decode')\n"
        "        ):\n"
        "            self._truncate_mamba_request_for_prefill(request)\n"
        "        if self.is_producer:\n"
        "            return 0, False\n"
        "\n"
        "        token_ids = request.prompt_token_ids or []\n"
    )
    # --- H3b: D-side WRITE return N-1 for mamba (was: len(token_ids) - num_computed)
    h3b_old = (
        "        if self.mode == MoRIIOMode.WRITE:\n"
        "            # MoriiO in write mode, no remote prefill\n"
        "\n"
        "            return len(token_ids) - num_computed_tokens, True\n"
        "\n"
        "        return len(token_ids) - 1 - num_computed_tokens, False\n"
    )
    h3b_new = (
        "        if self.mode == MoRIIOMode.WRITE:\n"
        "            # MoriiO in write mode, no remote prefill.\n"
        "            # " + MARK + ": D-side returns N-1 for mamba (decode recomputes the\n"
        "            # last prompt token from h(N-1)); non-mamba keeps N.\n"
        "            _k3_n = self._get_remote_prefill_token_count(len(token_ids))\n"
        "            return _k3_n - num_computed_tokens, True\n"
        "\n"
        "        return len(token_ids) - 1 - num_computed_tokens, False\n"
    )

    for old, new, tag in [
        (h1_old, h1_new, "H1 _has_mamba"),
        (h2_anchor, h2_new, "H2 helpers"),
        (h3a_old, h3a_new, "H3a P-side truncate pre-producer"),
        (h3b_old, h3b_new, "H3b D-side N-1"),
    ]:
        if old not in src:
            print(f"[{MARK}] {tag}: ANCHOR MISSING", file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)

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
