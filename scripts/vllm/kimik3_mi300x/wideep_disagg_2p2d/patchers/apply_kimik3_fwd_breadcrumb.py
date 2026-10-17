#!/usr/bin/env python3
"""Pre-forward breadcrumbs + cuda.synchronize to locate the disagg producer GPU fault
(no GPU debugger; sync attributes the async fault to the exact step).

Gated by K3_FWD_BREADCRUMB=1. Active runner is v1/worker/gpu_model_runner.py.
  [k3-bc] zero_block_ids SYNCED-ok / DONE+SYNCED   (KV block zeroing)
  [k3-bc] _model_forward START (pre-model synced-ok) / DONE+SYNCED   (the model fwd)
The LAST breadcrumb before "Memory access fault" names the offending step, because
each is followed by torch.cuda.synchronize() which surfaces the async fault there.

Idempotent, anchor-based, py_compile-checked.
Usage: apply_kimik3_fwd_breadcrumb.py <vllm_install_dir>
"""
import os
import sys


def _patch(path, anchor, repl, tag):
    if not os.path.isfile(path):
        print(f"[k3-bc] {tag}: file not found -- skip.")
        return 0
    src = open(path).read()
    if repl.split("\n")[0].strip() and repl in src:
        print(f"[k3-bc] {tag}: already applied.")
        return 0
    if anchor not in src:
        print(f"[k3-bc] {tag}: WARN anchor not found -- skip.")
        return 0
    src2 = src.replace(anchor, repl, 1)
    open(path, "w").write(src2)
    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:
        open(path, "w").write(src)
        print(f"[k3-bc] {tag}: ERROR compile: {e}", file=sys.stderr)
        return 1
    print(f"[k3-bc] {tag}: applied.")
    return 0


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    root = sys.argv[1]
    rc = 0
    gmr = os.path.join(root, "v1/worker/gpu_model_runner.py")

    # A) block-zeroing with syncs
    a_anchor = (
        "        if scheduler_output.new_block_ids_to_zero:\n"
        "            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)\n"
    )
    a_repl = (
        "        if scheduler_output.new_block_ids_to_zero:\n"
        "            import os as _os_k3z\n"
        "            _k3z = _os_k3z.environ.get(\"K3_FWD_BREADCRUMB\", \"0\") == \"1\"\n"
        "            if _k3z:\n"
        "                import torch as _t_k3z, logging as _lg_k3z\n"
        "                _t_k3z.cuda.synchronize()\n"
        "                _lg_k3z.getLogger(__name__).warning(\n"
        "                    \"[k3-bc] zero_block_ids SYNCED-ok ids=%s\",\n"
        "                    scheduler_output.new_block_ids_to_zero)\n"
        "            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)\n"
        "            if _k3z:\n"
        "                import torch as _t_k3z, logging as _lg_k3z\n"
        "                _t_k3z.cuda.synchronize()\n"
        "                _lg_k3z.getLogger(__name__).warning(\"[k3-bc] zero_block_ids DONE+SYNCED\")\n"
    )
    rc |= _patch(gmr, a_anchor, a_repl, "zero_block_ids")

    # B) _model_forward with syncs
    b_anchor = (
        "        return self.model(\n"
        "            input_ids=input_ids,\n"
        "            positions=positions,\n"
        "            intermediate_tensors=intermediate_tensors,\n"
        "            inputs_embeds=inputs_embeds,\n"
        "            **model_kwargs,\n"
        "        )\n"
    )
    b_repl = (
        "        import os as _os_k3mf\n"
        "        _k3mf = _os_k3mf.environ.get(\"K3_FWD_BREADCRUMB\", \"0\") == \"1\"\n"
        "        if _k3mf:\n"
        "            import torch as _t_k3mf, logging as _lg_k3mf\n"
        "            _t_k3mf.cuda.synchronize()\n"
        "            _lg_k3mf.getLogger(__name__).warning(\"[k3-bc] _model_forward START (pre-model synced-ok)\")\n"
        "        _out_k3mf = self.model(\n"
        "            input_ids=input_ids,\n"
        "            positions=positions,\n"
        "            intermediate_tensors=intermediate_tensors,\n"
        "            inputs_embeds=inputs_embeds,\n"
        "            **model_kwargs,\n"
        "        )\n"
        "        if _k3mf:\n"
        "            import torch as _t_k3mf, logging as _lg_k3mf\n"
        "            _t_k3mf.cuda.synchronize()\n"
        "            _lg_k3mf.getLogger(__name__).warning(\"[k3-bc] _model_forward DONE+SYNCED\")\n"
        "        return _out_k3mf\n"
    )
    rc |= _patch(gmr, b_anchor, b_repl, "model_forward")

    return rc


if __name__ == "__main__":
    sys.exit(main())
