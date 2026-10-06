#!/usr/bin/env python3
"""Log decode's actual input prep (positions / num_computed_tokens / block_table /
seq_lens) for the first few requests, to find why WRITE-mode decode produces wrong
logits from byte-perfect KV. Runs in gpu_model_runner._prepare_inputs (plain
Python, NOT the @eager_break_during_capture forward -> logging works).

Gated K3_INPUTS_PROBE=1. Logs once per ~request via a small counter. Idempotent,
anchor-based, py_compile-checked.
Usage: apply_kimik3_decode_inputs_probe.py <vllm_install_dir>
"""
import os, sys

REL = "v1/worker/gpu_model_runner.py"
MARK = "k3-inputsprobe"


def main():
    base = sys.argv[1]
    path = os.path.join(base, REL)
    if not os.path.isfile(path):
        print(f"[{MARK}] not found {path}", file=sys.stderr)
        return 1
    src = open(path).read()
    if MARK in src:
        print(f"[{MARK}] already applied.")
        return 0
    anchor = (
        "        token_indices_tensor = torch.from_numpy(token_indices)\n"
    )
    inject = (
        "        import os as _k3ipos\n"
        "        if _k3ipos.environ.get('K3_INPUTS_PROBE','0')=='1':\n"
        "            try:\n"
        "                _c=getattr(self,'_k3_ip_n',0)\n"
        "                if _c < 12:\n"
        "                    self._k3_ip_n=_c+1\n"
        "                    _nct=self.input_batch.num_computed_tokens_cpu[:num_reqs].tolist()\n"
        "                    _pos=positions_np[:min(8,len(positions_np))].tolist()\n"
        "                    _bt=None\n"
        "                    try:\n"
        "                        _bt=self.input_batch.block_table.block_table[0].get_cpu_tensor()[:num_reqs,:4].tolist()\n"
        "                    except Exception:\n"
        "                        try: _bt=self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs,:4].tolist()\n"
        "                        except Exception as _e2: _bt='ERR:'+repr(_e2)[:40]\n"
        "                    _tokfed=None; _ntok=None; _nprompt=None; _nout=None; _lastprompt=None\n"
        "                    try:\n"
        "                        _tidx=token_indices[:min(4,len(token_indices))]\n"
        "                        _tokfed=self.input_batch.token_ids_cpu.reshape(-1)[_tidx].tolist()\n"
        "                    except Exception as _te: _tokfed='ERR:'+repr(_te)[:30]\n"
        "                    try:\n"
        "                        _rid=self.input_batch.req_ids[0]\n"
        "                        _rs=self.requests.get(_rid) if hasattr(self,'requests') else None\n"
        "                        if _rs is not None:\n"
        "                            _ntok=getattr(_rs,'num_tokens',None); _nprompt=getattr(_rs,'num_prompt_tokens',None)\n"
        "                            _ao=getattr(_rs,'_all_token_ids',None) or getattr(_rs,'all_token_ids',None)\n"
        "                            if _ao is not None:\n"
        "                                _nout=len(_ao)-(_nprompt or 0)\n"
        "                                _lastprompt=list(_ao[max(0,(_nprompt or 1)-1):(_nprompt or 0)+2])\n"
        "                    except Exception as _re: _ntok='ERR:'+repr(_re)[:30]\n"
        "                    import logging as _k3iplg\n"
        "                    _k3iplg.getLogger('vllm.v1.worker.gpu_model_runner').info(\n"
        "                        '[" + MARK + "] num_reqs=%s num_sched=%s num_computed=%s pos=%s token_fed=%s num_tokens=%s num_prompt=%s num_out=%s around_boundary=%s bt=%s',\n"
        "                        num_reqs, list(num_scheduled_tokens[:num_reqs]), _nct, _pos, _tokfed, _ntok, _nprompt, _nout, _lastprompt, _bt)\n"
        "            except Exception as _k3ipe:\n"
        "                import logging as _k3iplg2\n"
        "                _k3iplg2.getLogger('vllm.v1.worker.gpu_model_runner').info('[" + MARK + "] EXC %r', _k3ipe)\n"
        "        token_indices_tensor = torch.from_numpy(token_indices)\n"
    )
    if anchor not in src:
        print(f"[{MARK}] anchor NOT found", file=sys.stderr)
        return 1
    src = src.replace(anchor, inject, 1)
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
