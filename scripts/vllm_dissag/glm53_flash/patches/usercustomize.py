import os
if os.environ.get("DISABLE_INDUCTOR_PM") == "1":
    try:
        import torch._inductor.config as _ic
        _ic.pattern_matcher = False
        try:
            import torch._inductor.fx_passes.fuse_attention as _fa
            _fa._sfdp_init = lambda *a, **k: None
        except Exception:
            pass
        print("[usercustomize] inductor pattern_matcher DISABLED (gfx950 sfdp guard)")
    except Exception as _e:
        print("[usercustomize] inductor guard failed:", _e)
if os.environ.get("DISABLE_DYNAMO") == "1":
    try:
        import torch._dynamo as _d
        _d.config.disable = True
        try:
            import torch
            torch.compile = (lambda model=None, *a, **k: (model if model is not None else (lambda f: f)))
        except Exception:
            pass
        print("[usercustomize] torch dynamo/inductor DISABLED (gfx950 no-kernel-image guard)")
    except Exception as _e:
        print("[usercustomize] dynamo guard failed:", _e)
