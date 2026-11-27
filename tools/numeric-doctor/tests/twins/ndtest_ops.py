"""Test ops for the twin check: a fused-op stand-in with a planted wrong-value bug, and its correct twin.

ndtest::rmsnorm_buggy(x, w, eps) -> Tensor      wrong (rows 0-3 scaled by 1.2) when x has exactly 37 rows
ndtest::rmsnorm_inplace_buggy(x!, w, eps)        the same, written into x (in place: its twin needs pre-call x)
ndtest::rmsnorm_nan(x, w, eps) -> Tensor         NaN rows on the 37-row call (the NaN path must still catch it)
ndtest::rmsnorm_ok(x, w, eps) -> Tensor          always correct (the no-false-positive control)

Shape-dependent, so the fault is deterministic and replays.  Importing this module registers the ops and their
twins (twins.REGISTRY), in the capture process and in the replay process alike.
"""
import torch

import twins

BAD_ROWS = 37


def _norm(x, w, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


@torch.library.custom_op("ndtest::rmsnorm_buggy", mutates_args=())
def rmsnorm_buggy(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    y = _norm(x, w, eps)
    if x.shape[0] == BAD_ROWS:
        y[:4] *= 1.2
    return y


@torch.library.custom_op("ndtest::rmsnorm_inplace_buggy", mutates_args=("x",))
def rmsnorm_inplace_buggy(x: torch.Tensor, w: torch.Tensor, eps: float) -> None:
    y = _norm(x, w, eps)
    if x.shape[0] == BAD_ROWS:
        y[:4] *= 1.2
    x.copy_(y)


@torch.library.custom_op("ndtest::rmsnorm_nan", mutates_args=())
def rmsnorm_nan(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """NaN in rows 0-3 on the 37-row call: the NaN path's, even with a twin registered (the twin compares finite
    positions only)."""
    y = _norm(x, w, eps)
    if x.shape[0] == BAD_ROWS:
        y[:4] = float("nan")
    return y


@torch.library.custom_op("ndtest::rmsnorm_noisy_buggy", mutates_args=())
def rmsnorm_noisy_buggy(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """1% off its twin on every call (an implementation gap, like an asm kernel vs a Triton one), and rows 0-3
    scaled by 1.2 on the 37-row call: the bug must be caught after the site's median error is established."""
    y = (_norm(x, w, eps).float() * 1.01).to(x.dtype)
    if x.shape[0] == BAD_ROWS:
        y[:4] *= 1.2
    return y


@torch.library.custom_op("ndtest::rmsnorm_ok", mutates_args=())
def rmsnorm_ok(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return _norm(x, w, eps)


def _twin_return(func, args, kwargs, pre, out):
    x, w, eps = args
    return {"got": {"return": out}, "ref": {"return": _norm(x, w, eps)}}


def _twin_inplace(func, args, kwargs, pre, out):
    x0 = ((pre or {}).get("args") or {}).get(0)
    if x0 is None:
        raise NotImplementedError("no pre-call copy of x")
    return {"got": {"x": args[0]}, "ref": {"x": _norm(x0, args[1], args[2])}}


twins.REGISTRY["ndtest.rmsnorm_buggy.default"] = [("test native norm", "native", _twin_return)]
twins.REGISTRY["ndtest.rmsnorm_ok.default"] = [("test native norm", "native", _twin_return)]
twins.REGISTRY["ndtest.rmsnorm_nan.default"] = [("test native norm", "native", _twin_return)]
twins.REGISTRY["ndtest.rmsnorm_noisy_buggy.default"] = [("test native norm", "native", _twin_return)]
twins.REGISTRY["ndtest.rmsnorm_inplace_buggy.default"] = [("test native norm", "native", _twin_inplace)]
