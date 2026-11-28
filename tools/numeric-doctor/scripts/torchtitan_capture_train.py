"""torchtitan.train with debug-numeric's op capture wired in, without editing torchtitan.

    torchrun ... scripts/torchtitan_capture_train.py --module <m> --config <c> ...

(in place of `-m torchtitan.train`, cwd = the torchtitan checkout, PYTHONPATH = it).
The capture is built at the first train_step (after model, optimizer and checkpoint
load: never during construction) and cap.step() runs at the top of every step on every
rank, outside any other collective (sync_stop all-reduces the stop flag there).  The
same shape fits any framework that owns its loop: patch the per-step method.
FlexAttention: set NAN_CAPTURE_SUSPEND=torchtitan.models.common.attention:FlexAttention.compiled_flex_attn
"""
import os
import sys

sys.path.insert(0, os.environ["NUMERIC_DOCTOR_ROOT"] + "/scripts")
from op_nan_capture import capture_from_env  # noqa: E402

from torchtitan.trainer import Trainer  # noqa: E402

_orig_train_step = Trainer.train_step
_cap = [None]


def train_step(self, data_iterator):
    if _cap[0] is None:
        _cap[0] = capture_from_env()
        every = int(os.environ.get("NAN_CAPTURE_REPLICAS_EVERY", "0"))
        if every and _cap[0].capture is not None:
            # T5: replicated parameters (DTensor Replicate dims) must stay identical.
            _cap[0].capture.watch_replicas(
                dict(("%d.%s" % (i, n), p) for i, part in enumerate(self.model_parts)
                     for n, p in part.named_parameters()), every=every)
    _cap[0].step()
    return _orig_train_step(self, data_iterator)


Trainer.train_step = train_step

if __name__ == "__main__":
    from torchtitan.train import main
    try:
        main()
    finally:
        if _cap[0] is not None:
            _cap[0].close()
