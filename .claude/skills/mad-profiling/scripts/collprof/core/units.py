"""Sizes, datatypes and the few formatting helpers the report and the CSVs share."""

from __future__ import annotations

GIB = 1024 ** 3
MIB = 1024 ** 2

#: RCCL/NCCL datatype enum -> (name, size in bytes). The log prints the enum value, and the message
#: size is the element count times this width, so a wrong entry scales a whole report.
DATATYPES = {
    0: ("int8", 1),
    1: ("uint8", 1),
    2: ("int32", 4),
    3: ("uint32", 4),
    4: ("int64", 8),
    5: ("uint64", 8),
    6: ("fp16", 2),
    7: ("fp32", 4),
    8: ("fp64", 8),
    9: ("bf16", 2),
    10: ("fp8_e4m3", 1),
    11: ("fp8_e5m2", 1),
}

#: torch dtype name -> width, for the sizes carried by trace events.
TORCH_DTYPE_BYTES = {
    "Byte": 1, "Char": 1, "Bool": 1, "Float8_e4m3fn": 1, "Float8_e4m3fnuz": 1,
    "Float8_e5m2": 1, "Float8_e5m2fnuz": 1, "Half": 2, "BFloat16": 2, "Short": 2,
    "Int": 4, "Float": 4, "Long": 8, "Double": 8,
}

#: Below this a collective's cost is dominated by launch and handshake latency rather than by the
#: bytes it moves, so its share of the volume understates its share of the step.
LATENCY_BOUND_BYTES = MIB


def fmt_bytes(n: float) -> str:
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n:.0f} B"


def fmt_per_rank_calls(calls: float, reps: int) -> str:
    """Per-rank call count. A collective a single rank issues must not round down to zero."""
    per_rank = calls / max(reps, 1)
    if per_rank and per_rank < 1:
        return "<1"
    return f"{round(per_rank)}"


def datatype(enum_value: int) -> tuple[str, int]:
    """Name and width of an RCCL datatype enum, tolerating a value this build does not know."""
    return DATATYPES.get(enum_value, (f"dt{enum_value}", 4))
