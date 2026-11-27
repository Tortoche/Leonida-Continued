from __future__ import annotations

"""Semantic interpreter tests for scalar bit opcodes:
s_bitcmp0_b32, s_bitcmp1_b32, s_bfe_i32, s_bfm_b32.
"""
import os
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_loop_ballot_barrier import _translate_cs  # noqa: E402
from spirv_mini_interp import run_workgroup  # noqa: E402

FIXTURE = Path(HERE) / "fixtures" / "sop_bit_ops.asm"
MASK32 = 0xFFFFFFFF


def _ref_bfe_i32(val: int, ctrl: int) -> int:
    offset = ctrl & 0x1F
    width = (ctrl >> 16) & 0x7F
    if width == 0:
        return 0
    if width >= 32:
        return (val >> offset) & MASK32
    field = (val >> offset) & ((1 << width) - 1)
    if field & (1 << (width - 1)):
        field |= (~((1 << width) - 1)) & MASK32
    return field & MASK32


def _ref_bfm_b32(width: int, offset: int) -> int:
    w = width & 0x1F
    o = offset & 0x1F
    return (((1 << w) - 1) << o) & MASK32


def _translate():
    lines = [s for s in FIXTURE.read_text(encoding="utf-8").splitlines() if s and not s.startswith(";")]
    return _translate_cs("cs_sop_bit_ops", lines, threads=32, emulate=False, structured=True)[0]


class TestSopBitOps(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spirv_text = _translate()

    def _run(self, s4: int, s5: int, s6: int, s7: int) -> list[int]:
        mem = run_workgroup(
            self.spirv_text,
            1,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i, "%local_index": i},
            push_constants=[0, 0, 32, 0x31016FAC, s4 & MASK32, s5 & MASK32, s6 & MASK32, s7 & MASK32] + [0] * 24,
            memory={},
        )
        return [mem[off] for off in (0, 4, 8, 12)]

    def test_bit_ops(self) -> None:
        cases = [
            (0x80000001, (4 << 16) | 0, 5, 3),
            (0x12345678, (8 << 16) | 4, 8, 12),
            (0xF0000000, (4 << 16) | 28, 0, 5),
            (0x00000000, (6 << 16) | 1, 16, 16),
        ]
        for val, ctrl, w, o in cases:
            bit = (val >> (ctrl & 0x1F)) & 1
            expected = [
                1 if bit == 0 else 0,
                1 if bit == 1 else 0,
                _ref_bfe_i32(val, ctrl),
                _ref_bfm_b32(w, o),
            ]
            with self.subTest(val=hex(val), ctrl=hex(ctrl), w=w, o=o):
                self.assertEqual(self._run(val, ctrl, w, o), expected)


if __name__ == "__main__":
    unittest.main()
