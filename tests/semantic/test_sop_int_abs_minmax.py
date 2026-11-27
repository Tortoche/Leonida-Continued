from __future__ import annotations

"""Semantic interpreter tests for scalar integer abs/min/max opcodes:
s_abs_i32, s_min_i32, s_max_i32, s_min_u32, s_max_u32.
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

FIXTURE = Path(HERE) / "fixtures" / "sop_int_abs_minmax.asm"
MASK32 = 0xFFFFFFFF


def _s32(x: int) -> int:
    x &= MASK32
    return x - 0x100000000 if x >= 0x80000000 else x


def _translate():
    lines = [s for s in FIXTURE.read_text(encoding="utf-8").splitlines() if s and not s.startswith(";")]
    return _translate_cs("cs_sop_int_abs_minmax", lines, threads=32, emulate=False, structured=True)


class TestSopIntAbsMinMax(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spirv_text = _translate()[0]

    def _run(self, a: int, b: int) -> list[int]:
        mem = run_workgroup(
            self.spirv_text,
            1,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i, "%local_index": i},
            push_constants=[0, 0, 32, 0x31016FAC, a & MASK32, b & MASK32] + [0] * 26,
            memory={},
        )
        return [mem[off] for off in (0, 4, 8, 12, 16)]

    def test_positive_and_negative(self) -> None:
        cases = [
            (0, 0),
            (42, 17),
            (-15, 7),
            (-100, -25),
            (0x80000000, 1),
            (0x7FFFFFFF, 0x80000000),
        ]
        for a, b in cases:
            au, bu = a & MASK32, b & MASK32
            ai, bi = _s32(au), _s32(bu)
            expected = [
                abs(ai) & MASK32,
                min(ai, bi) & MASK32,
                max(ai, bi) & MASK32,
                min(au, bu),
                max(au, bu),
            ]
            with self.subTest(a=hex(au), b=hex(bu)):
                self.assertEqual(self._run(au, bu), expected)


if __name__ == "__main__":
    unittest.main()
