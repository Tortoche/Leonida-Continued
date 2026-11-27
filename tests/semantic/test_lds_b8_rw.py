from __future__ import annotations

"""Semantic interpreter tests for LDS 8-bit read/write:
ds_write_b8, ds_read_u8, ds_read_i8.
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

FIXTURE = Path(HERE) / "fixtures" / "lds_b8_rw.asm"
MASK32 = 0xFFFFFFFF


def _translate():
    lines = [s for s in FIXTURE.read_text(encoding="utf-8").splitlines() if s and not s.startswith(";")]
    return _translate_cs("cs_lds_b8_rw", lines, threads=32, emulate=False, structured=True)[0]


class TestLdsB8Rw(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spirv_text = _translate()

    def _run(self, val: int) -> list[int]:
        mem = run_workgroup(
            self.spirv_text,
            1,
            32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i, "%local_index": i},
            push_constants=[0, 0, 32, 0x31016FAC, val & MASK32] + [0] * 27,
            memory={},
        )
        return [mem[0], mem[4]]

    def test_b8_rw(self) -> None:
        cases = [
            (0x00, 0x00000000, 0x00000000),
            (0x7F, 0x0000007F, 0x0000007F),
            (0x80, 0x00000080, 0xFFFFFF80),
            (0x85, 0x00000085, 0xFFFFFF85),
            (0xFF, 0x000000FF, 0xFFFFFFFF),
        ]
        for val, exp_u8, exp_i8 in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(self._run(val), [exp_u8, exp_i8])


if __name__ == "__main__":
    unittest.main()
