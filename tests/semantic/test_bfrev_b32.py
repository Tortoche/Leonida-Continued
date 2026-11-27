"""Semantic tests for v_bfrev_b32."""

from pathlib import Path
import unittest

from support import HERE, mrt0_words, translate_listing


class BfrevB32Test(unittest.TestCase):
    def test_bfrev_b32_fixture(self):
        fixture = Path(HERE) / "fixtures" / "bfrev_b32.asm"
        text = translate_listing("ps_bfrev_b32", fixture.read_bytes())
        self.assertEqual(mrt0_words(text), [0x00000000, 0x80000000, 0x00000001, 0x1E6A2C48])
