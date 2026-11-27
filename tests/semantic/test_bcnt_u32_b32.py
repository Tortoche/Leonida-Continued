"""Semantic tests for v_bcnt_u32_b32."""

from pathlib import Path
import unittest

from support import HERE, mrt0_words, translate_listing


class BcntU32B32Test(unittest.TestCase):
    def test_bcnt_u32_b32_fixture(self):
        fixture = Path(HERE) / "fixtures" / "bcnt_u32_b32.asm"
        text = translate_listing("ps_bcnt_u32_b32", fixture.read_bytes())
        self.assertEqual(mrt0_words(text), [0, 32, 26, 21])
