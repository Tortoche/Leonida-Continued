"""Semantic tests for v_alignbit_b32."""

from pathlib import Path
import unittest

from support import HERE, mrt0_words, translate_listing


class AlignbitB32Test(unittest.TestCase):
    def test_alignbit_b32_fixture(self):
        fixture = Path(HERE) / "fixtures" / "alignbit_b32.asm"
        text = translate_listing("ps_alignbit_b32", fixture.read_bytes())
        self.assertEqual(mrt0_words(text), [0x9ABCDEF0, 0x789ABCDE, 0x56789ABC, 0x3456789A])
