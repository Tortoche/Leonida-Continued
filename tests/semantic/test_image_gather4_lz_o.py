"""Regression tests for image_gather4_lz_o (2D, 2D_ARRAY, a16, and error guards)."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"], check=True)
        subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", sv], check=True)


class ImageGather4LzOTests(unittest.TestCase):
    def test_fixture_2d_and_2d_array(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "image_gather4_lz_o.asm"
        spv = support.translate_lines("ps_" + fixture.stem, fixture.read_text(encoding="ascii").splitlines())

        self.assertIn("OpCapability ImageGatherExtended", spv)
        self.assertRegex(spv, r"OpBitFieldSExtract %uint %\w+ %c0 %c6")
        self.assertRegex(spv, r"OpBitFieldSExtract %uint %\w+ %c8 %c6")

        gathers = [ln.strip() for ln in spv.splitlines() if "= OpImageGather " in ln]
        # 2D routes through 3 typed aliases (int, uint, float) with component 1; 2D_ARRAY emits 1 float gather with component 2.
        self.assertEqual(len(gathers), 4)
        self.assertTrue(all(" Offset " in g for g in gathers))
        self.assertEqual(sum(1 for g in gathers if re.search(r" %c1 Offset %\w+$", g)), 3)
        self.assertEqual(sum(1 for g in gathers if re.search(r" %c2 Offset %\w+$", g)), 1)
        _validate(spv)

    def test_a16_form_and_unsupported_guards(self) -> None:
        spv = support.translate_lines(
            "ps_gather4_lz_o_a16",
            [
                "v_mov_b32 v4, 0x0102",
                "v_mov_b32 v5, 0x3c003800",
                "image_gather4_lz_o v[0:3], v[4:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
                "exp mrt0 v0, v1, v2, v3 done vm",
                "s_endpgm",
            ],
        )
        self.assertIn("OpCapability ImageGatherExtended", spv)
        self.assertIn("Offset", spv)
        _validate(spv)

        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines(
                "ps_gather4_lz_o_a16_tfe",
                [
                    "v_mov_b32 v4, 0x0102",
                    "v_mov_b32 v5, 0x3c003800",
                    "image_gather4_lz_o v[0:4], v[4:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 tfe",
                    "s_endpgm",
                ],
            )

        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines(
                "ps_gather4_lz_o_bad_dmask",
                [
                    "v_mov_b32 v4, 0x0102",
                    "v_mov_b32 v5, 0.25",
                    "v_mov_b32 v6, 0.75",
                    "image_gather4_lz_o v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D",
                    "s_endpgm",
                ],
            )


if __name__ == "__main__":
    unittest.main()
