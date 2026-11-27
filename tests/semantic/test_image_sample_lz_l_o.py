"""Regression tests for image_sample_lz_o, image_sample_l_o, and image_sample_c_lz_o."""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import support

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"], check=True)
        subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv], check=True)


class ImageSampleLzLOTests(unittest.TestCase):
    def test_ps_fixture_explicit_lod_and_offset(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "image_sample_lz_l_o.asm"
        spv = support.translate_lines("ps_" + fixture.stem, fixture.read_text(encoding="ascii").splitlines())

        self.assertRegex(spv, r"OpBitFieldSExtract %uint %\w+ %c0 %c6")
        self.assertRegex(spv, r"OpBitFieldSExtract %uint %\w+ %c8 %c6")

        samples = [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]
        self.assertEqual(len(samples), 3)
        self.assertRegex(samples[0], r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod\|Offset %f0 %\w+$")
        self.assertRegex(samples[1], r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod\|Offset %t\d+ %\w+$")
        self.assertRegex(samples[2], r"= OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %f0 %\w+$")
        _validate(spv)

    def test_cs_explicit_lod_offset_forms(self) -> None:
        for op, regs, pat in (
            ("image_sample_lz_o v[0:3]", "v[4:6]", r"OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod\|Offset %f0 %\w+$"),
            ("image_sample_l_o v[0:3]", "v[4:7]", r"OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod\|Offset %t\d+ %\w+$"),
            ("image_sample_c_lz_o v9", "v[4:7]", r"OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %f0 %\w+$"),
        ):
            with self.subTest(op=op):
                dmask = "0x1" if "_c_" in op else "0xf"
                spv = support.translate_lines(
                    "cs_sample_o",
                    [
                        "v_mov_b32 v4, 0x023f",
                        "v_mov_b32 v5, 0.25",
                        "v_mov_b32 v6, 0.5",
                        "v_mov_b32 v7, 1.0",
                        f"{op}, {regs}, s[8:15], s[16:19] dmask:{dmask} dim:SQ_RSRC_IMG_2D",
                        "s_endpgm",
                    ],
                    fssm=CS_FSSM,
                )
                samples = [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]
                self.assertEqual(len(samples), 1)
                self.assertRegex(samples[0], pat)
                _validate(spv)


if __name__ == "__main__":
    unittest.main()
