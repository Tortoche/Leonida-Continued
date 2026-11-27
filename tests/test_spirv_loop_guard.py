import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tools"))
import spirv_loop_guard  # noqa: E402


class TestSpirvLoopGuard(unittest.TestCase):
    def test_parse_limit_valid_and_invalid(self):
        self.assertEqual(spirv_loop_guard.parse_limit("0"), 0)
        self.assertEqual(spirv_loop_guard.parse_limit("65536"), 65536)
        self.assertEqual(spirv_loop_guard.parse_limit(str(0xFFFFFFFF)), 0xFFFFFFFF)
        for bad in ("-1", str(0x100000000)):
            with self.assertRaises(argparse.ArgumentTypeError):
                spirv_loop_guard.parse_limit(bad)
        with self.assertRaises(ValueError):
            spirv_loop_guard.parse_limit("abc")

    def test_add_guards_noop_and_bounds(self):
        asm = "%fn = OpFunction %void None %ft\n%entry = OpLabel\nOpReturn\nOpFunctionEnd\n"
        self.assertEqual(spirv_loop_guard.add_guards(asm), asm)
        self.assertEqual(spirv_loop_guard.add_guards(asm, limit=0), asm)
        with self.assertRaises(ValueError):
            spirv_loop_guard.add_guards(asm, limit=-1)
        with self.assertRaises(ValueError):
            spirv_loop_guard.add_guards(asm, limit=0x100000000)

    def test_unconditional_back_edge_adds_types_constants_and_guard(self):
        asm = (
            "%fn = OpFunction %void None %ft\n"
            "%entry = OpLabel\n"
            "OpBranch %hdr\n"
            "%hdr = OpLabel\n"
            "OpLoopMerge %exit %cont None\n"
            "OpBranch %cont\n"
            "%cont = OpLabel\n"
            "OpBranch %hdr\n"
            "%exit = OpLabel\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        out = spirv_loop_guard.add_guards(asm, limit=16)
        self.assertIn("%loop_guard_uint = OpTypeInt 32 0", out)
        self.assertIn("%loop_guard_bool = OpTypeBool", out)
        self.assertIn("%loop_guard_c1 = OpConstant %loop_guard_uint 1", out)
        self.assertIn("%loop_guard_c16 = OpConstant %loop_guard_uint 16", out)
        self.assertIn("%loop_guard_0_count = OpPhi %loop_guard_uint %loop_guard_c1 %entry %loop_guard_0_next %cont", out)
        self.assertIn("OpBranchConditional %loop_guard_0_room %hdr %exit", out)

    def test_conditional_back_edge_with_exit_and_split_latch(self):
        asm_direct = (
            "%u32 = OpTypeInt 32 0\n"
            "%bool = OpTypeBool\n"
            "%c1 = OpConstant %u32 1\n"
            "%c8 = OpConstant %u32 8\n"
            "%fn = OpFunction %void None %ft\n"
            "%entry = OpLabel\n"
            "OpBranch %hdr\n"
            "%hdr = OpLabel\n"
            "OpLoopMerge %exit %hdr None\n"
            "OpBranchConditional %cond %hdr %exit\n"
            "%exit = OpLabel\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        out_direct = spirv_loop_guard.add_guards(asm_direct, limit=8)
        self.assertIn("%loop_guard_0_take = OpLogicalAnd %bool %cond %loop_guard_0_room", out_direct)
        self.assertIn("OpBranchConditional %loop_guard_0_take %hdr %exit", out_direct)

        asm_split = (
            "%u32 = OpTypeInt 32 0\n"
            "%bool = OpTypeBool\n"
            "%c1 = OpConstant %u32 1\n"
            "%c8 = OpConstant %u32 8\n"
            "%fn = OpFunction %void None %ft\n"
            "%entry = OpLabel\n"
            "OpBranch %hdr\n"
            "%hdr = OpLabel\n"
            "%val = OpPhi %u32 %c1 %entry %c8 %cont\n"
            "OpLoopMerge %exit %cont None\n"
            "OpBranch %cont\n"
            "%cont = OpLabel\n"
            "OpBranchConditional %cond %hdr %inner\n"
            "%inner = OpLabel\n"
            "OpBranch %exit\n"
            "%exit = OpLabel\n"
            "%out = OpPhi %u32 %c1 %inner\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        out_split = spirv_loop_guard.add_guards(asm_split, limit=8)
        self.assertIn("%loop_guard_0_latch = OpLabel", out_split)
        self.assertIn("%val = OpPhi %u32 %c1 %entry %c8 %loop_guard_0_latch", out_split)
        self.assertIn("%loop_guard_undef0 = OpUndef %u32", out_split)
        self.assertIn("%out = OpPhi %u32 %c1 %inner %loop_guard_undef0 %loop_guard_0_latch", out_split)

    def test_prefix_collision_avoided(self):
        asm = (
            "; %loop_guard_ marker\n"
            "%fn = OpFunction %void None %ft\n"
            "%entry = OpLabel\n"
            "OpBranch %hdr\n"
            "%hdr = OpLabel\n"
            "OpLoopMerge %exit %hdr None\n"
            "OpBranch %hdr\n"
            "%exit = OpLabel\n"
            "OpReturn\n"
            "OpFunctionEnd\n"
        )
        out = spirv_loop_guard.add_guards(asm, limit=4)
        self.assertIn("%loop_guard__uint = OpTypeInt 32 0", out)


if __name__ == "__main__":
    unittest.main()
