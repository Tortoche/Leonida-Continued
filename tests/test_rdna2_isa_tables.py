import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tools"))
import rdna2_isa_tables as tables  # noqa: E402

EXPECTED_TABLES = (
    "DS", "FLAT", "GLOBAL", "MIMG", "MTBUF", "MUBUF", "SCRATCH",
    "SMEM", "SOP1", "SOP2", "SOPC", "SOPK", "SOPP", "VINTRP",
    "VOP1", "VOP2", "VOP3", "VOP3B", "VOP3P", "VOPC",
)


class TestRdna2IsaTables(unittest.TestCase):
    def test_all_expected_tables_present_and_non_empty(self):
        for name in EXPECTED_TABLES:
            table = getattr(tables, name, None)
            self.assertIsInstance(table, dict, name)
            self.assertGreater(len(table), 0, name)

    def test_table_entries_have_valid_schema(self):
        for name in EXPECTED_TABLES:
            table = getattr(tables, name)
            for opcode, entry in table.items():
                self.assertIsInstance(opcode, int, f"{name}: {opcode!r}")
                self.assertGreaterEqual(opcode, 0, f"{name}: {opcode!r}")
                self.assertIsInstance(entry, tuple, f"{name}[{opcode:#x}]")
                self.assertEqual(len(entry), 2, f"{name}[{opcode:#x}]")
                mnemonic, operands = entry
                self.assertIsInstance(mnemonic, str, f"{name}[{opcode:#x}]")
                self.assertTrue(mnemonic, f"{name}[{opcode:#x}]")
                self.assertIsInstance(operands, tuple, f"{name}[{opcode:#x}]")
                for op in operands:
                    self.assertIsInstance(op, tuple, f"{name}[{opcode:#x}]")
                    self.assertEqual(len(op), 3, f"{name}[{opcode:#x}]")
                    field_name, width, kind = op
                    self.assertIsInstance(field_name, str)
                    self.assertIsInstance(width, int)
                    self.assertGreaterEqual(width, -1)
                    self.assertIsInstance(kind, str)

    def test_known_opcodes(self):
        self.assertEqual(tables.DS[0x14][0], "ds_nop")
        self.assertEqual(tables.SOPP[0x01][0], "s_endpgm")
        self.assertEqual(tables.SOP2[0x00][0], "s_add_u32")
        self.assertEqual(tables.VOP2[0x03][0], "v_add_f32")


if __name__ == "__main__":
    unittest.main()
