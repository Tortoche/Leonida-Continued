"""Value tests for RDNA2 64-bit buffer atomics (buffer_atomic_*_x2), run through spirv_mini_interp on the CPU.

M = op(M, DATA) on the 64-bit element (two dwords, low first) at the MUBUF address; with glc the old 64 bits come
back in vdata[0:1]. Lowered to native SPIR-V 64-bit atomics (OpAtomicIAdd, OpAtomicCompareExchange, OpAtomicUMin/SMin, etc.)
with OpCapability Int64Atomics. Bounds check covers the whole 8-byte element all or nothing (fixed_fmt=62).
Standard library only; no GPU, no game files.
"""
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)     # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)


def s64(v):
    return v - (1 << 64) if (v >> 63) else v


MODEL = {
    "and": lambda m, d: m & d,
    "or": lambda m, d: m | d,
    "xor": lambda m, d: m ^ d,
    "swap": lambda m, d: d,
    "add": lambda m, d: (m + d) & M64,
    "sub": lambda m, d: (m - d) & M64,
    "umin": lambda m, d: min(m, d),
    "umax": lambda m, d: max(m, d),
    "smin": lambda m, d: m if s64(m) <= s64(d) else d,
    "smax": lambda m, d: m if s64(m) >= s64(d) else d,
}
VALUES = [0, M64, 0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000, 0x12345678_9ABCDEF0,
          0x7FFFFFFF_FFFFFFFF, 0x80000000_00000000, 0xFEDCBA98_76543210]
ELEM, OUT, RECORDS = 0x40, 0x80, 0x100      # byte offsets in the buffer (base 0), num_records


def translate(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_batomic.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_batomic.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_batomic.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_batomic\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def run(body, memory):
    """Run `body` as one invocation over `memory` (the buffer base is 0: byte offsets are addresses)."""
    spv = translate(body)
    mem = dict(memory)
    si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def split(v):
    return v & M32, v >> 32


def atomic_x2(kind, m, d, glc=True, offset=ELEM, records=RECORDS):
    body = ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {records:#x}", "s_mov_b32 s11, 0x31016fac",
            f"v_mov_b32_e32 v1, {offset:#x}", f"v_mov_b32_e32 v2, {d & M32:#x}", f"v_mov_b32_e32 v3, {d >> 32:#x}",
            f"buffer_atomic_{kind}_x2 v[2:3], v1, s[8:11], 0 offen" + (" glc" if glc else ""), "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v4, {OUT:#x}", "s_mov_b32 s10, 0x1000", "buffer_store_dwordx2 v[2:3], v4, s[8:11], 0 offen"]
    mem = run(body, {offset: m & M32, offset + 4: m >> 32})
    return (mem.get(OUT), mem.get(OUT + 4)), (mem.get(offset), mem.get(offset + 4))


def atomic_cmpswap_x2(m, data, cmp_val, glc=True, offset=ELEM, records=RECORDS):
    body = ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {records:#x}", "s_mov_b32 s11, 0x31016fac",
            f"v_mov_b32_e32 v1, {offset:#x}",
            f"v_mov_b32_e32 v2, {data & M32:#x}", f"v_mov_b32_e32 v3, {data >> 32:#x}",
            f"v_mov_b32_e32 v4, {cmp_val & M32:#x}", f"v_mov_b32_e32 v5, {cmp_val >> 32:#x}",
            f"buffer_atomic_cmpswap_x2 v[2:5], v1, s[8:11], 0 offen" + (" glc" if glc else ""), "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}", "s_mov_b32 s10, 0x1000", "buffer_store_dwordx2 v[2:3], v6, s[8:11], 0 offen"]
    mem = run(body, {offset: m & M32, offset + 4: m >> 32})
    return (mem.get(OUT), mem.get(OUT + 4)), (mem.get(offset), mem.get(offset + 4))


class BufferAtomicsX2(unittest.TestCase):
    def test_returning_forms(self):
        for kind, fn in MODEL.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        self.assertEqual(atomic_x2(kind, m, d), (split(m), split(fn(m, d))))

    def test_cmpswap_x2_match_and_mismatch(self):
        m, d = 0x12345678_9ABCDEF0, 0xAABBCCDD_11223344
        # Match: memory updated, returns old m
        self.assertEqual(atomic_cmpswap_x2(m, d, m, glc=True), (split(m), split(d)))
        # Mismatch: memory unchanged, returns old m
        self.assertEqual(atomic_cmpswap_x2(m, d, 0xDEADBEEF, glc=True), (split(m), split(m)))
        # Without glc: vdata kept, memory updated
        self.assertEqual(atomic_cmpswap_x2(m, d, m, glc=False), (split(d), split(d)))

    def test_without_glc_vdata_is_kept(self):
        m, d = 0xF0F0F0F0_0F0F0F0F, 0xFF00FF00_00FF00FF
        for kind, fn in MODEL.items():
            with self.subTest(op=kind):
                self.assertEqual(atomic_x2(kind, m, d, glc=False), (split(d), split(fn(m, d))))

    def test_bounds_cover_the_whole_element(self):
        m, d = 0x11111111_22222222, 0x0F0F0F0F_F0F0F0F0
        # ends exactly at num_records: in range
        self.assertEqual(atomic_x2("xor", m, d, offset=RECORDS - 8), (split(m), split(m ^ d)))
        self.assertEqual(atomic_x2("add", m, d, offset=RECORDS - 8), (split(m), split((m + d) & M64)))
        # only the low dword in range, or none: memory unchanged, the returned value is 0 (as buffer_atomic)
        for offset in (RECORDS - 4, RECORDS):
            with self.subTest(offset=hex(offset)):
                self.assertEqual(atomic_x2("xor", m, d, offset=offset), ((0, 0), split(m)))

    def test_unaligned_address_returns_zero(self):
        m, d = 0x12345678_9ABCDEF0, 0x01010101_02020202
        # ELEM + 4 is unaligned (offset % 8 == 4)
        vdst, mem = atomic_x2("add", m, d, offset=ELEM + 4)
        self.assertEqual(vdst, (0, 0))
        self.assertEqual(mem, split(m))

    def test_seeded(self):
        rng = random.Random(0xB0A2)
        for _ in range(20):
            kind, m, d = rng.choice(list(MODEL)), rng.getrandbits(64), rng.getrandbits(64)
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(atomic_x2(kind, m, d), (split(m), split(MODEL[kind](m, d))))

    def test_bad_data_and_forms_rejected(self):
        for line in ("buffer_atomic_or_x2 v2, v1, s[8:11], 0 offen glc",
                     "buffer_atomic_cmpswap_x2 v[2:3], v1, s[8:11], 0 offen glc",
                     "buffer_atomic_inc_x2 v[2:3], v1, s[8:11], 0 offen glc",
                     "buffer_atomic_fmin_x2 v[2:3], v1, s[8:11], 0 offen glc"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])


if __name__ == "__main__":
    unittest.main()
