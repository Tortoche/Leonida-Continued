"""Value tests for RDNA2 64-bit global atomics (global_atomic_*_x2), run through spirv_mini_interp on the CPU.

M = op(M, DATA) on the 64-bit element at the global_* address (two dwords, low first); with glc the old 64 bits land in
vdst[0:1]. Lowered to native SPIR-V 64-bit atomics (OpAtomicIAdd, OpAtomicCompareExchange, OpAtomicUMin/SMin, etc.)
with OpCapability Int64Atomics. Standard library only; no GPU, no game files.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_global_atomics import OUT, TARGET, run, translate  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1


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


def atomic_x2(kind, m, d, glc=True, offset=0, target=TARGET):
    body = [f"v_mov_b32_e32 v1, {target - offset:#x}", f"v_mov_b32_e32 v2, {d & M32:#x}",
            f"v_mov_b32_e32 v3, {d >> 32:#x}", "v_mov_b32_e32 v6, 0xdead", "v_mov_b32_e32 v7, 0xbeef"]
    off = f" offset:{offset}" if offset else ""
    body.append(f"global_atomic_{kind}_x2 v[6:7], v1, v[2:3], s[0:1]{off} glc" if glc else
                f"global_atomic_{kind}_x2 v1, v[2:3], s[0:1]{off}")
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v8, {OUT:#x}", "global_store_dwordx2 v8, v[6:7], s[0:1]"]
    mem = run(body, {target: m & M32, target + 4: m >> 32})
    return (mem.get(OUT), mem.get(OUT + 4)), (mem.get(target), mem.get(target + 4))


def atomic_cmpswap_x2(m, data, cmp_val, glc=True, offset=0):
    body = [f"v_mov_b32_e32 v1, {TARGET - offset:#x}",
            f"v_mov_b32_e32 v2, {data & M32:#x}",
            f"v_mov_b32_e32 v3, {data >> 32:#x}",
            f"v_mov_b32_e32 v4, {cmp_val & M32:#x}",
            f"v_mov_b32_e32 v5, {cmp_val >> 32:#x}",
            "v_mov_b32_e32 v6, 0xdead", "v_mov_b32_e32 v7, 0xbeef"]
    off = f" offset:{offset}" if offset else ""
    body.append(f"global_atomic_cmpswap_x2 v[6:7], v1, v[2:5], s[0:1]{off} glc" if glc else
                f"global_atomic_cmpswap_x2 v1, v[2:5], s[0:1]{off}")
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v8, {OUT:#x}", "global_store_dwordx2 v8, v[6:7], s[0:1]"]
    mem = run(body, {TARGET: m & M32, TARGET + 4: m >> 32})
    return (mem.get(OUT), mem.get(OUT + 4)), (mem.get(TARGET), mem.get(TARGET + 4))


def split(v):
    return v & M32, v >> 32


class GlobalAtomicsX2(unittest.TestCase):
    def test_returning_forms(self):
        for kind, fn in MODEL.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        self.assertEqual(atomic_x2(kind, m, d), (split(m), split(fn(m, d))))

    def test_cmpswap_x2_match_and_mismatch(self):
        m, d = 0x12345678_9ABCDEF0, 0xAABBCCDD_11223344
        # Match: memory changes to data, returns old m
        self.assertEqual(atomic_cmpswap_x2(m, d, m, glc=True), (split(m), split(d)))
        # Mismatch: memory unchanged, returns old m
        self.assertEqual(atomic_cmpswap_x2(m, d, 0xDEADBEEF, glc=True), (split(m), split(m)))
        # Without glc: vdst preserved, memory updated
        self.assertEqual(atomic_cmpswap_x2(m, d, m, glc=False), ((0xDEAD, 0xBEEF), split(d)))

    def test_without_glc_vdst_is_kept(self):
        m, d = 0xF0F0F0F0_0F0F0F0F, 0xFF00FF00_00FF00FF
        for kind, fn in MODEL.items():
            with self.subTest(op=kind):
                self.assertEqual(atomic_x2(kind, m, d, glc=False), ((0xDEAD, 0xBEEF), split(fn(m, d))))

    def test_offset_and_seeded(self):
        self.assertEqual(atomic_x2("xor", 0x11111111_22222222, 0x0000FFFF_FFFF0000, offset=-8),
                         (split(0x11111111_22222222), split(0x11111111_22222222 ^ 0x0000FFFF_FFFF0000)))
        self.assertEqual(atomic_x2("add", 0x00000001_FFFFFFFF, 1, offset=-8),
                         (split(0x00000001_FFFFFFFF), split(0x00000002_00000000)))
        rng = random.Random(0x6A72)
        for _ in range(20):
            kind, m, d = rng.choice(list(MODEL)), rng.getrandbits(64), rng.getrandbits(64)
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(atomic_x2(kind, m, d), (split(m), split(MODEL[kind](m, d))))

    def test_unaligned_address_returns_zero(self):
        m, d = 0x12345678_9ABCDEF0, 0x55555555_AAAAAAA0
        # TARGET + 4 is unaligned for 8-byte atomic (addr % 8 == 4)
        vdst, mem = atomic_x2("add", m, d, target=TARGET + 4)
        self.assertEqual(vdst, (0, 0))
        self.assertEqual(mem, split(m))

    def test_rejected_forms(self):
        for kind in ("inc", "dec", "csub", "fmin", "fmax"):
            with self.subTest(op=kind):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", f"global_atomic_{kind}_x2 v[6:7], v1, v[2:3], s[0:1] glc"])
        # cmpswap_x2 requires 4 VGPRs for vdata, not 2
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["v_mov_b32_e32 v1, 0", "global_atomic_cmpswap_x2 v[6:7], v1, v[2:3], s[0:1] glc"])
        # single VGPR destination is not valid for 64-bit atomic
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["v_mov_b32_e32 v1, 0", "global_atomic_or_x2 v6, v1, v[2:3], s[0:1] glc"])


if __name__ == "__main__":
    unittest.main()
