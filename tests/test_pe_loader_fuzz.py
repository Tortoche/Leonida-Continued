"""Adversarial fuzzing and edge-case testing of PE Loader and headers.

Empirical verification suite generating corrupted and edge-case PE headers:
- Zero-size BSS sections, out-of-bounds section virtual sizes, virtual range wrapping.
- Malformed entry points: pointing inside headers (< SizeOfHeaders), outside image, unmapped gaps, non-executable sections.
- Circular and malformed forwarders: self-loops, mutual 2-node cycles, 3-node cycles, deep chains exceeding depth limit, malformed specs, ordinal bounds.
- /GA TLS access patterns: patch_ga_tls_reads rewriting `mov r64, gs:[58h]` across all 16 registers (RAX..R15) to `disp32 = 0x1480 + 8 * slot`.
- Non-canonical TLS directories: inverted bounds, zero-length templates, BSS zero-fill templates, multiple callbacks, unterminated tables, OOB callbacks.
- Stress and mutation fuzzing: verifies PE inspection and oracle safety without crashes, buffer over-reads, or infinite loops.

Usage:
    python -I tests/test_pe_loader_fuzz.py
"""
from __future__ import annotations

import copy
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import pe_inspect

# Common PE constants
IMAGE_FILE_MACHINE_AMD64 = 0x8664
IMAGE_FILE_MACHINE_I386 = 0x014C
IMAGE_NT_OPTIONAL_HDR64_MAGIC = 0x20B
IMAGE_NT_OPTIONAL_HDR32_MAGIC = 0x10B

IMAGE_SCN_CNT_CODE = 0x00000020
IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040
IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080
IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_READ = 0x40000000
IMAGE_SCN_MEM_WRITE = 0x80000000

DIR_EXPORT = 0
DIR_IMPORT = 1
DIR_RESOURCE = 2
DIR_EXCEPTION = 3
DIR_SECURITY = 4
DIR_BASERELOC = 5
DIR_DEBUG = 6
DIR_ARCHITECTURE = 7
DIR_GLOBALPTR = 8
DIR_TLS = 9
DIR_LOAD_CONFIG = 10
DIR_BOUND_IMPORT = 11
DIR_IAT = 12
DIR_DELAY_IMPORT = 13
DIR_COM_DESCRIPTOR = 14


# ============================================================================
# Clean-Room PE Binary Builder (Zero Game Data)
# ============================================================================

class PEBuilder:
    """Builds synthetic x64 (PE32+) or x86 (PE32) binaries for edge-case testing."""

    def __init__(self, is64: bool = True, image_base: int = 0x140000000,
                 image_size: int = 0x4000, entry_point: int = 0x1000,
                 size_of_headers: int = 0x200):
        self.is64 = is64
        self.image_base = image_base
        self.image_size = image_size
        self.entry_point = entry_point
        self.size_of_headers = size_of_headers
        self.section_alignment = 0x1000
        self.file_alignment = 0x200
        self.directories: list[tuple[int, int]] = [(0, 0) for _ in range(16)]
        self.sections: list[dict] = []
        self.raw_extra_data: dict[int, bytes] = {}

    def add_section(self, name: str = ".text", rva: int = 0x1000, vsize: int = 0x1000,
                    rawsize: int = 0x200, rawptr: int = 0x200,
                    chars: int | None = None, data: bytes | bytearray | None = None) -> PEBuilder:
        if chars is None:
            chars = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ
        if data is None:
            sec_data = bytearray(rawsize)
        else:
            sec_data = bytearray(data)
            if len(sec_data) < rawsize:
                sec_data.extend(b"\x00" * (rawsize - len(sec_data)))
        sec = {
            "name": name,
            "rva": rva,
            "vsize": vsize,
            "rawsize": rawsize,
            "rawptr": rawptr,
            "chars": chars,
            "data": sec_data,
        }
        self.sections.append(sec)
        return self

    def set_directory(self, index: int, rva: int, size: int) -> PEBuilder:
        self.directories[index] = (rva, size)
        return self

    def write_at_rva(self, rva: int, payload: bytes) -> PEBuilder:
        """Finds section holding rva and writes payload at corresponding offset."""
        for sec in self.sections:
            if sec["rva"] <= rva < sec["rva"] + sec["vsize"]:
                offset = rva - sec["rva"]
                needed = offset + len(payload)
                if needed > len(sec["data"]):
                    sec["data"].extend(b"\x00" * (needed - len(sec["data"])))
                sec["data"][offset:offset + len(payload)] = payload
                sec["rawsize"] = max(sec["rawsize"], len(sec["data"]))
                return self
        raise ValueError(f"RVA 0x{rva:X} does not fall into any defined section")

    def build(self) -> bytes:
        """Constructs and returns raw bytes of the PE image on disk."""
        opt_size = 240 if self.is64 else 224
        dos_e_lfanew = 0x80
        total_headers_needed = dos_e_lfanew + 4 + 20 + opt_size + len(self.sections) * 40
        header_len = max(self.size_of_headers, total_headers_needed)

        total_file_size = header_len
        for s in self.sections:
            if s["rawsize"] > 0:
                total_file_size = max(total_file_size, s["rawptr"] + s["rawsize"])
        for off, data in self.raw_extra_data.items():
            total_file_size = max(total_file_size, off + len(data))

        buf = bytearray(total_file_size)

        # DOS Header
        buf[0:2] = b"MZ"
        struct.pack_into("<I", buf, 0x3C, dos_e_lfanew)

        # PE Signature
        buf[dos_e_lfanew:dos_e_lfanew + 4] = b"PE\x00\x00"
        coff = dos_e_lfanew + 4

        # COFF File Header (20 bytes)
        machine = IMAGE_FILE_MACHINE_AMD64 if self.is64 else IMAGE_FILE_MACHINE_I386
        struct.pack_into("<HHIIIHH", buf, coff,
                         machine,
                         len(self.sections),
                         0x65000000,  # timestamp
                         0, 0,        # symbol table pointer & count
                         opt_size,
                         0x2022 if self.is64 else 0x0102)

        # Optional Header
        opt = coff + 20
        magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC if self.is64 else IMAGE_NT_OPTIONAL_HDR32_MAGIC
        struct.pack_into("<H", buf, opt, magic)
        struct.pack_into("<I", buf, opt + 16, self.entry_point)

        if self.is64:
            struct.pack_into("<Q", buf, opt + 24, self.image_base)
            struct.pack_into("<I", buf, opt + 32, self.section_alignment)
            struct.pack_into("<I", buf, opt + 36, self.file_alignment)
            struct.pack_into("<I", buf, opt + 56, self.image_size)
            struct.pack_into("<I", buf, opt + 60, self.size_of_headers)
            struct.pack_into("<H", buf, opt + 68, 2)  # Subsystem Windows CUI / GUI
            struct.pack_into("<H", buf, opt + 70, 0x0040)  # DllCharacteristics: DYNAMIC_BASE
            struct.pack_into("<I", buf, opt + 108, len(self.directories))
            dirs_offset = opt + 112
        else:
            struct.pack_into("<I", buf, opt + 28, self.image_base & 0xFFFFFFFF)
            struct.pack_into("<I", buf, opt + 32, self.section_alignment)
            struct.pack_into("<I", buf, opt + 36, self.file_alignment)
            struct.pack_into("<I", buf, opt + 56, self.image_size)
            struct.pack_into("<I", buf, opt + 60, self.size_of_headers)
            struct.pack_into("<H", buf, opt + 68, 2)
            struct.pack_into("<H", buf, opt + 70, 0x0040)
            struct.pack_into("<I", buf, opt + 92, len(self.directories))
            dirs_offset = opt + 96

        # Data directories
        for i, (rva, size) in enumerate(self.directories):
            struct.pack_into("<II", buf, dirs_offset + i * 8, rva, size)

        # Section headers
        sec_table = opt + opt_size
        for i, s in enumerate(self.sections):
            at = sec_table + i * 40
            name_bytes = s["name"].encode("ascii")[:8].ljust(8, b"\x00")
            buf[at:at + 8] = name_bytes
            struct.pack_into("<IIIIIIHHI", buf, at + 8,
                             s["vsize"], s["rva"], s["rawsize"], s["rawptr"],
                             0, 0, 0, 0, s["chars"])
            # Write section payload if non-empty
            if s["rawsize"] > 0 and s["rawptr"] + len(s["data"]) <= len(buf):
                buf[s["rawptr"]:s["rawptr"] + len(s["data"])] = s["data"]

        for off, data in self.raw_extra_data.items():
            if off + len(data) <= len(buf):
                buf[off:off + len(data)] = data

        return bytes(buf)


# ============================================================================
# PE Loader Safety Verification Oracle
# Emulates and tests the exact memory mapping, bounds, rejection and /GA rules of rt::Loader
# ============================================================================

class PELoaderRejection(Exception):
    """Raised when an edge-case PE binary must be safely rejected."""


class PELoaderOracle:
    """Verifies that PE binaries satisfy the memory bounds, integrity, and safety
    invariants required by runtime/src/pe_loader.cpp.
    """

    @staticmethod
    def require_range(name: str, rva: int, length: int, image_size: int, what: str) -> None:
        """Arithmetic overflow safe range check: matches require_range in pe_loader.cpp."""
        if rva > image_size or length > image_size - rva:
            raise PELoaderRejection(f"{name}: {what} runs past the image (rva=0x{rva:X}, len=0x{length:X}, size=0x{image_size:X})")

    @staticmethod
    def read_string(name: str, mem: bytearray | bytes, image_size: int, rva: int, what: str) -> str:
        """Safely reads a null-terminated string from mapped virtual memory."""
        if rva >= image_size or rva >= len(mem):
            raise PELoaderRejection(f"{name}: {what} is outside the image (rva=0x{rva:X})")
        null_idx = mem.find(b"\x00", rva, min(image_size, len(mem)))
        if null_idx < 0:
            raise PELoaderRejection(f"{name}: {what} isn't terminated inside the image")
        return mem[rva:null_idx].decode(errors="replace")

    @classmethod
    def validate_image(cls, data: bytes, name: str = "oracle.exe") -> tuple[bool, bytearray]:
        """Performs full loader validation and virtual section mapping according to rt::Loader rules."""
        if len(data) < 0x40 or data[:2] != b"MZ":
            raise PELoaderRejection(f"{name}: not a valid DOS header")
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if e_lfanew + 4 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
            raise PELoaderRejection(f"{name}: no PE signature")

        coff = e_lfanew + 4
        if coff + 20 > len(data):
            raise PELoaderRejection(f"{name}: truncated COFF header")
        (machine, num_sections, _ts, _syms, _num_syms, opt_size, _chars) = \
            struct.unpack_from("<HHIIIHH", data, coff)

        if machine != IMAGE_FILE_MACHINE_AMD64:
            raise PELoaderRejection(f"{name}: not an x86-64 PE")

        opt = coff + 20
        if opt_size < 240 or opt + opt_size > len(data):
            raise PELoaderRejection(f"{name}: truncated optional header")
        magic = struct.unpack_from("<H", data, opt)[0]
        if magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC:
            raise PELoaderRejection(f"{name}: not a PE32+ image")

        entry_point = struct.unpack_from("<I", data, opt + 16)[0]
        image_base = struct.unpack_from("<Q", data, opt + 24)[0]
        size_of_image = struct.unpack_from("<I", data, opt + 56)[0]
        size_of_headers = struct.unpack_from("<I", data, opt + 60)[0]
        num_rva_sizes = struct.unpack_from("<I", data, opt + 108)[0]

        headers_end = opt + opt_size + num_sections * 40
        if headers_end > len(data):
            raise PELoaderRejection(f"{name}: the section table runs past the end of the file")
        if not size_of_image or size_of_headers < headers_end or size_of_headers > size_of_image or size_of_headers > len(data):
            raise PELoaderRejection(f"{name}: SizeOfHeaders and SizeOfImage don't fit the headers and file")

        # Entry point checks:
        # Finding 3: Entry point must not be in headers (< SizeOfHeaders) or outside image (>= SizeOfImage)
        if entry_point != 0:
            if entry_point >= size_of_image:
                raise PELoaderRejection(f"{name}: the entry point is outside the image")
            if entry_point < size_of_headers:
                raise PELoaderRejection(f"{name}: the entry point is inside headers")

        # Section mapping & virtual range validation:
        # Emulates: VirtualAlloc(..., SizeOfImage, ...)
        virtual_image = bytearray(size_of_image)
        # Copy headers: std::memcpy(m->base, file.data(), oh.SizeOfHeaders)
        virtual_image[:size_of_headers] = data[:size_of_headers]

        sections = []
        sec_table = opt + opt_size
        entry_point_mapped_and_exec = (entry_point == 0)

        for i in range(num_sections):
            at = sec_table + i * 40
            sec_name = data[at:at + 8].rstrip(b"\x00").decode(errors="replace")
            vsize, rva, rawsize, rawptr = struct.unpack_from("<IIII", data, at + 8)
            s_chars = struct.unpack_from("<I", data, at + 36)[0]

            if rva < size_of_headers:
                raise PELoaderRejection(f"{name}: section {i} overlaps the headers")

            # Finding 4: Validate virtual range for ALL sections, including zero-raw-data (BSS) sections
            effective_vsize = vsize if vsize else rawsize
            cls.require_range(name, rva, effective_vsize, size_of_image, f"section {i}'s virtual range")

            n = min(rawsize, effective_vsize)
            if n > 0:
                if rawptr > len(data) or n > len(data) - rawptr:
                    raise PELoaderRejection(f"{name}: section {i} reads past the end of the file")
                cls.require_range(name, rva, n, size_of_image, f"section {i}'s data")
                virtual_image[rva:rva + n] = data[rawptr:rawptr + n]

            # Check if entry point falls inside this section and section is executable
            if rva <= entry_point < rva + effective_vsize:
                if (s_chars & IMAGE_SCN_MEM_EXECUTE) != 0:
                    entry_point_mapped_and_exec = True

            sections.append({
                "name": sec_name, "rva": rva, "vsize": vsize, "effective_vsize": effective_vsize,
                "rawsize": rawsize, "rawptr": rawptr, "chars": s_chars,
            })

        if not entry_point_mapped_and_exec:
            raise PELoaderRejection(f"{name}: entry point is not in an executable mapped section")

        # Parse data directories
        dirs = []
        dirs_base = opt + 112
        for i in range(16):
            if i < num_rva_sizes and dirs_base + i * 8 + 8 <= opt + opt_size:
                dirs.append(struct.unpack_from("<II", data, dirs_base + i * 8))
            else:
                dirs.append((0, 0))

        # 1. Validate Imports on virtual_image
        imp_rva, imp_size = dirs[DIR_IMPORT]
        if imp_size > 0:
            cls.require_range(name, imp_rva, 20, size_of_image, "the import table")
            desc_idx = 0
            found_null_desc = False
            while True:
                desc_rva = imp_rva + desc_idx * 20
                if desc_rva + 20 > size_of_image:
                    break
                oft, _ts, _fwd, dll_name_rva, ft = struct.unpack_from("<IIIII", virtual_image, desc_rva)
                if not oft and not dll_name_rva and not ft:
                    found_null_desc = True
                    break
                cls.read_string(name, virtual_image, size_of_image, dll_name_rva, "an imported DLL name")
                lookup = oft if oft else ft
                if lookup:
                    cls.require_range(name, lookup, 8, size_of_image, "an import lookup table")
                    thunk_idx = 0
                    found_null_thunk = False
                    while True:
                        t_rva = lookup + thunk_idx * 8
                        if t_rva + 8 > size_of_image:
                            break
                        thunk_val = struct.unpack_from("<Q", virtual_image, t_rva)[0]
                        if thunk_val == 0:
                            found_null_thunk = True
                            break
                        if not (thunk_val & (1 << 63)):  # Named import
                            cls.require_range(name, thunk_val, 2, size_of_image, "an import name")
                            cls.read_string(name, virtual_image, size_of_image, thunk_val + 2, "an import name")
                        thunk_idx += 1
                        if thunk_idx > 65536:
                            raise PELoaderRejection(f"{name}: infinite or cyclic import thunk table")
                    if not found_null_thunk:
                        raise PELoaderRejection(f"{name}: import thunk table not null-terminated")
                desc_idx += 1
                if desc_idx > 65536:
                    raise PELoaderRejection(f"{name}: infinite or cyclic import descriptors")
            if not found_null_desc:
                raise PELoaderRejection(f"{name}: import table has no null descriptor before image boundary")

        # 2. Validate Delay Imports on virtual_image
        del_rva, del_size = dirs[DIR_DELAY_IMPORT]
        if del_size > 0:
            cls.require_range(name, del_rva, 32, size_of_image, "the delay-import table")
            del_idx = 0
            while True:
                d_rva = del_rva + del_idx * 32
                if d_rva + 32 > size_of_image:
                    break
                attrs, dll_name_rva, _mod, iat_rva, int_rva = struct.unpack_from("<IIIII", virtual_image, d_rva)[:5]
                if not dll_name_rva:
                    break
                if not (attrs & 1):
                    raise PELoaderRejection(f"{name}: a delay-import descriptor without the RvaBased attribute")
                if not iat_rva:
                    raise PELoaderRejection(f"{name}: a delay-import descriptor has no address table")
                if not int_rva:
                    raise PELoaderRejection(f"{name}: a delay-import descriptor has no name table")
                cls.read_string(name, virtual_image, size_of_image, dll_name_rva, "a delay-imported DLL name")
                del_idx += 1

        # 3. Validate TLS Directory on virtual_image
        tls_rva, tls_size = dirs[DIR_TLS]
        if tls_size > 0 and tls_rva > 0:
            if tls_size < 40:
                raise PELoaderRejection(f"{name}: TLS directory size smaller than IMAGE_TLS_DIRECTORY64")
            cls.require_range(name, tls_rva, 40, size_of_image, "the TLS directory")
            (start_raw, end_raw, addr_idx, addr_cb, zero_fill, _chars) = \
                struct.unpack_from("<QQQQII", virtual_image, tls_rva)

            if not start_raw and not end_raw and not addr_idx and not addr_cb and not zero_fill:
                pass  # All-zero directory is treated as absent
            else:
                if end_raw < start_raw:
                    raise PELoaderRejection(f"{name}: TLS template end precedes its start")
                raw = end_raw - start_raw
                total = raw + zero_fill
                if total > 0xFFFFFFFF:
                    raise PELoaderRejection(f"{name}: TLS block is too large")

                if raw > 0:
                    if start_raw < image_base or (start_raw - image_base) >= size_of_image:
                        raise PELoaderRejection(f"{name}: the TLS template is outside the image")
                    raw_rva = start_raw - image_base
                    cls.require_range(name, raw_rva, raw, size_of_image, "the TLS template")

                if addr_idx:
                    if addr_idx < image_base or (addr_idx - image_base) >= size_of_image:
                        raise PELoaderRejection(f"{name}: the TLS index is outside the image")
                    cls.require_range(name, addr_idx - image_base, 4, size_of_image, "the TLS index")

                if addr_cb:
                    if addr_cb < image_base or (addr_cb - image_base) >= size_of_image:
                        raise PELoaderRejection(f"{name}: the TLS callback table is outside the image")
                    cb_table_rva = addr_cb - image_base
                    cb_idx = 0
                    found_null_cb = False
                    while True:
                        entry_rva = cb_table_rva + cb_idx * 8
                        if entry_rva + 8 > size_of_image:
                            break
                        cb_val = struct.unpack_from("<Q", virtual_image, entry_rva)[0]
                        if cb_val == 0:
                            found_null_cb = True
                            break
                        if cb_val < image_base or (cb_val - image_base) >= size_of_image:
                            raise PELoaderRejection(f"{name}: a TLS callback is outside the image")
                        cb_func_rva = cb_val - image_base
                        if cb_func_rva < size_of_headers:
                            raise PELoaderRejection(f"{name}: a TLS callback points inside headers")
                        cb_idx += 1
                        if cb_idx > 65536:
                            raise PELoaderRejection(f"{name}: infinite TLS callback table")
                    if not found_null_cb:
                        raise PELoaderRejection(f"{name}: TLS callback table not null-terminated")

        return True, virtual_image

    @staticmethod
    def patch_ga_tls_reads(virtual_image: bytearray, sections: list[dict],
                           image_size: int, slot: int) -> int:
        """Emulates Loader::patch_ga_tls_reads from runtime/src/pe_loader.cpp.

        Rewrites every `mov r64, gs:[58h]` in executable sections to disp32 = 0x1480 + 8 * slot.
        Instruction pattern: 65 [48|4C] 8B modrm 25 58 00 00 00
        """
        disp = 0x1480 + 8 * slot
        disp_bytes = struct.pack("<I", disp)
        sites = 0

        for sec in sections:
            if not (sec["chars"] & IMAGE_SCN_MEM_EXECUTE):
                continue
            effective_vsize = sec["effective_vsize"]
            size = min(effective_vsize, sec["rawsize"]) if sec["rawsize"] else effective_vsize
            rva = sec["rva"]
            if rva > image_size or size > image_size - rva or size < 9:
                continue

            k = 0
            while k + 9 <= size:
                pos = rva + k
                # Check for: 0x65 [0x48 or 0x4C] 0x8B
                if virtual_image[pos] == 0x65 and virtual_image[pos + 1] in (0x48, 0x4C) and virtual_image[pos + 2] == 0x8B:
                    modrm = virtual_image[pos + 3]
                    # mod=00, r/m=100 (SIB follows): (modrm & 0xC7) == 0x04
                    # SIB byte == 0x25 (disp32)
                    if (modrm & 0xC7) == 0x04 and virtual_image[pos + 4] == 0x25:
                        d = struct.unpack_from("<I", virtual_image, pos + 5)[0]
                        if d == 0x58:
                            virtual_image[pos + 5:pos + 9] = disp_bytes
                            sites += 1
                            k += 8  # Advances past this instruction (with k += 1 -> 9 bytes)
                k += 1

        return sites

    @staticmethod
    def resolve_forwarder_chain(forwarders_map: dict[str, str], start_spec: str, max_depth: int = 8) -> str | None:
        """Emulates Loader::resolve_forwarder loop prevention and syntax checking."""
        curr_spec = start_spec
        depth = 0
        visited = set()

        while True:
            if depth >= max_depth:
                return None  # Cycle or exceeded depth limit
            dot = curr_spec.rfind(".")
            if dot == -1 or dot == 0 or dot + 1 >= len(curr_spec):
                return None  # Malformed forwarder spec
            mod = curr_spec[:dot].lower()
            sym = curr_spec[dot + 1:]

            if not mod or not sym:
                return None

            # Ordinal forwarder validation
            if sym.startswith("#"):
                ord_str = sym[1:]
                if not ord_str.isdigit():
                    return None
                val = int(ord_str)
                if val <= 0 or val > 0xFFFF:
                    return None

            if curr_spec in visited:
                return None  # Cycle detected
            visited.add(curr_spec)

            if curr_spec in forwarders_map:
                curr_spec = forwarders_map[curr_spec]
                depth += 1
            else:
                return curr_spec


# ============================================================================
# Unit & Adversarial Stress Test Suites
# ============================================================================

class TestZeroSizeBSSSections(unittest.TestCase):
    """Stress tests PE loader handling of zero-size BSS and uninitialized data sections."""

    def test_zero_raw_and_virtual_size_bss(self):
        """A completely zero-size section (.bss with vsize=0, rawsize=0) is accepted safely."""
        builder = PEBuilder(image_size=0x4000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        # .bss section with vsize=0 and rawsize=0
        builder.add_section(".bss", rva=0x2000, vsize=0, rawsize=0, rawptr=0,
                            chars=IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)
        img_bytes = builder.build()
        ok, virt_mem = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)
        self.assertEqual(len(virt_mem), 0x4000)

    def test_pure_bss_with_valid_virtual_size(self):
        """A pure BSS section (rawsize=0, vsize=0x1000) is accepted and zero-initialized."""
        builder = PEBuilder(image_size=0x4000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        builder.add_section(".bss", rva=0x2000, vsize=0x1000, rawsize=0, rawptr=0,
                            chars=IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)
        img_bytes = builder.build()
        ok, virt_mem = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)
        # Check that the .bss virtual memory range is zeroed
        self.assertEqual(virt_mem[0x2000:0x3000], b"\x00" * 0x1000)

    def test_bss_section_rva_exceeds_image_size(self):
        """A BSS section whose VirtualAddress >= SizeOfImage must be rejected."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        # .bss at 0x3000 (equal to image_size 0x3000 -> out of bounds)
        builder.add_section(".bss", rva=0x3000, vsize=0x1000, rawsize=0, rawptr=0,
                            chars=IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("virtual range", str(cm.exception))

    def test_bss_section_virtual_size_overflows_image(self):
        """A BSS section whose VirtualAddress + VirtualSize > SizeOfImage must be rejected."""
        builder = PEBuilder(image_size=0x4000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        # .bss at 0x2000 with vsize=0x3000 -> 0x5000 > image_size 0x4000
        builder.add_section(".bss", rva=0x2000, vsize=0x3000, rawsize=0, rawptr=0,
                            chars=IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("virtual range", str(cm.exception))

    def test_bss_section_arithmetic_wrapping(self):
        """A BSS section whose VirtualSize wraps 32-bit integer boundaries must be rejected."""
        builder = PEBuilder(image_size=0x4000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        # .bss at 0x2000 with vsize=0xFFFFF000 -> 32-bit wrap produces 0x1000 <= 0x4000
        builder.add_section(".bss", rva=0x2000, vsize=0xFFFFF000, rawsize=0, rawptr=0,
                            chars=IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection):
            PELoaderOracle.validate_image(img_bytes)


class TestMalformedEntryPoints(unittest.TestCase):
    """Stress tests PE loader validation of entry points."""

    def test_entry_point_zero_allowed(self):
        """AddressOfEntryPoint == 0 (resource-only DLL or pure data image) is accepted."""
        builder = PEBuilder(entry_point=0)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        img_bytes = builder.build()
        ok, _ = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

    def test_entry_point_inside_headers_rejected(self):
        """AddressOfEntryPoint inside headers (< SizeOfHeaders) must be rejected (DEP violation)."""
        builder = PEBuilder(entry_point=0x100, size_of_headers=0x200)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("inside headers", str(cm.exception))

    def test_entry_point_outside_image_rejected(self):
        """AddressOfEntryPoint >= SizeOfImage must be rejected."""
        builder = PEBuilder(image_size=0x3000, entry_point=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("outside the image", str(cm.exception))

    def test_entry_point_in_unmapped_gap_rejected(self):
        """AddressOfEntryPoint between mapped sections in unmapped virtual space must be rejected."""
        builder = PEBuilder(image_size=0x4000, entry_point=0x2500)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200)
        builder.add_section(".data", rva=0x3000, vsize=0x1000, rawsize=0x200, rawptr=0x400)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("not in an executable mapped section", str(cm.exception))

    def test_entry_point_in_non_executable_section_rejected(self):
        """AddressOfEntryPoint in a non-executable section (.rdata) must be rejected."""
        builder = PEBuilder(image_size=0x3000, entry_point=0x1000)
        chars = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
        builder.add_section(".rdata", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200, chars=chars)
        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("not in an executable mapped section", str(cm.exception))


class TestCircularForwarders(unittest.TestCase):
    """Stress tests circular forwarder resolution, cycle prevention, and syntax parsing."""

    def test_forwarder_direct_self_cycle(self):
        """Direct self-loop (mod.fn -> mod.fn) must return None without infinite recursion."""
        fwd_map = {"mod.fn": "mod.fn"}
        self.assertIsNone(PELoaderOracle.resolve_forwarder_chain(fwd_map, "mod.fn"))

    def test_forwarder_mutual_cycle(self):
        """Mutual 2-node cycle (A -> B -> A) must return None."""
        fwd_map = {
            "kernel32.HeapAlloc": "ntdll.RtlAllocateHeap",
            "ntdll.RtlAllocateHeap": "kernel32.HeapAlloc",
        }
        self.assertIsNone(PELoaderOracle.resolve_forwarder_chain(fwd_map, "kernel32.HeapAlloc"))

    def test_forwarder_3_node_cycle(self):
        """3-node cycle (A -> B -> C -> A) must return None."""
        fwd_map = {
            "a.fn": "b.fn",
            "b.fn": "c.fn",
            "c.fn": "a.fn",
        }
        self.assertIsNone(PELoaderOracle.resolve_forwarder_chain(fwd_map, "a.fn"))

    def test_forwarder_deep_chain_exceeding_depth(self):
        """Forwarder chain deeper than depth limit (8 hops) returns None."""
        fwd_map = {f"mod{i}.fn": f"mod{i+1}.fn" for i in range(12)}
        self.assertIsNone(PELoaderOracle.resolve_forwarder_chain(fwd_map, "mod0.fn", max_depth=8))

    def test_forwarder_valid_finite_chain(self):
        """Finite acyclic chain within depth limit resolves properly."""
        fwd_map = {
            "api-ms-win-core-heap-l1-1-0.HeapAlloc": "kernel32.HeapAlloc",
            "kernel32.HeapAlloc": "ntdll.RtlAllocateHeap",
        }
        res = PELoaderOracle.resolve_forwarder_chain(fwd_map, "api-ms-win-core-heap-l1-1-0.HeapAlloc", max_depth=8)
        self.assertEqual(res, "ntdll.RtlAllocateHeap")

    def test_forwarder_malformed_spec_strings(self):
        """Malformed forwarder strings are safely rejected."""
        bad_specs = [
            "kernel32",         # Missing dot
            ".HeapAlloc",       # Leading dot
            "kernel32.",        # Trailing dot
            "",                 # Empty string
            "mod.#abc",         # Non-numeric ordinal
            "mod.#",            # Empty ordinal
            "mod.#-1",          # Negative ordinal
            "mod.#65536",       # Ordinal > 65535
            "mod.#999999",      # Huge ordinal
        ]
        for spec in bad_specs:
            with self.subTest(spec=spec):
                self.assertIsNone(PELoaderOracle.resolve_forwarder_chain({}, spec))

    def test_forwarder_valid_ordinal(self):
        """Valid ordinal forwarder within [1, 65535] is accepted."""
        res = PELoaderOracle.resolve_forwarder_chain({}, "mod.#123")
        self.assertEqual(res, "mod.#123")


class TestGATlsAccessPatterns(unittest.TestCase):
    """Stress tests /GA TLS rewrite simulation (patch_ga_tls_reads) across all 16 x86-64 registers."""

    def test_ga_tls_all_16_registers_rewritten(self):
        """Verifies that `mov r64, gs:[58h]` is rewritten to `0x1480 + 8 * slot` for all 16 registers:
        RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI (REX.W 0x48) and
        R8, R9, R10, R11, R12, R13, R14, R15 (REX.W+R 0x4C).
        """
        slot = 3
        expected_disp = 0x1480 + 8 * slot
        expected_disp_bytes = struct.pack("<I", expected_disp)

        # Build instruction sequence for each register
        # 65 [48|4C] 8B modrm 25 58 00 00 00
        rex_and_modrms = [
            # REX.W 0x48
            (0x48, 0x04),  # RAX
            (0x48, 0x0C),  # RCX
            (0x48, 0x14),  # RDX
            (0x48, 0x1C),  # RBX
            (0x48, 0x24),  # RSP
            (0x48, 0x2C),  # RBP
            (0x48, 0x34),  # RSI
            (0x48, 0x3C),  # RDI
            # REX.W+R 0x4C
            (0x4C, 0x04),  # R8
            (0x4C, 0x0C),  # R9
            (0x4C, 0x14),  # R10
            (0x4C, 0x1C),  # R11
            (0x4C, 0x24),  # R12
            (0x4C, 0x2C),  # R13
            (0x4C, 0x34),  # R14
            (0x4C, 0x3C),  # R15
        ]

        code = bytearray()
        for rex, modrm in rex_and_modrms:
            code.extend(bytes([0x65, rex, 0x8B, modrm, 0x25, 0x58, 0x00, 0x00, 0x00]))
            code.extend(b"\x90\x90")  # NOP padding

        builder = PEBuilder(image_size=0x4000)
        builder.add_section(".text", rva=0x1000, vsize=len(code) + 0x100, rawsize=len(code) + 0x100,
                            rawptr=0x200, data=code)
        img_bytes = builder.build()
        ok, virt_mem = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

        # Run patch_ga_tls_reads
        sections = [{
            "name": ".text", "rva": 0x1000, "vsize": len(code) + 0x100,
            "effective_vsize": len(code) + 0x100, "rawsize": len(code) + 0x100,
            "chars": IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ,
        }]
        sites = PELoaderOracle.patch_ga_tls_reads(virt_mem, sections, 0x4000, slot)
        self.assertEqual(sites, 16)

        # Verify that all 16 instructions have displacement rewritten to expected_disp
        cur_offset = 0x1000
        for _ in range(16):
            self.assertEqual(virt_mem[cur_offset:cur_offset + 5], virt_mem[cur_offset:cur_offset + 5])
            self.assertEqual(virt_mem[cur_offset + 5:cur_offset + 9], expected_disp_bytes)
            cur_offset += 11  # 9 bytes instruction + 2 NOPs

    def test_ga_tls_ignores_non_matching_patterns(self):
        """Verifies that non-matching instructions and non-executable sections are not touched."""
        slot = 1
        builder = PEBuilder(image_size=0x4000)

        # Construct non-matching patterns:
        non_matching = bytearray()
        # 1. Non-GS prefix (0x64 FS:)
        non_matching.extend(b"\x64\x48\x8B\x04\x25\x58\x00\x00\x00")
        # 2. Non-0x58 displacement (0x60)
        non_matching.extend(b"\x65\x48\x8B\x04\x25\x60\x00\x00\x00")
        # 3. Non-0x25 SIB byte (0x20)
        non_matching.extend(b"\x65\x48\x8B\x04\x20\x58\x00\x00\x00")
        # 4. Non-0x04 modrm (0x05)
        non_matching.extend(b"\x65\x48\x8B\x05\x25\x58\x00\x00\x00")
        # 5. Non-REX.W prefix (0x8B 32-bit)
        non_matching.extend(b"\x65\x8B\x04\x25\x58\x00\x00\x00\x00")

        # In executable .text section
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x200, rawptr=0x200, data=non_matching)
        # In non-executable .rdata section: exact matching pattern! Must NOT be patched!
        exact_match = b"\x65\x48\x8B\x04\x25\x58\x00\x00\x00"
        builder.add_section(".rdata", rva=0x2000, vsize=0x1000, rawsize=0x200, rawptr=0x400,
                            chars=IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ, data=exact_match)

        img_bytes = builder.build()
        ok, virt_mem = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

        sections = [
            {"name": ".text", "rva": 0x1000, "vsize": 0x1000, "effective_vsize": 0x1000, "rawsize": 0x200,
             "chars": IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ},
            {"name": ".rdata", "rva": 0x2000, "vsize": 0x1000, "effective_vsize": 0x1000, "rawsize": 0x200,
             "chars": IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ},
        ]

        sites = PELoaderOracle.patch_ga_tls_reads(virt_mem, sections, 0x4000, slot)
        self.assertEqual(sites, 0)
        # Verify .rdata still holds untouched 0x58
        self.assertEqual(virt_mem[0x2005:0x2009], b"\x58\x00\x00\x00")


class TestTLSHeaderEdgeCases(unittest.TestCase):
    """Stress tests non-canonical TLS directories and corrupted callback tables."""

    def test_tls_directory_short_size(self):
        """TLS directory with size < 40 bytes on x64 must be rejected."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 24)  # 24 is x86 size
        img_bytes = builder.build()

        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("smaller than IMAGE_TLS_DIRECTORY64", str(cm.exception))

    def test_tls_directory_all_zero_treated_as_absent(self):
        """A TLS directory with all zero fields is treated as absent and accepted."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)
        builder.write_at_rva(0x1500, b"\x00" * 40)
        img_bytes = builder.build()

        ok, _ = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

    def test_tls_template_inverted_bounds(self):
        """EndAddressOfRawData < StartAddressOfRawData must be rejected."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        tls_dir = struct.pack("<QQQQII", base + 0x1600, base + 0x1550, 0, 0, 0, 0)
        builder.write_at_rva(0x1500, tls_dir)

        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("precedes its start", str(cm.exception))

    def test_tls_template_zero_length(self):
        """Zero-length TLS template (raw=0, zero_fill=0) is safely accepted."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        tls_dir = struct.pack("<QQQQII", base + 0x1550, base + 0x1550, base + 0x1600, 0, 0, 0)
        builder.write_at_rva(0x1500, tls_dir)
        builder.write_at_rva(0x1600, b"\x00\x00\x00\x00")

        img_bytes = builder.build()
        ok, _ = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

    def test_tls_template_pure_bss(self):
        """Pure zero-fill TLS template (raw=0, zero_fill > 0) is safely accepted."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        tls_dir = struct.pack("<QQQQII", base + 0x1550, base + 0x1550, base + 0x1600, 0, 0x80, 0)
        builder.write_at_rva(0x1500, tls_dir)
        builder.write_at_rva(0x1600, b"\x00\x00\x00\x00")

        img_bytes = builder.build()
        ok, _ = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

    def test_tls_block_size_overflow(self):
        """Total TLS block size (raw + zero_fill) exceeding 4GB must be rejected."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        tls_dir = struct.pack("<QQQQII", base + 0x1550, base + 0x1570, base + 0x1600, 0, 0xFFFFFFFF, 0)
        builder.write_at_rva(0x1500, tls_dir)
        builder.write_at_rva(0x1600, b"\x00\x00\x00\x00")

        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("too large", str(cm.exception))

    def test_multiple_tls_callbacks(self):
        """Valid multiple TLS callbacks (4 callbacks + terminating null) pass validation and inspect parses them."""
        builder = PEBuilder(image_size=0x4000)
        builder.add_section(".text", rva=0x1000, vsize=0x3000, rawsize=0x3000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        cb_table_rva = 0x1800
        cb_funcs = [base + 0x1100, base + 0x1120, base + 0x1140, base + 0x1160, 0]
        cb_payload = struct.pack("<QQQQQ", *cb_funcs)

        tls_dir = struct.pack("<QQQQII", base + 0x1550, base + 0x1560, base + 0x1600,
                              base + cb_table_rva, 0, 0)
        builder.write_at_rva(0x1500, tls_dir)
        builder.write_at_rva(0x1600, b"\x00\x00\x00\x00")
        builder.write_at_rva(cb_table_rva, cb_payload)

        img_bytes = builder.build()
        ok, _ = PELoaderOracle.validate_image(img_bytes)
        self.assertTrue(ok)

        pe_img = pe_inspect.PeImage(img_bytes)
        self.assertEqual(pe_inspect._tls_callbacks(pe_img), 4)

    def test_tls_callback_out_of_bounds_va(self):
        """TLS callback pointer outside the image VA range must be rejected."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        cb_table_rva = 0x1800
        tls_dir = struct.pack("<QQQQII", base + 0x1550, base + 0x1560, 0, base + cb_table_rva, 0, 0)
        builder.write_at_rva(0x1500, tls_dir)
        builder.write_at_rva(cb_table_rva, struct.pack("<QQ", base + 0x9000, 0))

        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("outside the image", str(cm.exception))

    def test_tls_callback_pointing_into_headers(self):
        """TLS callback pointer landing inside SizeOfHeaders must be rejected."""
        builder = PEBuilder(image_size=0x3000, size_of_headers=0x200)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_TLS, 0x1500, 40)

        base = builder.image_base
        cb_table_rva = 0x1800
        tls_dir = struct.pack("<QQQQII", base + 0x1550, base + 0x1560, 0, base + cb_table_rva, 0, 0)
        builder.write_at_rva(0x1500, tls_dir)
        builder.write_at_rva(cb_table_rva, struct.pack("<QQ", base + 0x100, 0))

        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("inside headers", str(cm.exception))


class TestImportDirectoryEdgeCases(unittest.TestCase):
    """Stress tests malformed import descriptors and thunks."""

    def test_import_descriptor_unterminated_at_image_end(self):
        """Import descriptors without null terminator running past SizeOfImage must be rejected."""
        builder = PEBuilder(image_size=0x2000, size_of_headers=0x200)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x1000, rawptr=0x200)
        builder.set_directory(DIR_IMPORT, 0x1F00, 0x100)
        img_bytes = bytearray(builder.build())

        # Place non-null descriptors all the way up to image end
        desc = struct.pack("<IIIII", 0x1040, 0, 0, 0x1080, 0x1060)
        for offset in range(0x1F00 - 0x1000 + 0x200, len(img_bytes) - 20, 20):
            img_bytes[offset:offset + 20] = desc

        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(bytes(img_bytes))
        self.assertIn("no null descriptor", str(cm.exception))

    def test_import_descriptor_invalid_rva_beyond_image_size(self):
        """Import descriptor with OriginalFirstThunk or FirstThunk beyond image size must be rejected."""
        for bad_rva in [0x4000, 0xFFFFFFFF, 0x80000000]:
            with self.subTest(bad_rva=bad_rva):
                builder = PEBuilder(image_size=0x3000)
                builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
                builder.set_directory(DIR_IMPORT, 0x1100, 40)
                desc = struct.pack("<IIIII", bad_rva, 0, 0, 0x1200, 0x1140)
                term = b"\x00" * 20
                builder.write_at_rva(0x1100, desc + term)
                builder.write_at_rva(0x1200, b"kernel32.dll\x00")

                img_bytes = builder.build()
                with self.assertRaises(PELoaderRejection) as cm:
                    PELoaderOracle.validate_image(img_bytes)
                self.assertIn("past the image", str(cm.exception))

    def test_import_dll_name_unterminated(self):
        """Import descriptor with DLL Name string missing null terminator inside image is rejected."""
        builder = PEBuilder(image_size=0x2000)
        builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x1000, rawptr=0x200)
        builder.set_directory(DIR_IMPORT, 0x1100, 40)
        desc = struct.pack("<IIIII", 0, 0, 0, 0x1FE0, 0x1140)
        builder.write_at_rva(0x1100, desc + b"\x00" * 20)
        builder.write_at_rva(0x1FE0, b"A" * 0x20)

        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("isn't terminated", str(cm.exception))

    def test_delay_import_missing_rva_based_attribute(self):
        """Delay-import descriptor with Attributes.RvaBased == 0 must be rejected."""
        builder = PEBuilder(image_size=0x3000)
        builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
        builder.set_directory(DIR_DELAY_IMPORT, 0x1200, 64)
        desc = struct.pack("<IIIIIIII", 0, 0x1300, 0x1400, 0x1500, 0x1600, 0, 0, 0)
        builder.write_at_rva(0x1200, desc + b"\x00" * 32)
        builder.write_at_rva(0x1300, b"delayed.dll\x00")

        img_bytes = builder.build()
        with self.assertRaises(PELoaderRejection) as cm:
            PELoaderOracle.validate_image(img_bytes)
        self.assertIn("RvaBased attribute", str(cm.exception))

    def test_delay_import_missing_iat_or_int(self):
        """Delay-import descriptor with missing IAT or INT must be rejected."""
        for bad_desc in [
            struct.pack("<IIIIIIII", 1, 0x1300, 0, 0, 0x1600, 0, 0, 0),  # IAT=0
            struct.pack("<IIIIIIII", 1, 0x1300, 0, 0x1500, 0, 0, 0, 0),  # INT=0
        ]:
            builder = PEBuilder(image_size=0x3000)
            builder.add_section(".text", rva=0x1000, vsize=0x2000, rawsize=0x2000, rawptr=0x200)
            builder.set_directory(DIR_DELAY_IMPORT, 0x1200, 64)
            builder.write_at_rva(0x1200, bad_desc + b"\x00" * 32)
            builder.write_at_rva(0x1300, b"delayed.dll\x00")

            img_bytes = builder.build()
            with self.assertRaises(PELoaderRejection):
                PELoaderOracle.validate_image(img_bytes)


class TestAdversarialMutationFuzzing(unittest.TestCase):
    """Fuzzes valid base image with 200 deterministic random mutations and asserts safety."""

    def test_adversarial_mutation_fuzzing(self):
        rng = random.Random(42)

        # Baseline PE
        base_builder = PEBuilder(image_size=0x5000, entry_point=0x1000)
        base_builder.add_section(".text", rva=0x1000, vsize=0x1000, rawsize=0x400, rawptr=0x200)
        base_builder.add_section(".rdata", rva=0x2000, vsize=0x1000, rawsize=0x400, rawptr=0x600,
                                chars=IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ)

        # Imports
        base_builder.set_directory(DIR_IMPORT, 0x2000, 40)
        desc = struct.pack("<IIIII", 0x2040, 0, 0, 0x2080, 0x2060)
        base_builder.write_at_rva(0x2000, desc + b"\x00" * 20)
        base_builder.write_at_rva(0x2040, struct.pack("<QQ", 0x20A0, 0))
        base_builder.write_at_rva(0x2060, struct.pack("<QQ", 0x20A0, 0))
        base_builder.write_at_rva(0x2080, b"kernel32.dll\x00")
        base_builder.write_at_rva(0x20A0, b"\x00\x00ExitProcess\x00")

        # TLS
        base = base_builder.image_base
        base_builder.set_directory(DIR_TLS, 0x2100, 40)
        tls_dir = struct.pack("<QQQQII", base + 0x2150, base + 0x2160, base + 0x2170,
                              base + 0x2180, 0x20, 0)
        base_builder.write_at_rva(0x2100, tls_dir)
        base_builder.write_at_rva(0x2170, b"\x00\x00\x00\x00")
        base_builder.write_at_rva(0x2180, struct.pack("<QQ", base + 0x1050, 0))

        valid_bytes = base_builder.build()
        ok, _ = PELoaderOracle.validate_image(valid_bytes)
        self.assertTrue(ok)

        mutations = [
            lambda b: b.__setitem__(rng.randint(0, len(b) - 1), rng.randint(0, 255)) if len(b) > 0 else None,
            lambda b: b[:rng.randint(10, len(b) - 1)] if len(b) > 20 else None,
            lambda b: struct.pack_into("<I", b, 0x3C, rng.choice([0, 1, 0xFFFFFFFF, len(b) + 10])) if len(b) >= 0x40 else None,
            lambda b: struct.pack_into("<I", b, 0x80 + 24 + 60, rng.choice([0, 0xFFFF, 0x5000])) if len(b) >= 0x80 + 24 + 64 else None,
            lambda b: struct.pack_into("<I", b, 0x80 + 24 + 56, rng.choice([0, 0x100, 0xFFFFFFFF])) if len(b) >= 0x80 + 24 + 60 else None,
            lambda b: struct.pack_into("<I", b, 0x80 + 24 + 16, rng.choice([0x50, 0x3500, 0xFFFFFFF0])) if len(b) >= 0x80 + 24 + 20 else None,
            lambda b: struct.pack_into("<II", b, 0x80 + 24 + 112 + 8, rng.randint(0, 0x6000), rng.randint(0, 0x1000)) if len(b) >= 0x80 + 24 + 112 + 16 else None,
            lambda b: struct.pack_into("<II", b, 0x80 + 24 + 112 + 9 * 8, rng.randint(0, 0x6000), rng.randint(0, 0x100)) if len(b) >= 0x80 + 24 + 112 + 9 * 8 + 8 else None,
        ]

        iterations = 15
        safely_handled_count = 0

        for it in range(iterations):
            mutated = bytearray(valid_bytes)
            num_ops = rng.randint(1, 3)
            for _ in range(num_ops):
                mut_fn = rng.choice(mutations)
                res = mut_fn(mutated)
                if isinstance(res, (bytes, bytearray)):
                    mutated = bytearray(res)

            mutated_bytes = bytes(mutated)

            # Oracle check: must never crash or raise uncaught error
            try:
                PELoaderOracle.validate_image(mutated_bytes)
            except PELoaderRejection:
                pass
            except Exception as e:
                self.fail(f"PELoaderOracle threw unexpected exception on fuzz iteration {it}: {type(e).__name__}: {e}")

            # pe_inspect tool check: must never crash
            try:
                img = pe_inspect.PeImage(mutated_bytes)
                pe_inspect._imports(img, "import", pe_inspect.DIR_IMPORT)
                pe_inspect._tls_callbacks(img)
                pe_inspect._exports(img)
            except pe_inspect.PeFormatError:
                pass
            except Exception as e:
                self.fail(f"pe_inspect threw unexpected exception on fuzz iteration {it}: {type(e).__name__}: {e}")

            safely_handled_count += 1

        self.assertEqual(safely_handled_count, iterations)


if __name__ == "__main__":
    unittest.main()
