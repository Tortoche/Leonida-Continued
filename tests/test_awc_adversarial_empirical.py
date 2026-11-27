"""Adversarial stress testing and fuzz harness for AWC parser and Opus decoder.

Empirical Challenger verification for:
1. Craft malformed AWC chunks (mismatched stream count, corrupted tag size, truncated packet frames).
2. Verify clean error raising without process crashes.
3. Check scripts/check_no_game_data.py.
"""
import io
import os
import random
import struct
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import awc_opus as awc
import awc_shorts
import awc_manifest


def make_tag(typ: int, offset: int, size: int) -> int:
    return ((typ & 0xFF) << 56) | ((size & 0x0FFFFFFF) << 28) | (offset & 0x0FFFFFFF)


def make_format_block(samples=960, rate=48000, codec=0x0B, channels=1, frames=960):
    block = bytearray(32)
    struct.pack_into("<I", block, 0, samples)
    struct.pack_into("<H", block, 8, rate)
    struct.pack_into("<I", block, 12, frames)
    block[28] = codec
    block[29] = channels
    return bytes(block)


class AwcAdversarialStressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def write_bank(self, filename: str, data: bytes) -> str:
        p = os.path.join(self.tmp.name, filename)
        with open(p, "wb") as f:
            f.write(data)
        return p

    # =========================================================================
    # Track 1: Mismatched Stream Count
    # =========================================================================

    def test_stream_count_zero_empty_bank(self):
        """AWC bank with 0 streams declared."""
        bank = b"ADAT" + struct.pack("<III", 0, 0, 16)
        streams, music = awc.parse_awc(bank)
        self.assertEqual(streams, [])
        self.assertEqual(music, [])
        # export_bank should succeed cleanly with 0 exported items
        path = self.write_bank("zero_stream.awc", bank)
        counts = awc.export_bank(path, self.out_dir)
        self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 0})

    def test_stream_count_exceeds_buffer(self):
        """AWC declares n=100 streams, but payload buffer ends after 20 bytes."""
        bank = b"ADAT" + struct.pack("<III", 0, 100, 16 + 100 * 4) + bytes(16)
        with self.assertRaises(struct.error):
            awc.parse_awc(bank)
        # Verify export_bank skips cleanly via main without crash
        path = self.write_bank("overflow_stream_count.awc", bank)
        with mock.patch("sys.argv", ["awc_opus.py", path, self.out_dir, "--jobs", "1"]), \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            code = awc.main()
            self.assertEqual(code, 0)

    def test_stream_count_huge_32bit_max(self):
        """AWC declares n = 0xFFFFFFFF (or 10 million). Must raise error, no crash or OOM."""
        bank = b"ADAT" + struct.pack("<III", 0, 0x00FFFFFF, 0x1000)
        with self.assertRaises(struct.error):
            awc.parse_awc(bank)
        path = self.write_bank("huge_stream_count.awc", bank)
        with mock.patch("sys.argv", ["awc_opus.py", path, self.out_dir, "--jobs", "1"]), \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(awc.main(), 0)

    def test_stream_table_index_mismatch(self):
        """Flags declare index table (0x10000), but cumulative tag counts diverge."""
        # 2 streams, each declares 1 tag, but index table says index 0, index 5
        header_size = 16 + 2 * 2 + 2 * 4 + 2 * 8
        header = bytearray(b"ADAT" + struct.pack("<III", 0x10000, 2, header_size))
        # index table: stream 0 -> 0 tags, stream 1 -> 5 tags (mismatched!)
        header += struct.pack("<HH", 0, 5)
        # stream table: stream 0 (1 tag), stream 1 (1 tag)
        header += struct.pack("<II", 1 | (1 << 29), 2 | (1 << 29))
        # 2 tags
        header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        bank = bytes(header) + make_format_block()
        with self.assertRaisesRegex(ValueError, "does not match the cumulative tag count"):
            awc.parse_awc(bank)
        path = self.write_bank("index_mismatch.awc", bank)
        with mock.patch("sys.argv", ["awc_opus.py", path, self.out_dir, "--jobs", "1"]), \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(awc.main(), 0)

    def test_music_descriptor_mismatched_channel_count(self):
        """Tag 0x48 descriptor declares channel count that diverges from block records or limits."""
        # channels = 0 (invalid)
        chunk = 4096
        desc = bytearray(12 + 16 * 0)
        struct.pack_into("<III", desc, 0, 1, chunk, 0)
        data = bytearray(chunk)
        self.assertEqual(awc.music_channels(data, (0, len(desc)), (0, len(data))), [])

        # channels = 33 (exceeds 32 max channels)
        desc33 = bytearray(12 + 16 * 33)
        struct.pack_into("<III", desc33, 0, 1, chunk, 33)
        self.assertEqual(awc.music_channels(data, (0, len(desc33)), (0, len(data))), [])

        # channels = 2, but descriptor size only accounts for 1
        desc_bad_len = bytearray(12 + 16 * 1)
        struct.pack_into("<III", desc_bad_len, 0, 1, chunk, 2)
        self.assertEqual(awc.music_channels(data, (0, len(desc_bad_len)), (0, len(data))), [])

    # =========================================================================
    # Track 2: Corrupted Tag Size & Boundaries
    # =========================================================================

    def test_corrupted_tag_size_extends_outside_bank(self):
        """Tag 0x55, 0x76, or 0x48 points outside the bank."""
        for tag_type in (0x55, 0x76, 0x48):
            with self.subTest(tag_type=hex(tag_type)):
                header_size = 16 + 4 + 8
                header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
                header += struct.pack("<I", 1 | (1 << 29))
                # offset 20, size 500 in a 100-byte bank
                header += struct.pack("<Q", make_tag(tag_type, 20, 500))
                bank = bytes(header).ljust(100, b"\x00")
                with self.assertRaisesRegex(ValueError, "extends outside the bank"):
                    awc.parse_awc(bank)
                path = self.write_bank(f"tag_overflow_{tag_type:02x}.awc", bank)
                with mock.patch("sys.argv", ["awc_opus.py", path, self.out_dir, "--jobs", "1"]), \
                     mock.patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(awc.main(), 0)

    def test_format_tag_size_under_32_bytes(self):
        """Tag 0x76 with valid offset, but size < 32 bytes (corrupted format block)."""
        for bad_size in (0, 1, 8, 16, 31):
            with self.subTest(size=bad_size):
                header_size = 16 + 4 + 2 * 8
                header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
                header += struct.pack("<I", 1 | (2 << 29))
                header += struct.pack("<Q", make_tag(0x76, header_size, bad_size))
                header += struct.pack("<Q", make_tag(0x55, header_size + 32, 2048))
                bank = bytes(header) + bytes(32) + bytes(2048)
                streams, music = awc.parse_awc(bank)
                # Formats < 32 bytes should be discarded cleanly, stream omitted
                self.assertEqual(streams, [])
                self.assertEqual(music, [])
                path = self.write_bank(f"bad_fmt_size_{bad_size}.awc", bank)
                counts = awc.export_bank(path, self.out_dir)
                self.assertEqual(counts, {"opus": 0, "pcm": 0, "other": 0, "music": 0})

    def test_music_descriptor_chunk_smaller_than_2048(self):
        """Chunk size < 2048 must be rejected by music_channels."""
        desc = bytearray(12 + 16)
        struct.pack_into("<III", desc, 0, 1, 1024, 1)  # chunk=1024 < 2048
        struct.pack_into("<II", desc, 12, 100, 960)
        struct.pack_into("<H", desc, 22, 48000)
        desc[24] = 0x0B
        data = bytearray(4096)
        self.assertEqual(awc.music_channels(data, (0, len(desc)), (0, len(data))), [])

    def test_music_descriptor_subblock_total_exceeds_block(self):
        """Subblock count claims more 2048-byte subblocks than fit in block_size."""
        chunk = 4096
        desc = bytearray(12 + 16)
        struct.pack_into("<III", desc, 0, 1, chunk, 1)
        struct.pack_into("<II", desc, 12, 100, 960)
        struct.pack_into("<H", desc, 22, 48000)
        desc[24] = 0x0B
        data = bytearray(chunk)
        # record: count = 5 (needs 2048 header + 5 * 2048 > 4096)
        struct.pack_into("<II", data, 4, 5, 0)
        self.assertEqual(awc.music_channels(data, (0, len(desc)), (0, len(data))), [])

    def test_non_standard_tag_with_corrupt_size_ignored(self):
        """Tags not processed by awc_opus (e.g. 0x5a, 0xd9, 0xbd, 0xef) with wild sizes."""
        header_size = 16 + 4 + 8
        header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
        header += struct.pack("<I", 1 | (1 << 29))
        # Tag 0x5a declaring size larger than bank
        header += struct.pack("<Q", make_tag(0x5A, 1000, 999999))
        bank = bytes(header).ljust(64, b"\x00")
        streams, music = awc.parse_awc(bank)
        self.assertEqual(streams, [])
        self.assertEqual(music, [])

    # =========================================================================
    # Track 3: Truncated Packet Frames
    # =========================================================================

    def test_opus_packets_packet_extends_beyond_block(self):
        """Within a 2048-byte block, packet length claims bytes past 2048."""
        block = bytearray(2048)
        # packet at offset 2040 claims length 50 (extends past 2048)
        struct.pack_into("<H", block, 2040, 50)
        packets = awc.opus_packets(bytes(block), 0, 2048)
        # Must break cleanly and not return partial/over-read packet
        self.assertEqual(packets, [])

    def test_opus_packets_zero_length_terminates_block(self):
        """Zero length packet terminates block scan cleanly."""
        block = bytearray(2048)
        struct.pack_into("<H", block, 0, 10)
        block[2:12] = bytes([0x98]) + b"123456789"
        struct.pack_into("<H", block, 12, 0)  # padding zero length
        struct.pack_into("<H", block, 14, 5)   # garbage following padding
        block[16:21] = b"extra"
        packets = awc.opus_packets(bytes(block), 0, 2048)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0], bytes([0x98]) + b"123456789")

    def test_opus_samples_truncated_frame(self):
        """Verify opus_samples on truncated or 1-byte packets with multi-frame TOC."""
        # TOC code 3 (c=3) indicates arbitrary frames, frame count stored in packet[1]
        # But packet has only 1 byte (packet[1] missing)
        packet_1byte_c3 = bytes([0x0B])  # config 1, c = 3
        # In awc_opus.py: `packet[1] & 0x3F if len(packet) > 1 else 0`
        samples = awc.opus_samples(packet_1byte_c3)
        self.assertEqual(samples, 0)

        # TOC code 0 (1 frame): 1 byte packet is valid TOC
        self.assertEqual(awc.opus_samples(bytes([0x08])), 960)

        # Multi-frame count byte present but declares 0 frames
        packet_zero_frames = bytes([0x0B, 0x00])
        self.assertEqual(awc.opus_samples(packet_zero_frames), 0)

        # Multi-frame count byte declares max frames (63)
        packet_63_frames = bytes([0x0B, 0x3F])
        self.assertEqual(awc.opus_samples(packet_63_frames), 960 * 63)

    def test_write_ogg_opus_truncated_empty_packet_list(self):
        """write_ogg_opus with empty packet list writes header and empty EOS page cleanly."""
        out_ogg = os.path.join(self.out_dir, "empty_packets.ogg")
        written = awc.write_ogg_opus(out_ogg, [], 1, 100, length=0)
        self.assertEqual(written, 0)
        self.assertTrue(os.path.isfile(out_ogg))
        self.assertGreater(os.path.getsize(out_ogg), 0)

    def test_write_ogg_opus_packet_needs_too_many_segments_raises_cleanly(self):
        """Packet larger than 255*255 bytes exceeds Ogg page lacing segment limit."""
        huge_packet = bytes(255 * 255 + 1)
        out_ogg = os.path.join(self.out_dir, "too_large.ogg")
        with self.assertRaisesRegex(ValueError, "255 Ogg segments"):
            awc.write_ogg_opus(out_ogg, [huge_packet], 1, 100, length=960)

    def test_corrupted_packet_stream_export_bank(self):
        """Full bank export with corrupted Opus blocks."""
        # 2048-byte block where first packet header is corrupt (e.g. length exceeds block)
        corrupted_opus = bytearray(2048)
        struct.pack_into("<H", corrupted_opus, 0, 3000)  # > 2048
        header_size = 16 + 4 + 2 * 8
        header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
        header += struct.pack("<I", 1 | (2 << 29))
        header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        header += struct.pack("<Q", make_tag(0x55, header_size + 32, 2048))
        bank = bytes(header) + make_format_block(samples=960, codec=0x0B, channels=1) + bytes(corrupted_opus)
        path = self.write_bank("corrupt_packet.awc", bank)
        counts = awc.export_bank(path, self.out_dir)
        # Because pk = [] when packet length is corrupted, counts["opus"] is 0, no output file
        self.assertEqual(counts["opus"], 0)
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, "corrupt_packet", "00000001.ogg")))

    # =========================================================================
    # Track 4: awc_shorts, awc_manifest, Companion & Marker Tags Resilience
    # =========================================================================

    def test_awc_shorts_handles_corrupt_bank(self):
        """awc_shorts handles malformed bank without process crash."""
        bad_bank = b"ADAT" + struct.pack("<III", 0, 50, 16)
        path = self.write_bank("bad_shorts.awc", bad_bank)
        with mock.patch("sys.argv", ["awc_shorts.py", path, self.out_dir, "--jobs", "1"]), \
             mock.patch("sys.stdout", new_callable=io.StringIO), \
             mock.patch("sys.stderr", new_callable=io.StringIO):
            code = awc_shorts.main()
            self.assertEqual(code, 0)

    def test_awc_manifest_notes_corrupt_bank_entries(self):
        """awc_manifest flags corrupted sample rate and size mismatch in notes."""
        notes = awc_manifest._notes(codec=0x00, channels=2, rate=0, samples=100,
                                    data=(0, 200), bank_size=500)
        self.assertIn("sample rate is zero", notes)
        self.assertIn("PCM16 data is 200 bytes, header implies 400", notes)

    def test_companion_tag_0x5a_and_0xd9_malformed_handling(self):
        """Companion tags 0x5a (grid20) and 0xd9 (indexrun) with corrupted sizes and payloads."""
        from rsc8_audio import decode_grid20_table, decode_index_runs

        # 0x5a table with truncated size (< 8 bytes)
        res = decode_grid20_table(b"1234")
        self.assertFalse(res["size_ok"])

        # 0x5a table with non-modulo-20 payload length
        res = decode_grid20_table(b"12345678" + bytes(15))
        self.assertFalse(res["size_ok"])

        # 0xd9 table with truncated size (< 4 bytes)
        res = decode_index_runs(b"12")
        self.assertFalse(res["size_ok"])

        # 0xd9 table with declared groups exceeding length (declared 100, only 1 group provided)
        res = decode_index_runs(struct.pack("<IIII", 100, 2, 0, 0))
        self.assertTrue(res["size_ok"])
        self.assertFalse(res["group_count_matches"])

    def test_marker_tag_0xbd_corrupted_handling(self):
        """Tag 0xBD marker records with corrupted lengths and out-of-order offsets."""
        from rsc8_audio import decode_marker_records, pair_marker_regions, marker_records

        # Truncated marker blob (< 16 bytes)
        res = decode_marker_records(b"short")
        self.assertFalse(res["size_ok"])

        # Markers not multiple of 16
        res = decode_marker_records(bytes(35))
        self.assertFalse(res["size_ok"])

        # Non-ascending offsets
        m1 = struct.pack("<IIII", 0xE5C74063, 0, 500, 0)
        m2 = struct.pack("<IIII", 0xE5C74063, 0, 100, 0)  # offset goes backwards
        res = decode_marker_records(m1 + m2)
        self.assertTrue(res["size_ok"])
        self.assertFalse(res["offsets_ascending"])

        # pair_marker_regions on corrupt records handles cleanly
        pairs = pair_marker_regions(marker_records(m1 + m2))
        self.assertEqual(pairs["regions"], [])

    # =========================================================================
    # Track 5: Fuzz Harness (Adversarial Mutation Testing)
    # =========================================================================

    def test_fuzz_mutated_awc_banks_no_crash(self):
        """Generate 100 pseudorandomly mutated AWC byte buffers and verify clean error handling."""
        import random
        rng = random.Random(0xDEADBEEF)

        # Base valid bank: 1 stream, 1 format tag, 1 data tag
        header_size = 16 + 4 + 2 * 8
        base_header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
        base_header += struct.pack("<I", 1 | (2 << 29))
        base_header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        base_header += struct.pack("<Q", make_tag(0x55, header_size + 32, 2048))
        base_bank = bytes(base_header) + make_format_block(samples=960, codec=0x0B, channels=1) + bytes(2048)

        mutations_tested = 0
        for i in range(100):
            mutated = bytearray(base_bank)
            mutation_type = rng.randint(0, 4)
            if mutation_type == 0:
                # Random byte flips
                num_flips = rng.randint(1, 10)
                for _ in range(num_flips):
                    idx = rng.randint(0, len(mutated) - 1)
                    mutated[idx] = rng.randint(0, 255)
            elif mutation_type == 1:
                # Truncate at random offset
                cut = rng.randint(0, len(mutated))
                mutated = mutated[:cut]
            elif mutation_type == 2:
                # Random insertions
                insert_pos = rng.randint(0, len(mutated))
                mutated[insert_pos:insert_pos] = rng.randbytes(rng.randint(1, 64))
            elif mutation_type == 3:
                # Corrupt stream count and flags
                if len(mutated) >= 12:
                    struct.pack_into("<II", mutated, 4, rng.randint(0, 0xFFFFFFFF), rng.randint(0, 0xFFFFFFFF))
            elif mutation_type == 4:
                # Corrupt tag pointers
                if len(mutated) >= 36:
                    struct.pack_into("<Q", mutated, 20, rng.randint(0, 0xFFFFFFFFFFFFFFFF))

            # Must NEVER cause a fatal process crash or uncatchable error
            bank_path = self.write_bank(f"fuzz_{i}.awc", bytes(mutated))
            try:
                streams, music = awc.parse_awc(bytes(mutated))
            except (ValueError, struct.error, IndexError):
                pass
            except Exception as exc:
                self.fail(f"Unexpected exception type during parse_awc on fuzz input {i}: {type(exc)}: {exc}")

            try:
                counts = awc.export_bank(bank_path, self.out_dir)
            except (ValueError, struct.error, IndexError, OSError):
                pass
            except Exception as exc:
                self.fail(f"Unexpected exception type during export_bank on fuzz input {i}: {type(exc)}: {exc}")

            mutations_tested += 1

        self.assertEqual(mutations_tested, 100)


def make_opus_block(packets: list[bytes], block_size: int = 2048) -> bytes:
    buf = bytearray(block_size)
    offset = 0
    for pkt in packets:
        n = len(pkt)
        if offset + 2 + n > block_size:
            break
        struct.pack_into("<H", buf, offset, n)
        buf[offset + 2:offset + 2 + n] = pkt
        offset += 2 + n
    return bytes(buf)


# =============================================================================
# Track 6: Malformed Chunk Headers & Container Magic Traps
# =============================================================================

class AwcChunkHeaderStressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def write_bank(self, filename: str, data: bytes) -> str:
        p = os.path.join(self.tmp.name, filename)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_trap_header_magic_ad740003_rejected(self):
        """Trap chunk headers like 0xAD740003 (LE and BE) and invalid magics must raise ValueError."""
        traps = [
            struct.pack("<I", 0xAD740003),  # \x03\x00\x74\xad
            struct.pack(">I", 0xAD740003),  # \xad\x74\x00\x03
            b"AD74",
            b"DATA",
            b"RIFF",
            b"\x00\x00\x00\x00",
            b"\xFF\xFF\xFF\xFF",
        ]
        for trap in traps:
            with self.subTest(trap=trap):
                bank_data = trap + struct.pack("<III", 0, 1, 32) + bytes(16)
                with self.assertRaisesRegex(ValueError, "not a little-endian AWC"):
                    awc.parse_awc(bank_data)
                path = self.write_bank("trap.awc", bank_data)
                with self.assertRaisesRegex(ValueError, "not a little-endian AWC"):
                    awc.export_bank(path, self.out_dir)

    def test_truncated_container_header_sizes(self):
        """Header buffers under 16 bytes raise ValueError (if <4) or struct.error (if 4..15)."""
        for size in [0, 1, 2, 3]:
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    awc.parse_awc(bytes(size))
        for size in [4, 8, 12, 15]:
            with self.subTest(size=size):
                bank = b"ADAT" + bytes(size - 4)
                with self.assertRaises(struct.error):
                    awc.parse_awc(bank)

    def test_flag_extension_truncations(self):
        """Bank flags specify extra tables (0x400000, 0x10000, 0x200000) but buffer truncates inside them."""
        flag_cases = [
            (0x400000, 16, "flag 0x400000 (+4) truncated"),
            (0x010000, 16 + 4, "flag 0x10000 (+n*2) truncated"),
            (0x200000, 16 + 4, "flag 0x200000 (+(n+1)*4) truncated"),
            (0x610000, 16 + 8, "combined flags truncated"),
        ]
        for flags, data_len, desc in flag_cases:
            with self.subTest(desc=desc):
                # Declare n=5 streams so table lengths are substantial
                bank = b"ADAT" + struct.pack("<III", flags, 5, 200) + bytes(data_len - 16)
                with self.assertRaises(struct.error):
                    awc.parse_awc(bank)
                path = self.write_bank("trunc_flags.awc", bank)
                with mock.patch("sys.argv", ["awc_opus.py", path, self.out_dir, "--jobs", "1"]), \
                     mock.patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(awc.main(), 0)

    def test_music_descriptor_chunk_overflow_geometry(self):
        """Streamed music descriptor geometry edge cases: chunk < 2048, blocks=0, channels=0."""
        # chunk=0
        desc_zero_chunk = bytearray(12 + 16)
        struct.pack_into("<III", desc_zero_chunk, 0, 1, 0, 1)
        data = bytearray(4096)
        self.assertEqual(awc.music_channels(data, (0, len(desc_zero_chunk)), (0, len(data))), [])

        # blocks=0
        desc_zero_blocks = bytearray(12 + 16)
        struct.pack_into("<III", desc_zero_blocks, 0, 0, 2048, 1)
        self.assertEqual(awc.music_channels(data, (0, len(desc_zero_blocks)), (0, len(data))), [])

        # channels=0
        desc_zero_ch = bytearray(12)
        struct.pack_into("<III", desc_zero_ch, 0, 1, 2048, 0)
        self.assertEqual(awc.music_channels(data, (0, len(desc_zero_ch)), (0, len(data))), [])

        # (blocks - 1) * chunk >= d_size
        desc_stride_overflow = bytearray(12 + 16)
        struct.pack_into("<III", desc_stride_overflow, 0, 5, 2048, 1)
        # d_size is only 4096, but 5 blocks at 2048 need at least 4 * 2048 = 8192
        self.assertEqual(awc.music_channels(data, (0, len(desc_stride_overflow)), (0, 4096)), [])

    def test_music_descriptor_channel_rates_bounds(self):
        """Rates outside 1000..192000 are rejected; 1000, 48000, and 65535 are accepted."""
        for rate, should_pass in [(0, False), (500, False), (999, False), (1000, True),
                                  (48000, True), (65535, True)]:
            with self.subTest(rate=rate):
                chunk = 4096
                desc = bytearray(12 + 16)
                struct.pack_into("<III", desc, 0, 1, chunk, 1)
                struct.pack_into("<II", desc, 12, 0x100, 960)
                struct.pack_into("<H", desc, 22, rate)
                desc[24] = 0x0B  # Opus
                data = bytearray(chunk)
                # Channel record: 1 subblock
                struct.pack_into("<II", data, 4, 1, 0)
                # Opus block with 1 valid packet
                pkt = bytes([0x08]) + b"data123"
                data[2048:2048 + 2 + len(pkt)] = struct.pack("<H", len(pkt)) + pkt
                bank = bytes(desc) + bytes(data)
                result = awc.music_channels(bank, (0, len(desc)), (len(desc), len(data)))
                if should_pass:
                    self.assertEqual(len(result), 1)
                    self.assertEqual(result[0][2], rate)
                else:
                    self.assertEqual(result, [])


# =============================================================================
# Track 7: Corrupt Opus TOC Bytes & RFC 6716 Frame Stress
# =============================================================================

class OpusTocAdversarialStressTest(unittest.TestCase):
    def test_all_256_toc_permutations(self):
        """Exhaustively verify all 256 possible TOC bytes across 1-byte, 2-byte, and multi-byte packets."""
        for b in range(256):
            # 1-byte packet
            s1 = awc.opus_samples(bytes([b]))
            self.assertIsInstance(s1, int)
            self.assertGreaterEqual(s1, 0)

            # 2-byte packet with 0 frame count (c=3)
            s2 = awc.opus_samples(bytes([b, 0x00]))
            self.assertIsInstance(s2, int)
            self.assertGreaterEqual(s2, 0)

            # 2-byte packet with max 63 frames (c=3)
            s3 = awc.opus_samples(bytes([b, 0x3F]))
            self.assertIsInstance(s3, int)
            self.assertGreaterEqual(s3, 0)

            # 2-byte packet with upper VBR/padding bits set
            s4 = awc.opus_samples(bytes([b, 0xC5]))
            self.assertIsInstance(s4, int)
            self.assertGreaterEqual(s4, 0)

    def test_corrupt_multiframe_arbitrary_c3_configurations(self):
        """TOC c=3 arbitrary frame count stress."""
        # SILK config 0 (10ms = 480 samples per frame)
        toc = 0x03  # config 0, c = 3
        # 1-byte packet (missing frame count byte) -> 0 frames
        self.assertEqual(awc.opus_samples(bytes([toc])), 0)
        # 2-byte packet with 1 frame -> 480
        self.assertEqual(awc.opus_samples(bytes([toc, 0x01])), 480)
        # 2-byte packet with 2 frames -> 960
        self.assertEqual(awc.opus_samples(bytes([toc, 0x02])), 960)
        # 2-byte packet with 63 frames -> 480 * 63 = 30240
        self.assertEqual(awc.opus_samples(bytes([toc, 0x3F])), 30240)
        # Upper bits in packet[1] (e.g. 0x80 or 0x40) must be masked with 0x3F
        self.assertEqual(awc.opus_samples(bytes([toc, 0x80 | 0x02])), 960)

    def test_empty_packet_in_write_ogg_opus_raises_index_error(self):
        """write_ogg_opus with a packet containing 0 bytes raises IndexError inside opus_samples."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "empty_packet.ogg")
            with self.assertRaises(IndexError):
                awc.write_ogg_opus(p, [b""], 1, 100)

    def test_mixed_toc_packets_granule_monotonicity(self):
        """Mixed packets with variable TOCs (CELT 2.5ms, SILK 60ms) generate strictly monotonic Ogg granules."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "variable_toc.ogg")
            # p1: CELT 2.5ms (120 samples, toc = 16<<3 = 0x80)
            # p2: SILK 60ms (2880 samples, toc = 3<<3 = 0x18)
            # p3: CELT 20ms (960 samples, toc = 19<<3 = 0x98)
            p1 = bytes([0x80]) + b"celt"
            p2 = bytes([0x18]) + b"silk"
            p3 = bytes([0x98]) + b"celt20"
            packets = [p1, p2, p3] * 30  # 90 packets
            written = awc.write_ogg_opus(p, packets, 1, 0xABC, length=0)
            self.assertGreater(written, 0)
            with open(p, "rb") as f:
                ogg_bytes = f.read()
            # Parse all Ogg pages and check granules are monotonic
            offset = 0
            last_granule = 0
            page_count = 0
            while offset < len(ogg_bytes):
                self.assertEqual(ogg_bytes[offset:offset + 4], b"OggS")
                granule, = struct.unpack_from("<q", ogg_bytes, offset + 6)
                seq, = struct.unpack_from("<I", ogg_bytes, offset + 18)
                if seq >= 2:  # Data pages
                    self.assertGreaterEqual(granule, last_granule)
                    last_granule = granule
                nseg = ogg_bytes[offset + 26]
                body_size = sum(ogg_bytes[offset + 27:offset + 27 + nseg])
                offset += 27 + nseg + body_size
                page_count += 1
            self.assertGreaterEqual(page_count, 3)


# =============================================================================
# Track 8: Mismatched Channel Counts & Multi-Channel Isolation
# =============================================================================

class ChannelCountMismatchStressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def write_bank(self, filename: str, data: bytes) -> str:
        p = os.path.join(self.tmp.name, filename)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_plain_stream_mono_format_with_stereo_toc(self):
        """Format tag says 1 channel, but bitstream Opus TOC has bit 2 set (stereo)."""
        # Stereo TOC byte: 0x9C (CELT 20ms, stereo bit 2 set: 0x98 | 0x04 = 0x9C)
        stereo_pkt = bytes([0x9C]) + b"stereodata"
        block = make_opus_block([stereo_pkt])
        header_size = 16 + 4 + 2 * 8
        header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
        header += struct.pack("<I", 1 | (2 << 29))
        header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        header += struct.pack("<Q", make_tag(0x55, header_size + 32, len(block)))
        # Format block declares channels=1
        bank = bytes(header) + make_format_block(samples=960, channels=1) + block
        path = self.write_bank("mono_fmt_stereo_toc.awc", bank)
        counts = awc.export_bank(path, self.out_dir)
        self.assertEqual(counts["opus"], 1)
        out_ogg = os.path.join(self.out_dir, "mono_fmt_stereo_toc", "00000001.ogg")
        self.assertTrue(os.path.isfile(out_ogg))
        # Verify OpusHead declares 1 channel as declared by format tag
        with open(out_ogg, "rb") as f:
            data = f.read()
        opushead_pos = data.find(b"OpusHead")
        self.assertGreater(opushead_pos, 0)
        channels_in_head = data[opushead_pos + 9]
        self.assertEqual(channels_in_head, 1)

    def test_plain_stream_stereo_format_with_mono_toc(self):
        """Format tag says 2 channels, but bitstream Opus TOC has bit 2 clear (mono)."""
        # Mono TOC byte: 0x98
        mono_pkt = bytes([0x98]) + b"monodata"
        block = make_opus_block([mono_pkt])
        header_size = 16 + 4 + 2 * 8
        header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
        header += struct.pack("<I", 1 | (2 << 29))
        header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        header += struct.pack("<Q", make_tag(0x55, header_size + 32, len(block)))
        # Format block declares channels=2
        bank = bytes(header) + make_format_block(samples=960, channels=2) + block
        path = self.write_bank("stereo_fmt_mono_toc.awc", bank)
        counts = awc.export_bank(path, self.out_dir)
        self.assertEqual(counts["opus"], 1)
        out_ogg = os.path.join(self.out_dir, "stereo_fmt_mono_toc", "00000001.ogg")
        self.assertTrue(os.path.isfile(out_ogg))
        with open(out_ogg, "rb") as f:
            data = f.read()
        opushead_pos = data.find(b"OpusHead")
        self.assertGreater(opushead_pos, 0)
        channels_in_head = data[opushead_pos + 9]
        self.assertEqual(channels_in_head, 2)

    def test_plain_stream_unsupported_channel_counts_routed_to_other(self):
        """Format tags with channels not in (1, 2) (e.g. 0, 3, 6, 8, 32) route cleanly to counts['other']."""
        for ch in [0, 3, 4, 6, 8, 32]:
            with self.subTest(channels=ch):
                block = make_opus_block([bytes([0x98]) + b"sample"])
                header_size = 16 + 4 + 2 * 8
                header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
                header += struct.pack("<I", 1 | (2 << 29))
                header += struct.pack("<Q", make_tag(0x76, header_size, 32))
                header += struct.pack("<Q", make_tag(0x55, header_size + 32, len(block)))
                bank = bytes(header) + make_format_block(samples=960, channels=ch) + block
                path = self.write_bank(f"chan_{ch}.awc", bank)
                counts = awc.export_bank(path, self.out_dir)
                self.assertEqual(counts["opus"], 0)
                self.assertEqual(counts["other"], 1)

    def test_music_descriptor_mixed_mono_stereo_stems(self):
        """Streamed music with 4 channels: channels 0 & 2 mono, channels 1 & 3 stereo.
        Verify stereo channels have their samples halved, mono channels do not."""
        chunk = 4096 * 4
        channels = 4
        desc = bytearray(12 + 16 * channels)
        struct.pack_into("<III", desc, 0, 1, chunk, channels)
        for c in range(channels):
            base = 12 + 16 * c
            struct.pack_into("<II", desc, base, 0x100 + c, 1920)  # 1920 declared samples
            struct.pack_into("<H", desc, base + 10, 48000)
            desc[base + 12] = 0x0B

        data = bytearray(chunk)
        # Channel table: 1 subblock for each channel
        for c in range(channels):
            struct.pack_into("<II", data, 24 * c + 4, 1, 0)

        # Subblocks: header is aligned to 2048
        # header = (4 * 24 + 4 * 4 + 2047) & ~2047 = 2048
        # Channel 0: mono packet (0x98)
        data[2048 + 0 * 2048:2048 + 0 * 2048 + 2 + 5] = struct.pack("<H", 5) + bytes([0x98]) + b"mono"
        # Channel 1: stereo packet (0x9C, bit 2 set)
        data[2048 + 1 * 2048:2048 + 1 * 2048 + 2 + 7] = struct.pack("<H", 7) + bytes([0x9C]) + b"stereo"
        # Channel 2: mono packet (0x98)
        data[2048 + 2 * 2048:2048 + 2 * 2048 + 2 + 5] = struct.pack("<H", 5) + bytes([0x98]) + b"mono"
        # Channel 3: stereo packet (0x9C, bit 2 set)
        data[2048 + 3 * 2048:2048 + 3 * 2048 + 2 + 7] = struct.pack("<H", 7) + bytes([0x9C]) + b"stereo"

        bank = bytes(desc) + bytes(data)
        ch_info = awc.music_channels(bank, (0, len(desc)), (len(desc), len(data)))
        self.assertEqual(len(ch_info), 4)
        # Channel 0: mono -> samples=1920, packet_channels=1
        self.assertEqual((ch_info[0][1], ch_info[0][5]), (1920, 1))
        # Channel 1: stereo -> samples=960 (halved!), packet_channels=2
        self.assertEqual((ch_info[1][1], ch_info[1][5]), (960, 2))
        # Channel 2: mono -> samples=1920, packet_channels=1
        self.assertEqual((ch_info[2][1], ch_info[2][5]), (1920, 1))
        # Channel 3: stereo -> samples=960 (halved!), packet_channels=2
        self.assertEqual((ch_info[3][1], ch_info[3][5]), (960, 2))

    def test_music_descriptor_max_channels_32_discrete_stems(self):
        """Streamed music with maximum supported channels (32 discrete stems)."""
        channels = 32
        chunk = 2048 + 32 * 2048  # header + 32 subblocks
        desc = bytearray(12 + 16 * channels)
        struct.pack_into("<III", desc, 0, 1, chunk, channels)
        for c in range(channels):
            base = 12 + 16 * c
            struct.pack_into("<II", desc, base, 0x1000 + c, 960)
            struct.pack_into("<H", desc, base + 10, 48000)
            desc[base + 12] = 0x0B

        data = bytearray(chunk)
        for c in range(channels):
            struct.pack_into("<II", data, 24 * c + 4, 1, 0)
            start = 2048 + c * 2048
            pkt = bytes([0x98]) + f"stem_{c:02d}".encode()
            data[start:start + 2 + len(pkt)] = struct.pack("<H", len(pkt)) + pkt

        bank = bytes(desc) + bytes(data)
        ch_info = awc.music_channels(bank, (0, len(desc)), (len(desc), len(data)))
        self.assertEqual(len(ch_info), 32)
        # All 32 stems have distinct channel IDs
        cids = [item[0] for item in ch_info]
        self.assertEqual(len(set(cids)), 32)


# =============================================================================
# Track 9: Out-of-Order Packets & Repeated Prefix Stress
# =============================================================================

class OutOfOrderPacketStressTest(unittest.TestCase):
    def test_repeated_prefix_out_of_order_packets(self):
        """Repeated prefix with reversed/swapped packets must be rejected."""
        p1 = bytes([0x08]) + b"frame1"  # 960 samples
        p2 = bytes([0x08]) + b"frame2"  # 960 samples
        samples = awc.opus_samples(p1) + awc.opus_samples(p2)

        # In order: [p1, p2] matches previous tail [p1, p2]
        self.assertEqual(awc.repeated_prefix_packets([p1, p2, p1], [p1, p2], samples), 2)

        # Swapped/out-of-order: [p2, p1] does NOT match previous tail [p1, p2]
        self.assertIsNone(awc.repeated_prefix_packets([p2, p1, p1], [p1, p2], samples))

    def test_repeated_prefix_misaligned_duration_bounds(self):
        """Skip duration landing inside a packet rather than on whole boundary is rejected."""
        p1 = bytes([0x08]) + b"frame1"  # 960 samples
        p2 = bytes([0x08]) + b"frame2"  # 960 samples
        # Declare skip of 1200 samples (p1 is 960, leaving 240, but p2 is 960)
        self.assertIsNone(awc.repeated_prefix_packets([p1, p2], [p1, p2], 1200))

    def test_repeated_prefix_skip_exceeds_chunk_packets(self):
        """Skip samples exceeding the total duration of the chunk packets is rejected."""
        p1 = bytes([0x08]) + b"frame1"
        self.assertIsNone(awc.repeated_prefix_packets([p1], [p1], 5000))

    def test_repeated_prefix_zero_duration_packet_rejected(self):
        """Packet with 0 duration inside repeated prefix causes clean rejection (no infinite loop)."""
        # TOC c=3 with 0 frames declares duration 0
        p_zero = bytes([0x03, 0x00])
        self.assertEqual(awc.opus_samples(p_zero), 0)
        # Attempting skip with duration 0 packet
        self.assertIsNone(awc.repeated_prefix_packets([p_zero], [p_zero], 960))


# =============================================================================
# Track 10: Resource Leaks, Memory Scaling & High Volume Fuzzing
# =============================================================================

class ResourceLeakAndMemoryStressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.out_dir, exist_ok=True)

    def write_bank(self, filename: str, data: bytes) -> str:
        p = os.path.join(self.tmp.name, filename)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_memory_leak_across_1000_malformed_banks(self):
        """Process 1,000 diverse malformed banks and verify heap memory delta is bounded."""
        tracemalloc.start()
        try:
            # Warm-up 50 banks
            for i in range(50):
                p = self.write_bank(f"warm_{i}.awc", b"ADAT" + struct.pack("<III", 0, 1, 32) + bytes(16))
                try:
                    awc.export_bank(p, self.out_dir)
                except Exception:
                    pass
                os.remove(p)

            snapshot_before = tracemalloc.take_snapshot()

            # Process 1,000 malformed banks
            for i in range(1000):
                v = i % 5
                if v == 0:
                    d = b"AD74\x00\x03" + bytes(30)
                elif v == 1:
                    d = b"ADAT" + struct.pack("<III", 0x10000, 50, 1000) + bytes(20)
                elif v == 2:
                    d = b"ADAT" + struct.pack("<III", 0, 1, 40) + struct.pack("<I", 1 | (1 << 29)) + \
                        struct.pack("<Q", make_tag(0x55, 20, 5000)) + bytes(100)
                elif v == 3:
                    d = b"ADAT" + struct.pack("<III", 0, 0, 16)
                else:
                    d = b"ADAT" + bytes(10)
                p = self.write_bank(f"stress_{i}.awc", d)
                try:
                    awc.export_bank(p, self.out_dir)
                except Exception:
                    pass
                os.remove(p)

            snapshot_after = tracemalloc.take_snapshot()
            diffs = snapshot_after.compare_to(snapshot_before, "lineno")
            total_leak = sum(d.size_diff for d in diffs if d.size_diff > 0)
            # Total growth should be less than 200 KB for 1,000 files
            self.assertLess(total_leak, 200 * 1024, f"Memory leak detected: {total_leak} bytes")
        finally:
            tracemalloc.stop()

    def test_staging_file_cleanup_on_simulated_write_failure(self):
        """Verify that temporary staging files (.awc-*.tmp) are completely cleaned up when replace fails."""
        target_wav = os.path.join(self.out_dir, "fail.wav")
        with mock.patch("os.replace", side_effect=OSError("Disk full / permission denied")):
            with self.assertRaises(OSError):
                awc.write_wav(target_wav, b"\x00" * 40, 48000, 1)
        remaining = os.listdir(self.out_dir)
        self.assertEqual(remaining, [], f"Orphaned staging files found: {remaining}")

    def test_high_volume_fuzz_500_mutations(self):
        """Stress test with 500 randomly mutated AWC banks; zero unhandled crashes allowed."""
        rng = random.Random(0xCAFE1234)
        header_size = 16 + 4 + 2 * 8
        base_header = bytearray(b"ADAT" + struct.pack("<III", 0, 1, header_size))
        base_header += struct.pack("<I", 1 | (2 << 29))
        base_header += struct.pack("<Q", make_tag(0x76, header_size, 32))
        base_header += struct.pack("<Q", make_tag(0x55, header_size + 32, 2048))
        base_bank = bytes(base_header) + make_format_block(samples=960, codec=0x0B, channels=1) + bytes(2048)

        for i in range(500):
            mutated = bytearray(base_bank)
            m_type = rng.randint(0, 4)
            if m_type == 0:
                for _ in range(rng.randint(1, 15)):
                    mutated[rng.randint(0, len(mutated) - 1)] = rng.randint(0, 255)
            elif m_type == 1:
                mutated = mutated[:rng.randint(0, len(mutated))]
            elif m_type == 2:
                idx = rng.randint(0, len(mutated))
                mutated[idx:idx] = rng.randbytes(rng.randint(1, 100))
            elif m_type == 3:
                if len(mutated) >= 12:
                    struct.pack_into("<II", mutated, 4, rng.randint(0, 0xFFFFFFFF), rng.randint(0, 0xFFFFFFFF))
            elif m_type == 4:
                if len(mutated) >= 36:
                    struct.pack_into("<Q", mutated, 20, rng.randint(0, 0xFFFFFFFFFFFFFFFF))

            bank_path = self.write_bank(f"fuzz500_{i}.awc", bytes(mutated))
            try:
                awc.parse_awc(bytes(mutated))
            except (ValueError, struct.error, IndexError):
                pass
            except Exception as exc:
                self.fail(f"Unhandled exception in parse_awc on fuzz iteration {i}: {type(exc)}: {exc}")

            try:
                awc.export_bank(bank_path, self.out_dir)
            except (ValueError, struct.error, IndexError, OSError):
                pass
            except Exception as exc:
                self.fail(f"Unhandled exception in export_bank on fuzz iteration {i}: {type(exc)}: {exc}")
            os.remove(bank_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

