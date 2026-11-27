"""Unit tests for tools/dstorage_census.py.

    python -I tests/test_dstorage_census.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import dstorage_census  # noqa: E402


class DStorageCensusTest(unittest.TestCase):
    def test_census_groups_shapes_statuses_staging_and_rates(self):
        # options bitfield: bits 1..4 hold the BC codec (6 -> BC6H, 7 -> BC7)
        lines = [
            "[rt] ds-staging peak_bytes=4096 idle_bytes=1024",
            "[rt] ds-request file=alpha.bin offset=0 tick_ms=1000 options=0x0 dest=buf zlib=0 sw=0 src=128 dst=128 hr=0x0 cancelled=0 priority=normal paired=1 read_us=10 copy_us=5",
            "[rt] ds-request file=beta.bin offset=64 tick_ms=2000 options=0xc dest=pages zlib=1 sw=2 src=256 dst=1048576 hr=0x0 cancelled=0 priority=high read_us=20 inflate_us=30 bc_us=40 sw_us=50 copy_us=60",
            "[rt] ds-request file=gamma.bin offset=0 tick_ms=2000 options=0x0 dest=buf zlib=0 sw=0 src=64 dst=64 hr=0x0 cancelled=1 priority=normal",
            "[rt] ds-request file=delta.bin offset=0 tick_ms=2000 options=0x0 dest=buf zlib=0 sw=0 src=32 dst=32 hr=0x80070057 cancelled=0 reason=bad_arg",
            "[rt] ds-notify us=15",
            "[rt] ds-staging peak_bytes=8192 idle_bytes=0 ",
        ]
        res = dstorage_census.census(lines)

        self.assertEqual(res["shapes"]["buffer plain"], {
            "requests": 3,
            "source_bytes": 224,
            "destination_bytes": 224,
            "successful": 1,
            "written_bytes": 128,
            "cancelled": 1,
            "rejected": 1,
        })
        self.assertEqual(res["shapes"]["paged zlib BC6H swizzle2"], {
            "requests": 1,
            "source_bytes": 256,
            "destination_bytes": 1048576,
            "successful": 1,
            "written_bytes": 1048576,
        })
        self.assertEqual(res["failures"], {"0x80070057 bad_arg": 1})
        self.assertEqual(res["prepared_in_parallel"], 1)
        self.assertEqual(res["queue_priorities"], {"normal": 2, "high": 1})
        self.assertEqual(res["worker_microseconds"], {
            "read_us": 30,
            "inflate_us": 30,
            "bc_us": 40,
            "sw_us": 50,
            "copy_us": 65,
            "notify_us": 15,
        })
        self.assertEqual(res["retirement_window_ms"], 1000)
        self.assertEqual(res["successful_requests_per_second"], 2.0)
        self.assertEqual(res["written_MiB_per_second"], 1.0)
        self.assertEqual(res["process_peak_staging_reservation_bytes"], 8192)
        self.assertEqual(res["process_drained_through_peak_bytes"], 8192)

    def test_from_file_filters_case_insensitively_and_handles_empty(self):
        lines = [
            "[rt] ds-request file=early.bin offset=0 tick_ms=100 options=0x0 dest=buf zlib=0 sw=0 src=16 dst=16 hr=0x0 cancelled=0",
            "[rt] ds-notify us=99",
            "[rt] ds-request file=TargetPack.rpf offset=0 tick_ms=200 options=0xe dest=buf zlib=0 sw=0 src=32 dst=64 hr=0x0 cancelled=0",
            "[rt] ds-request file=after.bin offset=0 tick_ms=200 options=0x0 dest=buf zlib=0 sw=0 src=8 dst=8 hr=0x0 cancelled=0",
        ]
        res = dstorage_census.census(lines, from_file="targetpack")
        self.assertEqual(res["totals"]["requests"], 2)
        self.assertIn("buffer BC7", res["shapes"])
        self.assertNotIn("notify_us", res["worker_microseconds"])
        self.assertEqual(res["retirement_window_ms"], 0)
        self.assertNotIn("successful_requests_per_second", res)

        empty = dstorage_census.census([])
        self.assertEqual(empty["totals"], {})
        self.assertIsNone(empty["retirement_window_ms"])


if __name__ == "__main__":
    unittest.main()
