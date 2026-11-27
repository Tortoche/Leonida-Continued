import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "scripts"))
import check_compute_journal_overlay as overlay  # noqa: E402


class TestCheckComputeJournalOverlay(unittest.TestCase):
    def test_require(self):
        overlay.require(True, "should not raise")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            overlay.require(False, "boom")

    def test_entries_missing_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(overlay.entries(Path(tmp)), {})

    def test_entries_valid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            journal = cache / "pipeline_cache.d"
            journal.mkdir()
            payload = b"x" * 64
            entry = journal / "cs-0123456789abcdef.bin"
            entry.write_bytes(payload)
            result = overlay.entries(cache)
            self.assertEqual(list(result.keys()), ["cs-0123456789abcdef.bin"])
            digest, mtime_ns = result["cs-0123456789abcdef.bin"]
            self.assertEqual(digest, hashlib.sha256(payload).digest())
            self.assertIsInstance(mtime_ns, int)

    def test_entries_rejects_bad_name_or_small_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            journal = cache / "pipeline_cache.d"
            journal.mkdir()
            bad_name = journal / "other.bin"
            bad_name.write_bytes(b"x" * 64)
            with self.assertRaises(RuntimeError):
                overlay.entries(cache)
            bad_name.unlink()

            too_small = journal / "cs-0123456789abcdef.bin"
            too_small.write_bytes(b"x" * 32)
            with self.assertRaises(RuntimeError):
                overlay.entries(cache)


if __name__ == "__main__":
    unittest.main()
