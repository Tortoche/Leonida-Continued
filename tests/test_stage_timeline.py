"""Unit tests for tools/stage_timeline.py.

    python -I tests/test_stage_timeline.py
"""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import stage_timeline  # noqa: E402


SAMPLE_LOG = "\n".join([
    "[rt] peek t=1s: 0x7446510=0 0x98f828c=0 0x7453ed0=?",
    "[rt] peek t=2s: 0x7446510=2 0x98f828c=1",
    "[rt] peek t=3s: 0x7446510=3 0x98f828c=100000001 0x7453ed0=0",
    "[rt] gameinput: scripted press of button 1 at 3.5 s",
    "[rt] peek t=4s: 0x7446510=3 0x98f828c=3",
    "[rt] peek t=5s: 0x7446510=3 0x98f828c=0 0x7453ed0=2",
    "[rt] peek t=6s: 0x7446510=3 0x98f828c=0 0x7453ed0=0",
    "[rt] gameinput: scripted press of button 2 at 7.0 s",
]) + "\n"


class StageTimelineTest(unittest.TestCase):
    def test_first_value_parses_hex_and_handles_missing_or_unknown(self):
        self.assertIsNone(stage_timeline.first_value(None))
        self.assertIsNone(stage_timeline.first_value("?"))
        self.assertIsNone(stage_timeline.first_value("?,10"))
        self.assertEqual(stage_timeline.first_value("1a,ff"), 0x1A)

    def test_read_and_timeline_tracks_stages_loading_end_and_presses(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "boot.log")
            with open(log_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(SAMPLE_LOG)
            peeks, presses = stage_timeline.read(log_path)

        self.assertEqual(len(peeks), 6)
        self.assertEqual(len(presses), 2)
        events = stage_timeline.timeline(peeks, presses)
        self.assertEqual(events, [
            {"t": 1, "stage": "not started"},
            {"t": 2, "stage": "init"},
            {"t": 3, "stage": "boot screens"},
            {"t": 4, "stage": "landing"},
            {"t": 5, "stage": "story load"},
            {"t": 6, "stage": "story load (loading screen ended)"},
        ])
        self.assertEqual(presses[0]["next_change"], {"t": 4, "stage": "landing"})
        self.assertIsNone(presses[1]["next_change"])

    def test_timeline_handles_unknown_states(self):
        peeks = [
            {"t": 0, "game": None, "session": None, "load": None},
            {"t": 1, "game": 9, "session": None, "load": None},
            {"t": 2, "game": 3, "session": 0x42, "load": None},
        ]
        events = stage_timeline.timeline(peeks, [])
        self.assertEqual(events, [
            {"t": 0, "stage": "no title state yet"},
            {"t": 1, "stage": "game state 0x9"},
            {"t": 2, "stage": "session 0x42"},
        ])

    def test_main_json_and_missing_peeks_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "boot.log")
            with open(log_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(SAMPLE_LOG)

            stdout = io.StringIO()
            with patch.object(sys, "argv", ["stage_timeline.py", log_path, "--json"]):
                with patch("sys.stdout", stdout):
                    rc = stage_timeline.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(len(payload["stages"]), 6)
            self.assertEqual(len(payload["presses"]), 2)

            empty_path = os.path.join(tmp, "empty.log")
            with open(empty_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("no peek lines here\n")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["stage_timeline.py", empty_path]):
                with patch("sys.stderr", stderr):
                    rc_empty = stage_timeline.main()
            self.assertEqual(rc_empty, 1)
            self.assertIn("no peek lines", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
