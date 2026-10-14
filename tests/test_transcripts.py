"""Exercise transcript expiry without a microphone, speech model, or user files."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from orator import backend


class TranscriptCleanupTests(unittest.TestCase):
    def test_expiry_removes_only_old_managed_regular_files(self):
        now = 1_000_000
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "transcripts"
            directory.mkdir()
            for name, age in {
                "dictation-old.txt": 25 * 3600,
                "dictation-boundary.txt": 24 * 3600,
                "dictation-recent.txt": 24 * 3600 - 1,
                "notes.txt": 48 * 3600,
            }.items():
                path = directory / name
                path.write_text(name)
                os.utime(path, (now - age, now - age))
            outside = root / "original.txt"
            outside.write_text("keep this")
            os.utime(outside, (0, 0))
            (directory / "dictation-link.txt").symlink_to(outside)
            (directory / "dictation-broken.txt").symlink_to(root / "missing")
            (directory / "dictation-folder.txt").mkdir()
            removed = backend.cleanup_transcripts(directory, 24, now=now)
            self.assertEqual(
                (
                    removed,
                    sorted(path.name for path in directory.iterdir()),
                    outside.read_text(),
                ),
                (
                    2,
                    [
                        "dictation-broken.txt",
                        "dictation-folder.txt",
                        "dictation-link.txt",
                        "dictation-recent.txt",
                        "notes.txt",
                    ],
                    "keep this",
                ),
            )
            self.assertEqual(backend.cleanup_transcripts(directory, 24, now=now), 0)

    def test_saved_transcript_expires_from_save_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "speech.txt"
            text = "dictated words " * 2000
            source.write_text(text)
            os.utime(source, (0, 0))
            directory = Path(temporary) / "transcripts"
            result = backend.transcript_result(source, directory, retention_hours=2)
            (saved,) = directory.iterdir()
            saved_at = saved.stat().st_mtime
            self.assertEqual(
                (result, saved.read_text()),
                (f"Full transcript (auto-deletes after 2h): {saved}", text),
            )
            self.assertEqual(
                backend.cleanup_transcripts(directory, 2, now=saved_at + 7199), 0
            )
            self.assertEqual(
                backend.cleanup_transcripts(directory, 2, now=saved_at + 7200), 1
            )
            self.assertEqual(list(directory.iterdir()), [])

    def test_cleanup_command_uses_retention_setting_without_a_speech_model(self):
        for setting, deleted in [
            ("", 1),
            ("transcript_retention_hours = 1", 2),
            ("transcript_retention_hours = 2", 2),
        ]:
            with (
                self.subTest(setting=setting),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                config = root / "config/orator"
                config.mkdir(parents=True)
                (config / "config.toml").write_text(setting)
                directory = root / "data/orator/transcripts"
                directory.mkdir(parents=True)
                for name, hours in [
                    ("yesterday", 25),
                    ("earlier", 3),
                    ("recent", 0.25),
                ]:
                    path = directory / f"dictation-{name}.txt"
                    path.write_text(name)
                    timestamp = time.time() - hours * 3600
                    os.utime(path, (timestamp, timestamp))
                result = subprocess.run(
                    [sys.executable, backend.__file__, "--cleanup-transcripts"],
                    env=dict(
                        os.environ,
                        XDG_CONFIG_HOME=str(root / "config"),
                        XDG_DATA_HOME=str(root / "data"),
                        XDG_CACHE_HOME=str(root / "no-model"),
                    ),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(
                    json.loads(result.stdout), {"event": "cleanup", "deleted": deleted}
                )
                remaining = ["dictation-recent.txt"]
                if deleted == 1:
                    remaining.insert(0, "dictation-earlier.txt")
                self.assertEqual(
                    sorted(path.name for path in directory.iterdir()), remaining
                )

    def test_invalid_retention_fails_without_deleting_transcripts(self):
        for value in ("0", "-1", "true", "1.5"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = root / "config/orator"
                config.mkdir(parents=True)
                (config / "config.toml").write_text(
                    f"transcript_retention_hours = {value}\n"
                )
                directory = root / "data/orator/transcripts"
                directory.mkdir(parents=True)
                transcript = directory / "dictation-old.txt"
                transcript.write_text("keep this")
                os.utime(transcript, (0, 0))
                result = subprocess.run(
                    [sys.executable, backend.__file__, "--cleanup-transcripts"],
                    env=dict(
                        os.environ,
                        XDG_CONFIG_HOME=str(root / "config"),
                        XDG_DATA_HOME=str(root / "data"),
                    ),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    (
                        result.returncode,
                        json.loads(result.stdout),
                        transcript.read_text(),
                    ),
                    (
                        1,
                        {
                            "event": "error",
                            "message": "transcript_retention_hours must be a positive whole number.",
                        },
                        "keep this",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
