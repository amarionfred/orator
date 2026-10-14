"""Exercise bounded audio processing and complete transcript delivery offline."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from orator import backend


class BackendTests(unittest.TestCase):
    def test_chunks_preserve_samples_and_split_at_a_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.pcm"
            samples = np.full(16000 * 602 + 137, 2000, dtype="<i2")
            samples[16000 * 298 : 16000 * 299] = 0
            samples[-137:] = np.arange(137, dtype="<i2")
            samples.tofile(path)
            chunks = list(backend.audio_chunks(path))
        self.assertTrue(16000 * 298 < len(chunks[0]) < 16000 * 299)
        self.assertTrue(all(len(chunk) <= 16000 * 300 for chunk in chunks))
        np.testing.assert_array_equal(
            np.concatenate(chunks), samples.astype(np.float32) / 32768.0
        )

    def test_hour_of_audio_is_read_in_bounded_pieces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.pcm"
            with path.open("wb") as audio:
                audio.truncate(16000 * 2 * 3600)
            total = 0
            for chunk in backend.audio_chunks(path):
                self.assertLessEqual(len(chunk), 16000 * 300)
                total += len(chunk)
        self.assertEqual(total, 16000 * 3600)

    def test_transcription_joins_chunks_and_preserves_long_unicode_text(self):
        expected_parts = ["日本語 café " * 1800, "last words 🗣"]
        responses = [
            [SimpleNamespace(text=text, avg_logprob=-0.2, compression_ratio=1.0)]
            for text in expected_parts
        ]
        expected = " ".join(" ".join(text.split()) for text in expected_parts)
        connection = Mock()
        model = Mock()
        model.transcribe.side_effect = [(iter(parts), None) for parts in responses]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "speech.pcm"
            np.full(16000 * 301, 1000, dtype="<i2").tofile(path)
            connection.recv.return_value = str(path)
            config = {
                "model": "unused",
                "cpu_threads": 1,
                "language": "auto",
                "transcripts": directory / "saved",
            }
            with patch.dict(
                "sys.modules",
                {
                    "faster_whisper": SimpleNamespace(
                        WhisperModel=Mock(return_value=model)
                    )
                },
            ):
                backend.recognize(connection, config)
            connection.send.assert_called_once_with(
                {"event": "result", "transcript": str(path.with_suffix(".txt"))}
            )
            result = backend.transcript_result(
                path.with_suffix(".txt"), config["transcripts"]
            )
            (saved,) = config["transcripts"].glob("*.txt")
            self.assertEqual(saved.read_text(encoding="utf-8"), expected)
            self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                result, f"Full transcript (auto-deletes after 24h): {saved}"
            )
            connection.close.assert_called_once_with()
            self.assertEqual(model.transcribe.call_count, 2)

    def test_short_transcript_is_returned_without_creating_a_saved_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.txt"
            saved = Path(directory) / "saved"
            text = "日本語 café 🗣"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(backend.transcript_result(path, saved), text)
            self.assertFalse(saved.exists())

    def test_long_digital_silence_keeps_all_speech_samples(self):
        speech = np.array([1, -1, 0, 1000, -1000], dtype="<i2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.pcm"
            with path.open("wb") as audio:
                audio.seek(16000 * 2 * 185)
                audio.write(speech.tobytes())
                audio.write(bytes(16000 * 2))
            connection = Mock()
            connection.recv.return_value = str(path)
            model = Mock()
            model.transcribe.return_value = (iter([]), None)
            with patch.dict(
                "sys.modules",
                {
                    "faster_whisper": SimpleNamespace(
                        WhisperModel=Mock(return_value=model)
                    )
                },
            ):
                backend.recognize(
                    connection, {"model": "unused", "cpu_threads": 1, "language": "en"}
                )
            model.transcribe.assert_called_once()
            np.testing.assert_array_equal(
                model.transcribe.call_args.args[0], speech.astype(np.float32) / 32768.0
            )
            connection.send.assert_called_once_with(
                {"event": "result", "transcript": str(path.with_suffix(".txt"))}
            )

    def test_failed_save_removes_incomplete_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.txt"
            saved = Path(directory) / "saved"
            path.write_text("spoken words " * 2000, encoding="utf-8")
            with patch.object(
                backend.shutil, "copyfileobj", side_effect=OSError("disk full")
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    backend.transcript_result(path, saved)
            self.assertEqual(list(saved.iterdir()), [])

    def test_silence_unclear_speech_and_distortion_report_distinct_results(self):
        unclear = SimpleNamespace(
            text="unreliable", avg_logprob=-2, compression_ratio=1
        )
        for level, segments, expected in [
            (0, [], {"event": "result"}),
            (
                1000,
                [unclear],
                {
                    "event": "error",
                    "message": "Couldn't hear speech clearly. Speak closer to the microphone and retry.",
                },
            ),
            (
                32767,
                [],
                {
                    "event": "error",
                    "message": "Microphone audio is distorted. Lower your microphone input volume and retry.",
                },
            ),
        ]:
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "speech.pcm"
                np.full(16000, level, dtype="<i2").tofile(path)
                connection = Mock()
                connection.recv.return_value = str(path)
                model = Mock()
                model.transcribe.return_value = (iter(segments), None)
                config = {
                    "model": "unused",
                    "cpu_threads": 1,
                    "language": "en",
                    "transcripts": Path(directory) / "saved",
                }
                with patch.dict(
                    "sys.modules",
                    {
                        "faster_whisper": SimpleNamespace(
                            WhisperModel=Mock(return_value=model)
                        )
                    },
                ):
                    backend.recognize(connection, config)
                if expected["event"] == "result":
                    model.transcribe.assert_not_called()
                    expected = dict(expected, transcript=str(path.with_suffix(".txt")))
                    self.assertEqual(path.with_suffix(".txt").read_text(), "")
                connection.send.assert_called_once_with(expected)
                connection.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
