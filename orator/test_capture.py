"""Check single-channel capture using an isolated stereo input."""

from array import array
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import time
import unittest
import wave


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def event(worker, expected):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if select.select([worker.stdout], [], [], 0.2)[0]:
            line = worker.stdout.readline()
            if not line:
                raise AssertionError(worker.stderr.read().decode())
            message = json.loads(line)
            if message["event"] == "error":
                raise AssertionError(message)
            if message["event"] == expected:
                return
    raise AssertionError(f"Timed out waiting for {expected}")


class CaptureTests(unittest.TestCase):
    def test_primary_channel_is_not_mixed_with_secondary_channel(self):
        sink = f"orator_capture_test_{os.getpid()}"
        default_sink = command("pactl", "get-default-sink")
        module = command(
            "pactl", "load-module", "module-null-sink", f"sink_name={sink}"
        )
        worker = None
        try:
            command("pactl", "set-default-sink", default_sink)
            with tempfile.TemporaryDirectory(
                prefix="orator-capture-test-"
            ) as directory:
                directory = Path(directory)
                config = directory / "orator"
                config.mkdir()
                (config / "config.toml").write_text(
                    f'source = "{sink}.monitor"\nlanguage = "en"\n'
                )
                worker = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("backend.py"))],
                    env=dict(
                        os.environ,
                        XDG_CONFIG_HOME=str(directory),
                        TMPDIR=str(directory),
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                event(worker, "recording")
                recording = next(directory.glob("orator-*/speech.wav")).open("rb")
                with recording:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-re",
                            "-f",
                            "lavfi",
                            "-i",
                            "aevalsrc=0.15*sin(2*PI*440*t)|0.15*sin(2*PI*880*t):s=16000:d=2",
                            "-f",
                            "pulse",
                            "-device",
                            sink,
                            "-buffer_duration",
                            "100",
                            "orator-channel-test",
                        ],
                        check=True,
                        timeout=10,
                    )
                    time.sleep(0.3)
                    worker.stdin.write(b"stop\n")
                    worker.stdin.flush()
                    event(worker, "transcribing")
                    worker.stdin.close()
                    worker.wait(timeout=5)
                    with wave.open(recording, "rb") as audio:
                        self.assertEqual(
                            (audio.getnchannels(), audio.getframerate()), (1, 16000)
                        )
                        samples = array("h", audio.readframes(audio.getnframes()))
                powers = []
                for frequency in (440, 880):
                    real = sum(
                        v * math.cos(2 * math.pi * frequency * i / 16000)
                        for i, v in enumerate(samples)
                    )
                    imaginary = sum(
                        v * math.sin(2 * math.pi * frequency * i / 16000)
                        for i, v in enumerate(samples)
                    )
                    powers.append(math.hypot(real, imaginary) / len(samples))
                self.assertGreater(powers[0], 100)
                self.assertLess(powers[1], powers[0] * 0.01)
        finally:
            if worker is not None:
                if worker.poll() is None:
                    worker.stdin.close()
                    worker.wait(timeout=5)
                worker.stdout.close()
                worker.stderr.close()
            command("pactl", "unload-module", module)
            command("pactl", "set-default-sink", default_sink)


if __name__ == "__main__":
    unittest.main()
