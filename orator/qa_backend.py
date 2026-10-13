"""Check capture, transcription, and cancellation with a virtual audio input."""

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def wait_event(worker, event, timeout=45):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        ready, _, _ = select.select([worker.stdout], [], [], 0.2)
        if ready:
            line = worker.stdout.readline()
            assert line, f"Worker exited early: {worker.poll()}"
            message = json.loads(line)
            assert message["event"] != "error", message
            if message["event"] == event:
                return message
    raise AssertionError(f"Timed out waiting for {event}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Speech recording")
    parser.add_argument("--expect", required=True, help="Expected transcript fragment")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    expected = " ".join(args.expect.lower().split())
    if not expected:
        parser.error("--expect must contain words")
    sink = f"orator_qa_{os.getpid()}"
    default = command("pactl", "get-default-sink")
    module = command("pactl", "load-module", "module-null-sink", f"sink_name={sink}")
    worker = None
    try:
        command("pactl", "set-default-sink", default)
        with tempfile.TemporaryDirectory(prefix="orator-qa-") as directory:
            config = Path(directory) / "orator"
            config.mkdir()
            (config / "config.toml").write_text(
                f'source = "{sink}.monitor"\nlanguage = "en"\n'
            )
            env = dict(os.environ, XDG_CONFIG_HOME=directory)
            backend = Path.home() / ".local/share/orator/bin/orator-backend"
            for mode in (
                "speech",
                "silence",
                "cancel",
                "cancel_transcription",
                "terminal_closed",
            ):
                worker = subprocess.Popen(
                    [backend],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                wait_event(worker, "recording")
                if mode in ("speech", "cancel_transcription"):
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-re",
                            "-i",
                            str(args.audio.resolve()),
                            "-f",
                            "pulse",
                            "-device",
                            sink,
                            "-buffer_duration",
                            "100",
                            "orator-qa",
                        ],
                        check=True,
                    )
                    time.sleep(0.3)  # Let virtual playback reach the monitor source.
                else:
                    time.sleep(1)
                if mode in ("cancel", "terminal_closed"):
                    if mode == "terminal_closed":
                        worker.terminate()
                    else:
                        worker.stdin.close()
                    worker.wait(timeout=5)
                    assert b'"result"' not in worker.stdout.read()
                else:
                    worker.stdin.write(b"stop\n")
                    worker.stdin.flush()
                    wait_event(worker, "transcribing")
                    if mode == "cancel_transcription":
                        worker.stdin.close()
                        worker.wait(timeout=5)
                    else:
                        result = wait_event(worker, "result")
                        if mode == "speech":
                            assert expected in " ".join(
                                result["text"].lower().split()
                            ), result
                        else:
                            assert result["text"] == "", result
                        worker.wait(timeout=5)
                assert worker.returncode == 0, worker.stderr.read().decode()
                print(f"PASS: {mode}", flush=True)
                worker = None
    finally:
        if worker is not None and worker.poll() is None:
            worker.stdin.close()
            worker.wait(timeout=5)
        command("pactl", "unload-module", module)
        command("pactl", "set-default-sink", default)


if __name__ == "__main__":
    main()
