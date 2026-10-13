"""Local transcription worker: `stop` on stdin, JSON events on stdout, EOF to cancel."""

from array import array
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unicodedata
import wave


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def settings():
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = config / "orator/config.toml"
    data = tomllib.loads(path.read_text()) if path.exists() else {}
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    model = data.get("model", "base")
    data["model"] = str(
        Path(model) if Path(model).is_dir() else cache / "orator/models" / model
    )
    data.setdefault("source", "default")
    data.setdefault("language", "auto")
    data.setdefault("max_seconds", 180)
    data.setdefault("cpu_threads", 4)
    if not 1 <= data["max_seconds"] <= 1800 or not 1 <= data["cpu_threads"] <= 64:
        raise ValueError("Invalid max_seconds or cpu_threads in Orator's config.")
    if not (Path(data["model"]) / "model.bin").is_file():
        raise ValueError("Local speech model missing. Run the Orator installer.")
    return data


def recognize(connection, config):
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            config["model"],
            device="cpu",
            compute_type="int8",
            cpu_threads=config["cpu_threads"],
            local_files_only=True,
        )
        path = connection.recv()
        with wave.open(path, "rb") as audio:
            samples = array("h", audio.readframes(audio.getnframes()))
        if samples and sum(abs(s) >= 32440 for s in samples) / len(samples) > 0.02:
            raise ValueError(
                "Microphone audio is distorted. Lower your microphone input volume and retry."
            )
        segments, _ = model.transcribe(
            path,
            language=None if config["language"] == "auto" else config["language"],
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = list(segments)
        confident = [
            segment
            for segment in segments
            if segment.avg_logprob >= -1.0 and segment.compression_ratio <= 2.4
        ]
        if segments and not confident:
            raise ValueError(
                "Couldn't hear speech clearly. Speak closer to the microphone and retry."
            )
        text = " ".join(segment.text for segment in confident)
        text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
        connection.send(dict(event="result", text=" ".join(text.split())[:16000]))
    except Exception as error:
        connection.send(dict(event="error", message=str(error)))
    finally:
        connection.close()


def controls(messages):
    for line in sys.stdin:
        messages.put(line.strip())
    messages.put("cancel")


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def main():
    capture = worker = None
    try:
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        config = settings()
        commands = queue.Queue()
        threading.Thread(target=controls, args=(commands,), daemon=True).start()
        with tempfile.TemporaryDirectory(prefix="orator-") as directory:
            path = Path(directory) / "speech.wav"
            with (Path(directory) / "capture.log").open("w+b") as errors:
                try:
                    capture = subprocess.Popen(
                        [
                            "parecord",
                            f"--device={config['source']}",
                            "--client-name=Orator",
                            "--stream-name=Orator microphone",
                            "--rate=16000",
                            "--format=s16le",
                            "--channels=1",
                            "--channel-map=front-left",
                            "--no-remix",
                            "--no-remap",
                            "--latency-msec=100",
                            "--file-format=wav",
                            str(path),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=errors,
                    )
                    context = mp.get_context("spawn")
                    connection, child_connection = context.Pipe()
                    worker = context.Process(
                        target=recognize, args=(child_connection, config), daemon=True
                    )
                    worker.start()
                    child_connection.close()
                    started = time.monotonic()
                    recording_announced = False
                    transcribing = False
                    while True:
                        try:
                            control = commands.get(timeout=0.05)
                        except queue.Empty:
                            control = None
                        if control == "cancel":
                            return
                        if connection.poll():
                            result = connection.recv()
                            emit(**result)
                            return
                        if not worker.is_alive():
                            raise RuntimeError(
                                "Speech worker stopped unexpectedly. Try /record again."
                            )
                        elapsed = time.monotonic() - started
                        if not transcribing and (
                            control == "stop"
                            or capture.poll() is not None
                            or elapsed >= config["max_seconds"]
                        ):
                            if capture.poll() is None:
                                capture.send_signal(signal.SIGINT)
                                capture.wait(timeout=3)
                            if capture.returncode not in (0, 255):
                                errors.seek(0)
                                detail = (
                                    errors.read()
                                    .decode(errors="replace")[-400:]
                                    .strip()
                                )
                                raise RuntimeError("Microphone unavailable: " + detail)
                            if not path.exists() or path.stat().st_size < 6400:
                                raise RuntimeError(
                                    "Recording too short. Try /record and speak for a moment."
                                )
                            emit("transcribing")
                            connection.send(str(path))
                            transcribing = True
                        elif (
                            not transcribing
                            and not recording_announced
                            and path.exists()
                            and path.stat().st_size > 44
                        ):
                            emit("recording")
                            recording_announced = True
                        if transcribing and elapsed > config["max_seconds"] + 120:
                            raise RuntimeError(
                                "Transcription timed out. Try a shorter recording."
                            )
                finally:
                    if capture is not None and capture.poll() is None:
                        capture.kill()
                        capture.wait(timeout=2)
                    if worker is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=1)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=1)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as error:
        emit("error", message=str(error))


if __name__ == "__main__":
    main()
