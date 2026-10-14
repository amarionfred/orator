"""Local transcription worker: `stop` on stdin, JSON events on stdout, EOF to cancel."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import tomllib
import unicodedata


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
    data.setdefault("cpu_threads", 4)
    if not 1 <= data["cpu_threads"] <= 64:
        raise ValueError("Invalid cpu_threads in Orator's config.")
    data["transcripts"] = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        / "orator/transcripts"
    )
    if not (Path(data["model"]) / "model.bin").is_file():
        raise ValueError("Local speech model missing. Run the Orator installer.")
    return data


def audio_chunks(path):
    """Read bounded pieces of mono 16 kHz s16le PCM, preferring quiet boundaries."""
    import numpy as np

    chunk_samples = 16000 * 300
    size = path.stat().st_size
    with path.open("rb") as audio:
        while raw := audio.read(chunk_samples * 2):
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if audio.tell() < size:
                # Split in the quietest 100 ms of the final five seconds. Seek
                # back so every sample is processed exactly once, including speech.
                windows = samples[-16000 * 5 :].reshape(-1, 1600)
                energy = (windows * windows).mean(axis=1)
                cut = len(samples) - int(np.argmin(energy[::-1])) * 1600 - 800
                audio.seek((cut - len(samples)) * 2, os.SEEK_CUR)
                samples = samples[:cut]
            yield samples


def transcript_result(path, directory):
    """Keep the draft bounded while preserving a long transcript in a private file."""
    with path.open(encoding="utf-8") as transcript:
        text = transcript.read(16001)
        if len(text) <= 16000:
            return text
        directory.mkdir(parents=True, exist_ok=True)
        destination = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix="dictation-",
                suffix=".txt",
                dir=directory,
                delete=False,
            ) as saved:
                destination = Path(saved.name)
                transcript.seek(0)
                shutil.copyfileobj(transcript, saved)
            return f"Full transcript saved to: {destination}"
        except BaseException:
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise


def recognize(connection, config):
    try:
        from faster_whisper import WhisperModel
        from numpy import trim_zeros

        model = WhisperModel(
            config["model"],
            device="cpu",
            compute_type="int8",
            cpu_threads=config["cpu_threads"],
            local_files_only=True,
        )
        path = Path(connection.recv())
        transcript = path.with_suffix(".txt")
        heard_speech = confident = False
        separator = ""
        with transcript.open("w", encoding="utf-8") as output:
            for samples in audio_chunks(path):
                if (abs(samples) >= 32440 / 32768).mean() > 0.02:
                    raise ValueError(
                        "Microphone audio is distorted. Lower your microphone input volume and retry."
                    )
                # Skip digital silence without letting a long muted interval
                # affect speech detection. Every nonzero sample is preserved.
                samples = trim_zeros(samples)
                if not samples.size:
                    continue
                segments, _ = model.transcribe(
                    samples,
                    language=None
                    if config["language"] == "auto"
                    else config["language"],
                    beam_size=5,
                    temperature=0.0,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                for segment in segments:
                    heard_speech = True
                    if segment.avg_logprob < -1.0 or segment.compression_ratio > 2.4:
                        continue
                    confident = True
                    text = "".join(
                        c
                        for c in segment.text
                        if not unicodedata.category(c).startswith("C")
                    )
                    text = " ".join(text.split())
                    if text:
                        output.write(separator + text)
                        separator = " "
        if heard_speech and not confident:
            raise ValueError(
                "Couldn't hear speech clearly. Speak closer to the microphone and retry."
            )
        connection.send(dict(event="result", transcript=str(transcript)))
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
            path = Path(directory) / "speech.pcm"
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
                            "--raw",
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
                            if result["event"] == "result":
                                result["text"] = transcript_result(
                                    Path(result.pop("transcript")),
                                    config["transcripts"],
                                )
                            emit(**result)
                            return
                        if not worker.is_alive():
                            raise RuntimeError(
                                "Speech worker stopped unexpectedly. Try /record again."
                            )
                        if not transcribing and (
                            control == "stop" or capture.poll() is not None
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
                            and path.stat().st_size > 0
                        ):
                            emit("recording")
                            recording_announced = True
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
