# Orator

A `/record` mod for Codex CLI.

Type `/record`, speak, then press Enter. Your words go into the draft for editing.
Esc cancels. Transcription runs locally with Whisper and uses one microphone
channel. No speech API key is needed.
Recording continues until you stop it; there is no automatic time limit.

## Install

Supports Codex CLI **0.154.0** on **Linux x86_64**. Requires Git, Rust, Python 3,
`uv`, `parecord` from the PulseAudio utilities package, and a running systemd user
manager for automatic transcript cleanup. PipeWire works too.

```sh
git clone https://github.com/amarionfred/orator.git
cd orator
python3 install.py
```

Codex 0.154.0 has no extension hook for a native recording panel. The installer
downloads the official source into a local cache, applies the patch, and builds
it. The first build needs several GB of disk space and can take a while.

The repo contains the patch, speech worker, installer, and tests. It does not
contain the Codex source tree. The system Codex package is left in place;
`~/.local/bin/codex` selects the patched executable. Keep `~/.local/bin` ahead
of the system install in `PATH`, then open a fresh terminal.

The speech model is downloaded once. Recording and transcription then work offline.

## Settings

Edit `~/.config/orator/config.toml`:

```toml
model = "base"
language = "auto"
source = "default"
cpu_threads = 4
transcript_retention_hours = 24
```

Set `language = "en"` for English. To choose a microphone, run
`pactl list short sources` and put its name in `source`. A name ending in
`.monitor` records speaker output. Orator records only the first input channel,
with channel mixing disabled.

Older `max_seconds` settings are ignored. Temporary audio uses about 110 MiB per
hour in your system temporary directory (`TMPDIR` can select another disk).
Available storage still limits recording length. Transcription processes audio
in pieces of at most five minutes and has no fixed deadline; Esc still cancels it.

Dictations up to 16,000 characters go directly into the draft. Longer transcripts
are saved in full under `~/.local/share/orator/transcripts/` (or
`$XDG_DATA_HOME/orator/transcripts/`), and their file path goes into the draft for
review. Saved transcript files expire after **24 hours** by default. Set
`transcript_retention_hours` to `1` or `2` for shorter retention. A background timer
checks every 15 minutes, including when you are not recording, and catches up
after your next login if the computer was off. Starting a recording also removes
expired files. You can delete saved transcript files yourself at any time, or
copy a transcript elsewhere to keep it. This cleanup applies to Orator's saved
files; text already inserted into a Codex conversation follows Codex's history
settings. Temporary audio is removed after transcription or cancellation.

## Remove

```sh
python3 install.py --uninstall
```

This removes Orator's `codex` symlink. The next terminal session uses the system
installation. Settings, model files, and build cache are kept for reinstalling.
The cleanup timer continues to expire any remaining transcript files.

## Development

`patches/codex-0.154.0.patch` contains the native panel and draft insertion code.
`orator/backend.py` handles microphone capture and transcription.

To reuse an existing patched build:

```sh
python3 install.py --from-build /path/to/codex-checkout
```

Checks:

```sh
~/.local/share/orator/venv/bin/python -m unittest discover -s tests -v
~/.local/share/orator/venv/bin/python orator/test_capture.py
python3 orator/qa_backend.py --audio speech.wav --expect "words in the recording"
```

The audio checks need FFmpeg and use a temporary virtual input. They do not
record your microphone. Add `--record-seconds 185` to check recording past the old
cutoff, or `--backend orator/backend.py` to test the source worker with the Orator
venv's Python. The native patch includes TUI tests for draft insertion,
cancellation, and key handling.

[Apache-2.0](LICENSE). Upstream attribution is in [NOTICE](NOTICE).
