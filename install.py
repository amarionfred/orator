"""Install or remove the Orator mod for Codex CLI."""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
VERSION = "0.154.0"
COMMIT = "6b9826e3aa83b1a5947db50f4332cb9c65f1b340"
PATCH = ROOT / "patches" / f"codex-{VERSION}.patch"


def apply_patch(source, patch=PATCH):
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=source,
        capture_output=True,
    )
    if applied.returncode == 0:
        return
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)


def find_codex(data, explicit=None):
    candidates = (
        [explicit]
        if explicit
        else [Path(directory) / "codex" for directory in os.get_exec_path()]
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != data / "codex":
            return candidate.resolve()
    raise ValueError("Install Codex CLI first, or pass --codex /path/to/codex.")


def find_helper(codex):
    npm = codex.parent.parent
    candidates = [
        codex.parent / "codex-code-mode-host",
        npm
        / "node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex-code-mode-host",
        npm / "vendor/x86_64-unknown-linux-musl/bin/codex-code-mode-host",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        "Code-mode runtime not found. Pass --code-mode-host /path/to/codex-code-mode-host."
    )


def check_launcher(launcher, data):
    if launcher.exists() or launcher.is_symlink():
        if not launcher.is_symlink() or launcher.resolve() != data / "codex":
            raise ValueError(f"Refusing to replace unrelated {launcher}.")


def uninstall(launcher, data):
    check_launcher(launcher, data)
    launcher.unlink(missing_ok=True)
    print("Orator disabled. Open a new terminal to use the system Codex.")


def install_worker(data, cache, config):
    data.mkdir(parents=True, exist_ok=True)
    python = data / "venv/bin/python"
    if not python.exists():
        subprocess.run(
            ["uv", "venv", "--python", "3.12", str(data / "venv")], check=True
        )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(ROOT / "orator/requirements.txt"),
        ],
        check=True,
    )
    backend = data / "backend.py"
    backend.write_text(f"#!{python}\n" + (ROOT / "orator/backend.py").read_text())
    backend.chmod(0o755)
    (data / "bin").mkdir(exist_ok=True)
    worker = data / "bin/orator-backend"
    worker.unlink(missing_ok=True)
    worker.symlink_to(backend)
    model = cache / "models/base"
    if not (model / "model.bin").is_file():
        subprocess.run(
            [
                str(python),
                "-c",
                "from huggingface_hub import snapshot_download; "
                "import sys; snapshot_download('Systran/faster-whisper-base', local_dir=sys.argv[1])",
                str(model),
            ],
            check=True,
        )
    config.mkdir(parents=True, exist_ok=True)
    settings = config / "config.toml"
    if not settings.exists():
        settings.write_text(
            'model = "base"\nlanguage = "auto"\nsource = "default"\n'
            "max_seconds = 180\ncpu_threads = 4\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--codex", type=Path, help="System Codex executable")
    parser.add_argument(
        "--code-mode-host", type=Path, help="Matching Codex code-mode runtime"
    )
    parser.add_argument(
        "--from-build",
        type=Path,
        help="Reuse an already patched and built Codex checkout",
    )
    parser.add_argument(
        "--jobs", type=int, default=2, help="Parallel build jobs (default: 2)"
    )
    args = parser.parse_args()
    data = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "orator"
    )
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "orator"
    config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "orator"
    launcher = Path.home() / ".local/bin/codex"
    if args.uninstall:
        uninstall(launcher, data)
        return
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("This mod currently supports Linux x86_64.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    check_launcher(launcher, data)
    for tool in ("git", "uv", "parecord"):
        if shutil.which(tool) is None:
            raise ValueError(f"Required command not found: {tool}")
    codex = find_codex(data, args.codex)
    version = subprocess.check_output([str(codex), "--version"], text=True).strip()
    if version != f"codex-cli {VERSION}":
        raise ValueError(f"This patch supports Codex {VERSION}; found {version}.")
    helper = (
        args.code_mode_host.resolve() if args.code_mode_host else find_helper(codex)
    )
    if not helper.is_file():
        raise ValueError(f"Code-mode runtime not found: {helper}")
    if args.from_build:
        source = args.from_build.resolve()
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(PATCH)], cwd=source, check=True
        )
    else:
        if shutil.which("cargo") is None:
            raise ValueError("Install Rust before building the mod.")
        cache.mkdir(parents=True, exist_ok=True)
        source = cache / f"codex-{VERSION}"
        if not source.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    f"rust-v{VERSION}",
                    "https://github.com/openai/codex.git",
                    str(source),
                ],
                check=True,
            )
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        if revision != COMMIT:
            raise ValueError(f"Unexpected Codex revision in {source}.")
        apply_patch(source)
        print(
            "Building Codex with the Orator patch. The first build can take a while.",
            flush=True,
        )
        subprocess.run(
            [
                "cargo",
                "+stable",
                "build",
                "--locked",
                "--profile",
                "dev-small",
                "-p",
                "codex-cli",
                "--bin",
                "codex",
                "-j",
                str(args.jobs),
            ],
            cwd=source / "codex-rs",
            check=True,
        )
    binary = source / "codex-rs/target/dev-small/codex"
    built_version = subprocess.check_output(
        [str(binary), "--version"], text=True
    ).strip()
    if built_version != version:
        raise ValueError("Built and installed Codex versions differ.")
    with (
        binary.open("rb") as file,
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as content,
    ):
        if content.find(b"orator/bin/orator-backend") == -1:
            raise ValueError(
                "The binary does not contain the Orator patch. Rebuild it first."
            )
    install_worker(data, cache, config)
    for original, name in ((binary, "codex"), (helper, "codex-code-mode-host")):
        temporary = data / (name + ".new")
        shutil.copy2(original, temporary)
        temporary.replace(data / name)
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.unlink(missing_ok=True)
    launcher.symlink_to(data / "codex")
    (data / "install.json").write_text(
        json.dumps(
            {
                "codex_version": VERSION,
                "upstream_commit": COMMIT,
                "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
                "system_codex": str(codex),
            },
            indent=2,
        )
        + "\n"
    )
    print("Orator installed. Open codex and use /record.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
