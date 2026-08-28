"""
Sift.exe entry point

This script is compiled to Sift.exe via PyInstaller.  When double-clicked:

    1. First run  → runs `uv sync` to install all dependencies
    2. Every run  → runs `uv run launcher.py` (the normal Sift startup)

The PowerShell / console window stays visible the entire time so the user
can see Sift's output, spinner, and any errors.
"""

import os
import subprocess
import sys

# ── ANSI helpers (same style as launcher.py) ──────────────────

_G = "\033[92m"
_R = "\033[0m"
_TAG_W = 7


def _tag(tag: str, msg: str) -> None:
    sys.stdout.write(f"{_G}[[{tag:^{_TAG_W}}]]{_R}  {msg}\n")
    sys.stdout.flush()


def _pause_on_exit(code: int = 1) -> None:
    """Pause so the window doesn't vanish on error."""
    print()
    input("Press Enter to close...")
    sys.exit(code)


def main() -> None:

    # ── Locate the Sift project directory ─────────────────────
    #   When frozen (PyInstaller exe), the exe sits in the project dir.
    #   When running as plain .py, use the script's own directory.

    if getattr(sys, "frozen", False):
        project_dir = os.path.dirname(sys.executable)
    else:
        project_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(project_dir)

    # ── Verify uv is available ────────────────────────────────

    try:
        subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        _tag("ERROR", "'uv' is not installed or not on PATH.")
        _tag("ERROR", "Install it from: https://docs.astral.sh/uv/")
        _pause_on_exit(1)

    # ── First-run check: uv sync ──────────────────────────────
    #   If .venv doesn't exist yet, the project hasn't been synced.

    venv_dir = os.path.join(project_dir, ".venv")

    if not os.path.isdir(venv_dir):
        _tag("SETUP", "First run detected — installing dependencies...")
        _tag("SETUP", "This may take a few minutes.\n")

        result = subprocess.run(["uv", "sync"], check=False)

        if result.returncode != 0:
            _tag("ERROR", "`uv sync` failed (exit code {}).".format(result.returncode))
            _pause_on_exit(1)

        _tag("  OK  ", "Dependencies installed.\n")

    # ── Launch Sift ───────────────────────────────────────────

    _tag("SETUP", "Starting Sift...\n")

    result = subprocess.run(
        ["uv", "run", "launcher.py"],
        check=False,
    )

    if result.returncode != 0:
        _tag("ERROR", "Sift exited with code {}.".format(result.returncode))
        _pause_on_exit(result.returncode)


if __name__ == "__main__":
    main()
