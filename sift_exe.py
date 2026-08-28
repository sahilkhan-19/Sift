"""
Sift.exe entry point

This script is compiled to Sift.exe via PyInstaller. When double-clicked:

    1. First run  → runs `uv sync` to install all dependencies
    2. Every run  → runs `uv run launcher.py` (the normal Sift startup)

The PowerShell / console window stays visible the entire time so the user
can see Sift's output, spinner, and any errors.
"""

import os
import shutil
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


def find_uv_executable() -> str | None:
    """Find the path to the uv executable."""
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path

    # Common Windows installation directories for uv
    home = os.path.expanduser("~")
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    app_data = os.environ.get("APPDATA", "")

    candidates = [
        os.path.join(home, ".cargo", "bin", "uv.exe"),
        os.path.join(home, ".local", "bin", "uv.exe"),
        os.path.join(local_app_data, "bin", "uv.exe"),
        os.path.join(app_data, "uv", "uv.exe"),
        os.path.join(home, "AppData", "Local", "Programs", "uv", "uv.exe"),
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    return None


def main() -> None:

    # ── Locate the Sift project directory ─────────────────────
    if getattr(sys, "frozen", False):
        project_dir = os.path.dirname(sys.executable)
    else:
        project_dir = os.path.dirname(os.path.abspath(__file__))

    os.chdir(project_dir)

    # ── Verify uv is available ────────────────────────────────
    uv_bin = find_uv_executable()

    if not uv_bin:
        _tag("ERROR", "'uv' is not installed or not found on PATH.")
        _tag("SETUP", "Please run this command in PowerShell to install uv:")
        print()
        print('  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"')
        print()
        _tag("SETUP", "After installing uv, double-click Sift.exe again.")
        _pause_on_exit(1)

    try:
        subprocess.run(
            [uv_bin, "--version"],
            capture_output=True,
            check=True,
        )
    except Exception:
        _tag("ERROR", f"Failed to execute uv at '{uv_bin}'.")
        _pause_on_exit(1)

    # ── First-run check: uv sync ──────────────────────────────
    venv_dir = os.path.join(project_dir, ".venv")

    if not os.path.isdir(venv_dir):
        _tag("SETUP", "First run detected — installing dependencies...")
        _tag("SETUP", "This may take a few minutes...\n")

        result = subprocess.run([uv_bin, "sync"], check=False)

        if result.returncode != 0:
            _tag("ERROR", f"`uv sync` failed (exit code {result.returncode}).")
            _pause_on_exit(1)

        _tag("  OK  ", "Dependencies installed.\n")

    # ── Launch Sift ───────────────────────────────────────────
    _tag("SETUP", "Starting Sift...\n")

    result = subprocess.run(
        [uv_bin, "run", "launcher.py"],
        check=False,
    )

    if result.returncode != 0:
        _tag("ERROR", f"Sift exited with code {result.returncode}.")
        _pause_on_exit(result.returncode)


if __name__ == "__main__":
    main()
