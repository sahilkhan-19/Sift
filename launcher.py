import subprocess
import sys
import threading
import time

PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"

_G = "\033[92m"
_D = "\033[90m"
_R = "\033[0m"
_B = "\033[1m"
_W = "\033[97m"

_TAG_W = 7


def _print(tag: str, msg: str) -> None:
    sys.stdout.write(f"{_G}[[{tag:^{_TAG_W}}]]{_R}  {msg}\n")
    sys.stdout.flush()


def _ok(msg: str) -> None:
    _print("OK", msg)


def _warn(msg: str) -> None:
    _print("WARN", msg)


def _error(msg: str) -> None:
    _print("ERROR", msg)


def _setup(msg: str) -> None:
    _print("SETUP", msg)


def _gpu(msg: str) -> None:
    _print("GPU", msg)


def _cpu(msg: str) -> None:
    _print("CPU", msg)


_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _spin(label: str, thread: threading.Thread) -> None:
    """Show a spinner in-place while `thread` is alive."""
    i = 0
    while thread.is_alive():
        frame = _SPINNER[i % len(_SPINNER)]
        sys.stdout.write(f"\r{_G}[[{' SETUP ':^{_TAG_W}}]]{_R}  {frame}  {label}...")
        sys.stdout.flush()
        time.sleep(0.07)
        i += 1
    # clear the spinner line
    sys.stdout.write(f"\r{' ' * 60}\r")
    sys.stdout.flush()


def check_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode != 0:
            return False, None

        cuda_version = None

        for line in result.stdout.splitlines():
            if "CUDA Version" in line:
                cuda_version = line.split("CUDA Version:")[-1].strip()

                break

        return True, cuda_version

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        OSError,
    ):
        return False, None


def check_pytorch():

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return {
            "installed": False,
            "version": None,
            "cuda": None,
            "available": False,
        }

    lines = result.stdout.strip().splitlines()

    if len(lines) < 3:
        return {
            "installed": False,
            "version": None,
            "cuda": None,
            "available": False,
        }

    return {
        "installed": True,
        "version": lines[0],
        "cuda": lines[1],
        "available": lines[2].lower() == "true",
    }


def install_pytorch(index_url):

    result_holder = [None]

    def _run():
        result_holder[0] = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--index-url",
                index_url,
                "--reinstall",
            ],
            capture_output=True,
            check=False,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    _spin("Calibrating environment", thread)

    thread.join()

    if result_holder[0] is not None and result_holder[0].returncode != 0:
        _error("PyTorch installation failed.")
        sys.exit(1)

    _ok("Environment ready.")


def restart_sift(device):
    _setup(f"Restarting Sift  [{device.upper()}]")
    result = subprocess.run(
        [sys.executable, "main.py", "--device", device],
        check=False,
    )
    sys.exit(result.returncode)


def start_sift():

    import torch

    if torch.cuda.is_available():
        _gpu(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        _cpu("Device: CPU")

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
        ],
        check=False,
    )

    sys.exit(result.returncode)


def main():

    if "--start" in sys.argv:
        start_sift()

        return

    has_gpu, driver_cuda = check_nvidia_gpu()

    if has_gpu:
        label = (
            f"NVIDIA GPU detected  [CUDA {driver_cuda}]"
            if driver_cuda
            else "NVIDIA GPU detected"
        )
        _gpu(label)

    else:
        _cpu("No NVIDIA GPU detected — using CPU.")

        install_pytorch(PYTORCH_CPU_INDEX)

        restart_sift("cpu")

        return

    pytorch = check_pytorch()

    if pytorch["installed"]:
        _setup(f"PyTorch {pytorch['version']}")

    if pytorch["available"]:
        _ok("CUDA-enabled PyTorch active.")

        restart_sift("cuda")

        return

    _warn("CUDA PyTorch not active.")

    while True:
        sys.stdout.write(
            f"\n{_G}[[{' SETUP ':^{_TAG_W}}]]{_R}  Run Sift on  [1] CPU  [2] GPU : {_W}"
        )
        sys.stdout.flush()
        choice = input().strip()
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
        if choice == "1":
            _setup("Starting with CPU...")

            install_pytorch(PYTORCH_CPU_INDEX)

            restart_sift("cpu")

            return
        if choice == "2":
            _setup("Starting with GPU...")

            install_pytorch(PYTORCH_CUDA_INDEX)

            restart_sift("cuda")

            return

        _error("Invalid choice — enter 1 or 2.")


if __name__ == "__main__":
    main()
