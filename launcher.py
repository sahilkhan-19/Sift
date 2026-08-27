"""
Sift Launcher

Responsible for:
    1. Detecting NVIDIA GPU
    2. Choosing CPU/GPU
    3. Installing the correct PyTorch build
    4. Restarting Sift with the correct environment
"""

import subprocess
import sys

# ============================================================
# CONFIG
# ============================================================

PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"


# ============================================================
# NVIDIA GPU
# ============================================================


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


# ============================================================
# CHECK PYTORCH
# ============================================================


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


# ============================================================
# INSTALL PYTORCH
# ============================================================


def install_pytorch(index_url):

    print("\n📦 Installing PyTorch...")

    result = subprocess.run(
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
        check=False,
    )

    if result.returncode != 0:
        print("\n❌ Failed to install PyTorch.")

        sys.exit(1)


# ============================================================
# RESTART SIFT
# ============================================================


def restart_sift(device):
    print("\n🔄 Restarting Sift with the new PyTorch environment...\n")
    result = subprocess.run(
        [sys.executable, "main.py", "--device", device],
        check=False,
    )
    sys.exit(result.returncode)


# ============================================================
# START SIFT
# ============================================================


def start_sift():

    import torch

    # --------------------------------------------------------
    # Verify PyTorch
    # --------------------------------------------------------

    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")

        print(f"CUDA: {torch.version.cuda}")

    else:
        print("\n💻 Using CPU")

    print("\n🚀 Starting Sift...")

    # --------------------------------------------------------
    # Run main.py
    # --------------------------------------------------------

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
        ],
        check=False,
    )

    sys.exit(result.returncode)


# ============================================================
# MAIN LAUNCHER
# ============================================================


def main():

    # ========================================================
    # AFTER RESTART
    # ========================================================

    if "--start" in sys.argv:
        start_sift()

        return

    # ========================================================
    # GPU DETECTION
    # ========================================================

    has_gpu, driver_cuda = check_nvidia_gpu()

    if has_gpu:
        print("\n🚀 NVIDIA GPU detected!")

        if driver_cuda:
            print(f"Driver CUDA support: {driver_cuda}")

    else:
        print("\n💻 No NVIDIA GPU detected.")

        print("Sift will use CPU.")

        install_pytorch(PYTORCH_CPU_INDEX)

        restart_sift("cpu")

        return

    # ========================================================
    # CHECK CURRENT PYTORCH
    # ========================================================

    pytorch = check_pytorch()

    if pytorch["installed"]:
        print("\nCurrent PyTorch:")

        print(f"    {pytorch['version']}")

    # ========================================================
    # ALREADY CUDA
    # ========================================================

    if pytorch["available"]:
        print("\n🚀 CUDA-enabled PyTorch is already active.")

        restart_sift("cuda")

        return

    # ========================================================
    # GPU EXISTS BUT PYTORCH IS CPU
    # ========================================================

    print("\nCUDA-enabled PyTorch is not currently active.")

    while True:
        print("\nHow do you want Sift to run?")

        print("\n[1] CPU")

        print("[2] GPU")

        choice = input("\nChoose: ").strip()

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        if choice == "1":
            print("\n⚙️ Starting Sift with CPU...")

            install_pytorch(PYTORCH_CPU_INDEX)

            restart_sift("cpu")

            return

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        if choice == "2":
            print("\n⚙️ Starting Sift with GPU...")

            install_pytorch(PYTORCH_CUDA_INDEX)

            restart_sift("cuda")

            return

        print("\n❌ Invalid choice.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()