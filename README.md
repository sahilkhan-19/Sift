# Sift 🔍

**Sift** is a local-first RAG (Retrieval-Augmented Generation) knowledge retrieval system for PDF documents. It runs 100% locally on your machine with zero external API calls, dynamic GPU/CPU hardware calibration, local HuggingFace embeddings, local Qdrant vector storage, and Ollama LLMs.

---

## Features

- 🔒 **100% Local & Private**: No cloud APIs, no telemetry, no subscription fees. Your documents never leave your computer.
- ⚡ **Dynamic Hardware Calibration**: Automatically detects NVIDIA GPUs (`nvidia-smi`) and installs matching PyTorch CUDA or CPU wheels on demand.
- 📄 **Native File Picker**: Select any PDF through a native OS GUI file dialog.
- 🧠 **Smart Embedding Selection**: Chooses between lightweight English (`BGE-small-en-v1.5`) or multilingual (`BGE-M3`) models based on your document.
- 📊 **Dynamic Indexing & Progress**: Intelligent batching strategies for large PDFs with real-time Rich progress bars.
- 🤖 **Ollama Integration**: Interactive selection of local LLM models (`qwen2.5:7b`, `phi3.5`, `llama3.2`, `llama3.2:1b`) with automatic pulling.
- 🧹 **Automatic Cleanup**: Unloads LLMs from VRAM/RAM on exit and wipes transient vector databases (`./qdrant_db`) for fresh subsequent runs.

---

## Prerequisites

Before running Sift, ensure you have the following installed on your system:

1. **Python**: Version `3.12` or `3.13` (`>=3.12, <3.14`).
2. **Ollama**: Download and install [Ollama](https://ollama.com/). Ensure the Ollama service is running in the background.
3. **uv Package Manager**: Astral's fast Python package manager.
4. *(Optional)* **NVIDIA GPU**: NVIDIA graphics card with `nvidia-smi` on PATH for CUDA acceleration. (CPU execution is fully supported out of the box).

---

## Getting Started (Windows)

Follow these exact steps to set up and run Sift on Windows.

### 1. Install `uv`

Open PowerShell as your normal user and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify `uv` installation:

```powershell
uv --version
```

### 2. Clone the Repository & Navigate to Sift

```powershell
git clone https://github.com/sahilkhan-19/Sift.git
cd Sift
```

### 3. Install Dependencies

Install all required Python dependencies into a managed `.venv` using `uv`:

```powershell
uv sync
```

### 4. Launch Sift

Run the launcher script to automatically calibrate hardware (NVIDIA GPU / CPU) and start Sift:

```powershell
uv run launcher.py
```

*Alternatively, you can run `python sift_exe.py` which automatically runs `uv sync` on first launch before executing `launcher.py`.*

---

## Usage Workflow

```
[Launcher Calibration] ➔ [PDF GUI Picker] ➔ [Language & Embedding Setup]
       ➔ [Qdrant Indexing] ➔ [LLM Tier Selection] ➔ [Interactive Q&A Loop]
```

### Step 1: Hardware Calibration (`launcher.py`)
- Sift checks for an NVIDIA GPU using `nvidia-smi`.
- If an NVIDIA GPU is detected but PyTorch is CPU-only, Sift prompts you to select **CPU** or **GPU**, automatically installs the required PyTorch wheels silently, and restarts.

### Step 2: PDF Selection
- A native Windows file picker dialog opens.
- Browse and select the `.pdf` file you want to query.

### Step 3: Language & Embedding Model Setup
- Sift asks if your document is strictly in English:
  - **English**: Choose between **Faster** (`BAAI/bge-small-en-v1.5`) or **Better Quality** (`BAAI/bge-m3`).
  - **Non-English**: Automatically selects the multilingual model (`BAAI/bge-m3`).

### Step 4: Indexing
- Sift reads the PDF with PyMuPDF and applies an optimal batching strategy:
  - **$\le 100$ pages**: Processes the entire document in memory.
  - **$101–300$ pages**: Batches 30 pages at a time with a live progress bar.
  - **$> 300$ pages**: Batches 50 pages at a time with a live progress bar.
- Text is split into 1,000-character chunks with 200-character overlaps using `RecursiveCharacterTextSplitter`.
- Chunks are embedded and stored in a temporary local Qdrant database (`./qdrant_db`).

### Step 5: Select LLM
Choose from four pre-configured local LLM performance tiers:

| Tier | Model Tag | Description |
| :--- | :--- | :--- |
| **1** | `qwen2.5:7b` | Best Performance (Slower) |
| **2** | `phi3.5` | High Performance (Slower) |
| **3** | `llama3.2` | Balanced / Average (Faster) |
| **4** | `llama3.2:1b` | Quick / Lightweight (Faster) |

*If the selected model is not installed locally, Sift automatically downloads it via Ollama.*

### Step 6: Interactive Q&A
- Enter your question at the prompt.
- Sift performs cosine similarity search over the Qdrant index to retrieve the top 4 ($k=4$) most relevant PDF chunks.
- Sift passes the retrieved context with page references to the LLM and streams/renders the answer in a formatted Rich terminal panel.
- Ask follow-up questions or type `exit`, `bye`, or `close` to quit.

### Step 7: Session Cleanup
When you exit:
- Sift immediately unloads the LLM from RAM/VRAM via `ollama stop`.
- Sift releases file locks and deletes `./qdrant_db` so your next session starts clean.

---

## Packaging as a Standalone Executable (`Sift.exe`)

To compile Sift into a standalone Windows executable (`Sift.exe`):

1. Install PyInstaller:
   ```powershell
   uv pip install pyinstaller
   ```
2. Run the build batch script:
   ```powershell
   .\build_exe.bat
   ```
3. Copy the generated `dist\Sift.exe` to your project root folder and double-click to run.

---

## Environment Variables & Configuration

No `.env` file or external API tokens are required.

- **`OLLAMA_NUM_GPU`**: Managed internally by Sift (`0` when running on CPU; unset when running on GPU).
- **`NO_COLOR`**: Set `NO_COLOR=1` if you want to disable Rich colorized terminal output.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
