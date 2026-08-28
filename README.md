<div align="center">
  <h1>🔍 Sift</h1>
  <p><strong>A fast, local-first RAG (Retrieval-Augmented Generation) system for querying PDF documents.</strong></p>
</div>

Sift is a command-line application that allows you to chat with your PDF documents locally. It leverages local embedding models and local LLMs through Ollama, ensuring your data never leaves your machine. Sift features a dynamic hardware calibration system to seamlessly run on both CPU and NVIDIA GPUs, a beautiful terminal UI, and a native file picker for selecting your documents.

## ✨ Features

- **100% Local & Private:** No cloud APIs, no telemetry. Your documents and queries remain on your computer.
- **Dynamic Hardware Calibration:** Automatically detects NVIDIA GPUs (via `nvidia-smi`) and installs the optimal PyTorch build (CUDA or CPU) on demand using `uv`.
- **Native GUI File Picker:** Seamlessly select PDFs using a native OS file dialog.
- **Smart Embedding Selection:** Intelligently chooses between fast English models (`BGE-small-en-v1.5`) and robust multilingual models (`BGE-M3`) based on your input.
- **Optimized Indexing:** Uses dynamic batching for large PDFs (30 or 50 pages at a time) with real-time progress bars to manage memory efficiently.
- **Automated LLM Management:** Choose from curated performance tiers. Sift automatically pulls missing models via Ollama and immediately unloads them from VRAM upon exit.
- **Clean State Execution:** Transient vector databases (`qdrant_db`) are automatically wiped on exit, ensuring a fresh start every time.

## 🏗️ Architecture & Workflow

Sift operates on a finite state machine powered by **LangGraph**, structuring the interaction into a clear pipeline:

1. **Hardware Calibration (`launcher.py`):** Detects if an NVIDIA GPU is available, verifies the current PyTorch installation, and installs the appropriate CUDA or CPU wheels if needed.
2. **Initialization:** A native Windows file picker prompts the user to select a PDF document.
3. **Indexing:**
   - **Parse:** Extracts text using PyMuPDF.
   - **Chunk:** Splits text into 1,000-character chunks with 200-character overlaps using `RecursiveCharacterTextSplitter`.
   - **Embed & Store:** Generates HuggingFace embeddings and stores them in a temporary local Qdrant database.
4. **LLM Selection:** The user selects a performance tier (e.g., `qwen2.5:7b`, `phi3.5`, `llama3.2`). Sift ensures the model is available locally.
5. **Retrieval & Generation:** For each query, Sift performs a cosine similarity search to fetch the top 3 most relevant chunks and generates an answer strictly based on that context.
6. **Cleanup:** Unloads the LLM from memory and purges the vector database upon exiting.

## 🚀 Getting Started

### Prerequisites
- **Python:** `3.12` or `3.13`
- **Ollama:** Installed and running in the background. ([Download Ollama](https://ollama.com/))
- **uv:** Astral's fast Python package manager.

### 1. Install `uv` (Windows)
Open PowerShell and run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone & Install
```powershell
git clone https://github.com/sahilkhan-19/Sift.git
cd Sift
uv sync
```
*`uv sync` will automatically create a `.venv` and install the base dependencies specified in `pyproject.toml`.*

### 3. Run Sift
Start Sift by running the launcher. It will handle checking for an NVIDIA GPU and installing the correct PyTorch wheels before launching the main application.

```powershell
uv run launcher.py
```
*Alternatively, running `uv run sift_exe.py` will automatically sync dependencies on the first run before starting the launcher.*

## 📖 How to Use Sift

1. **Launch:** Run the launcher command. Sift will configure PyTorch for your hardware.
2. **Select PDF:** A file dialog will pop up. Choose the PDF you want to query.
3. **Language Setup:** Specify if your document is strictly in English to help Sift pick the best embedding model.
4. **Wait for Indexing:** Sift will parse and index your document. You will see a progress bar for larger files.
5. **Select LLM:** Choose an LLM tier. Sift will automatically download the model via Ollama if you don't have it.
   - `1`: `qwen2.5:7b` (Best Performance)
   - `2`: `phi3.5` (High Performance)
   - `3`: `llama3.2` (Balanced)
   - `4`: `llama3.2:1b` (Quick)
6. **Chat:** Ask questions about your document! Type `exit`, `bye`, or `close` to cleanly shut down.

## ⚙️ Configuration & Customization

Sift relies on internal variables rather than external `.env` files.
- **`NO_COLOR`**: Set the `NO_COLOR=1` environment variable to disable colorized terminal output.
- **Standalone Executable**: You can package Sift into a standalone `Sift.exe` binary.
  ```powershell
  uv pip install pyinstaller
  .\build_exe.bat
  ```

## 📂 Project Structure

- `main.py`: Application core, implements the LangGraph pipeline, indexing, and LLM interaction.
- `launcher.py`: Hardware calibration layer. Validates and installs correct PyTorch CPU/CUDA wheels.
- `sift_exe.py`: Executable entry point. Bootstraps the environment on the first run.
- `ui.py`: Centralized Rich UI module for themes, spinners, progress bars, and banners.
- `pyproject.toml` / `uv.lock`: Project metadata, dependency definitions, and custom uv PyTorch indices.

## 🛠️ Tech Stack

- **Runtime & Package Manager:** Python 3.12+, `uv`
- **Workflow & RAG:** LangChain, LangGraph
- **PDF Processing:** PyMuPDF (`pymupdf`)
- **Vector Database:** Qdrant (Local disk instance)
- **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`, `BAAI/bge-m3`)
- **LLM Engine:** Ollama (`langchain-ollama`)
- **User Interface:** Rich (Terminal styling), `tkinter` (File picker)
