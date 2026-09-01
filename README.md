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

### 1. Install `uv` and Ollama
Open PowerShell and run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Then install [Ollama](https://ollama.com/) and make sure it's running in the background.

### 2. Add your Hugging Face token
Grab a free token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Role: "Read" is enough), then paste it into the `.env` file:
```
HF_TOKEN="paste_your_token_here"
```

### 3. Run Sift
```powershell
sift.exe
```

That's it — Sift takes care of GPU detection, PyTorch setup, and everything else from there.

## 🎥 Demo

<!-- Drop your tutorial .mp4 here via the GitHub web editor (drag-and-drop) to embed a playable demo -->


https://github.com/user-attachments/assets/b2d378a3-8900-4f54-b332-bcdfe113e98c


## 📖 How to Use Sift

1. **Launch:** Run `sift.exe`. Sift will configure PyTorch for your hardware.
2. **Select PDF:** A file dialog will pop up. Choose the PDF you want to query.
3. **Language Setup:** Specify if your document is strictly in English to help Sift pick the best embedding model.
4. **Wait for Indexing:** Sift will parse and index your document. You will see a progress bar for larger files.
5. **Select LLM:** Choose an LLM tier. Sift will automatically download the model via Ollama if you don't have it.
   - `1`: `qwen2.5:7b` (Best Performance)
   - `2`: `phi3.5` (High Performance)
   - `3`: `llama3.2` (Balanced)
   - `4`: `llama3.2:1b` (Quick)
6. **Chat:** Ask questions about your document! Type `exit`, `bye`, or `close` to cleanly shut down.


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
