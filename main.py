"""
Sift v1
Flow: DEVICE -> LANGUAGE -> EMBEDDING MODEL -> INDEX -> SELECT LLM -> QUERY -> RETRIEVE -> GENERATE ANSWER
"""

import os
import shutil
import subprocess
import sys
from typing import TypedDict, List, Optional

import ollama
import pymupdf
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

# ============================================================
# CONFIG
# ============================================================

DOC_PATH = "cosmos.pdf"

QDRANT_PATH = "./qdrant_db"

COLLECTION_NAME = "sift"

TOP_K = 4

EXIT_WORDS = {"exit", "bye", "close"}

# tier -> (ollama model tag, label)
LLM_OPTIONS = {
    "1": ("qwen2.5:7b", "Best Performance (too slower)"),
    "2": ("phi3.5", "Performance (slower)"),
    "3": ("llama3.2", "Average (faster)"),
    "4": ("llama3.2:1b", "Quick (faster)"),
}


# ============================================================
# STATE
# ============================================================

class SiftState(TypedDict):
    doc_path: str
    device: str
    english: bool
    embedding_model: str
    embeddings: object
    total_pages: int
    batch_size: Optional[int]
    vector_store: object
    llm_model: str
    query: str
    results: Optional[List[Document]]
    answer: Optional[str]


# ============================================================
# PDF HELPERS
# ============================================================

def get_pdf_page_count(pdf_path):
    with pymupdf.open(pdf_path) as pdf:
        return len(pdf)


def get_pdf_pages(pdf_path):
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            yield Document(
                page_content=page.get_text(),
                metadata={"source": pdf_path, "page": page.number},
            )


def get_batch_size(total_pages):
    if total_pages <= 100:
        return None
    elif total_pages <= 300:
        return 30
    else:
        return 50


# ============================================================
# OLLAMA MODEL AVAILABILITY
# ============================================================

def _normalize_tag(name):
    # Ollama treats "llama3.2" and "llama3.2:latest" as the same model.
    return name if ":" in name else f"{name}:latest"


def ensure_model_available(model_name):

    normalized = _normalize_tag(model_name)
    local_tags = {_normalize_tag(m.model) for m in ollama.list().models}

    if normalized in local_tags:
        print(f"✅ {model_name} already available locally.")
        return

    print(f"\n⬇️ {model_name} not found locally. Pulling now (this may take a while)...")

    for progress in ollama.pull(model_name, stream=True):
        status = progress.get("status", "")
        print(f"\r{status:<80}", end="", flush=True)

    print(f"\n✅ {model_name} downloaded.")


# ============================================================
# NODE 1: DEVICE
# ============================================================

def detect_device_node(state: SiftState) -> SiftState:

    # --------------------------------------------------------
    # launcher.py already decided this — don't ask again
    # --------------------------------------------------------

    if "--device" in sys.argv:
        device = sys.argv[sys.argv.index("--device") + 1]

        print(f"\n⚙️ Device set by launcher: {device}")

        return {**state, "device": device}

    # --------------------------------------------------------
    # Standalone run (no launcher) — detect and ask as before
    # --------------------------------------------------------

    if torch.cuda.is_available():
        print(f"\n🚀 GPU found: {torch.cuda.get_device_name(0)}")

        while True:
            choice = input("Continue with GPU or CPU? (gpu/cpu): ").strip().lower()

            if choice in ("gpu", "cpu"):
                break

            print("Please enter gpu or cpu.")

        device = "cuda" if choice == "gpu" else "cpu"

    else:
        print("\n💻 GPU not found. Continuing with CPU.")

        device = "cpu"

    return {**state, "device": device}


# ============================================================
# NODE 2: LANGUAGE
# ============================================================

def ask_language_node(state: SiftState) -> SiftState:

    while True:
        answer = input("\nIs your document completely in English? (yes/no): ").strip().lower()

        if answer in ("yes", "no"):
            break

        print("Please enter yes or no.")

    return {**state, "english": answer == "yes"}


# ============================================================
# NODE 3: EMBEDDING MODEL
# ============================================================

def choose_embeddings_node(state: SiftState) -> SiftState:

    device = state["device"]
    english = state["english"]

    # --------------------------------------------------------
    # Non-English → must use the multilingual model
    # --------------------------------------------------------

    if not english:
        model_name = "BAAI/bge-m3"

        print("\n🌍 Non-English doc → using BGE-M3")

    # --------------------------------------------------------
    # English → ask fast vs quality, regardless of device
    # --------------------------------------------------------

    else:
        while True:
            print("\nChoose embedding preference:")
            print("[1] Faster")
            print("[2] Better Quality")

            choice = input("Choose: ").strip()

            if choice == "1":
                model_name = "BAAI/bge-small-en-v1.5"

                print("\n⚡ Using BGE-small-en-v1.5")

                break

            if choice == "2":
                model_name = "BAAI/bge-m3"

                print("\n🎯 Using BGE-M3")

                break

            print("❌ Invalid choice. Please enter 1 or 2.")

    print(f"\nEmbedding model: {model_name}")
    print(f"Device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    return {**state, "embedding_model": model_name, "embeddings": embeddings}


# ============================================================
# NODE 4: INDEX
# ============================================================

def index_node(state: SiftState) -> SiftState:

    pdf_path = state["doc_path"]
    embeddings = state["embeddings"]

    total_pages = get_pdf_page_count(pdf_path)

    print(f"\n📄 PDF: {pdf_path}")
    print(f"📑 Pages: {total_pages}")

    batch_size = get_batch_size(total_pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # --------------------------------------------------------
    # SMALL PDF → load entire doc at once
    # --------------------------------------------------------

    if batch_size is None:
        print("\n📦 Strategy: Load entire PDF")

        pages = list(get_pdf_pages(pdf_path))
        chunks = splitter.split_documents(pages)

        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            path=QDRANT_PATH,
            collection_name=COLLECTION_NAME,
        )

        print(f"Pages loaded: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")

    # --------------------------------------------------------
    # LARGE PDF → lazy load in batches
    # --------------------------------------------------------

    else:
        print(f"\n📦 Strategy: Lazy load → {batch_size} pages/batch")

        vector_store = None
        page_batch = []
        processed_pages = 0

        for page in get_pdf_pages(pdf_path):
            page_batch.append(page)

            if len(page_batch) == batch_size:
                chunks = splitter.split_documents(page_batch)

                if vector_store is None:
                    vector_store = QdrantVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        path=QDRANT_PATH,
                        collection_name=COLLECTION_NAME,
                    )
                else:
                    vector_store.add_documents(chunks)

                processed_pages += len(page_batch)

                print(f"Processed {processed_pages}/{total_pages} pages → {len(chunks)} chunks")

                page_batch.clear()

        if page_batch:
            chunks = splitter.split_documents(page_batch)

            # vector_store is guaranteed to exist here — total_pages is always
            # greater than batch_size, so at least one full batch already ran.
            vector_store.add_documents(chunks)

            processed_pages += len(page_batch)

            print(f"Processed {processed_pages}/{total_pages} pages → {len(chunks)} chunks")

    print("\n✅ Indexing complete!")

    return {
        **state,
        "vector_store": vector_store,
        "total_pages": total_pages,
        "batch_size": batch_size,
    }


# ============================================================
# NODE 5: SELECT LLM
# ============================================================

def select_llm_node(state: SiftState) -> SiftState:

    print("\nSelect the LLM:")

    for key, (model_name, label) in LLM_OPTIONS.items():
        print(f"[{key}] {label} — {model_name}")

    while True:
        choice = input("Choose: ").strip()

        if choice in LLM_OPTIONS:
            break

        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

    llm_model, label = LLM_OPTIONS[choice]

    print(f"\n✅ Selected: {label} ({llm_model})")

    ensure_model_available(llm_model)

    return {**state, "llm_model": llm_model}


# ============================================================
# NODE 6: QUERY
# ============================================================

def ask_query_node(state: SiftState) -> SiftState:
    query = input("\n\nEnter your Query (or type exit/bye/close to quit): ")
    return {**state, "query": query}


def route_after_query(state: SiftState) -> str:
    return "cleanup" if state["query"].strip().lower() in EXIT_WORDS else "retrieve"


# ============================================================
# NODE 7: RETRIEVE
# ============================================================

def retrieve_node(state: SiftState) -> SiftState:

    results = state["vector_store"].similarity_search(query=state["query"], k=TOP_K)

    print(f"\n🔎 Top {len(results)} results for: \"{state['query']}\"\n")

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} (page {doc.metadata.get('page')}) ---")
        print(doc.page_content[:300])
        print()

    return {**state, "results": results}


# ============================================================
# NODE 8: GENERATE ANSWER
# ============================================================

def generate_answer_node(state: SiftState) -> SiftState:

    # --------------------------------------------------------
    # Force Ollama onto the same device chosen at the start.
    # OLLAMA_NUM_GPU=0 → CPU only. Unset/high → let it use the GPU.
    # --------------------------------------------------------

    if state["device"] == "cpu":
        os.environ["OLLAMA_NUM_GPU"] = "0"
    else:
        os.environ.pop("OLLAMA_NUM_GPU", None)

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page')}] {doc.page_content}"
        for doc in state["results"]
    )

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}"
    )

    llm = ChatOllama(model=state["llm_model"])

    response = llm.invoke(prompt)

    print("\n💬 Answer:\n")
    print(response.content)

    return {**state, "answer": response.content}


# ============================================================
# NODE 9: CLEANUP (runs on exit)
# ============================================================

def cleanup_node(state: SiftState) -> SiftState:

    print("\n👋 Goodbye!")

    # --------------------------------------------------------
    # Unload the LLM immediately, instead of waiting for
    # Ollama's default ~5 min idle timeout to free the GPU.
    # --------------------------------------------------------

    llm_model = state.get("llm_model")

    if llm_model:
        try:
            subprocess.run(["ollama", "stop", llm_model], check=False, capture_output=True)
            print(f"🧠 Unloaded {llm_model} from memory.")
        except FileNotFoundError:
            pass

    # --------------------------------------------------------
    # Release Qdrant's file lock before deleting the folder
    # --------------------------------------------------------

    vector_store = state.get("vector_store")

    if vector_store is not None:
        try:
            vector_store.client.close()
        except Exception:
            pass

    # --------------------------------------------------------
    # Wipe the collection so the next run starts fresh
    # --------------------------------------------------------

    if os.path.exists(QDRANT_PATH):
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)

        print(f"🧹 Removed {QDRANT_PATH} — Sift will start fresh next time.")

    return state


# ============================================================
# GRAPH
# ============================================================

graph_builder = StateGraph(SiftState)

graph_builder.add_node("detect_device", detect_device_node)
graph_builder.add_node("ask_language", ask_language_node)
graph_builder.add_node("choose_embeddings", choose_embeddings_node)
graph_builder.add_node("index", index_node)
graph_builder.add_node("select_llm", select_llm_node)
graph_builder.add_node("ask_query", ask_query_node)
graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate_answer", generate_answer_node)
graph_builder.add_node("cleanup", cleanup_node)

graph_builder.add_edge(START, "detect_device")
graph_builder.add_edge("detect_device", "ask_language")
graph_builder.add_edge("ask_language", "choose_embeddings")
graph_builder.add_edge("choose_embeddings", "index")
graph_builder.add_edge("index", "select_llm")
graph_builder.add_edge("select_llm", "ask_query")
graph_builder.add_conditional_edges(
    "ask_query",
    route_after_query,
    {"cleanup": "cleanup", "retrieve": "retrieve"},
)
graph_builder.add_edge("retrieve", "generate_answer")
graph_builder.add_edge("generate_answer", "ask_query")
graph_builder.add_edge("cleanup", END)

graph = graph_builder.compile()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("\nHey alien 👽")
    print("Let's run Sift.\n")

    try:
        graph.invoke({"doc_path": DOC_PATH})

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted.")

        if os.path.exists(QDRANT_PATH):
            shutil.rmtree(QDRANT_PATH, ignore_errors=True)

            print(f"🧹 Removed {QDRANT_PATH} — Sift will start fresh next time.")