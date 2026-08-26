"""
Sift
INDEXING: DOC LOAD -> SPLIT -> EMBEDDING -> VECTOR DB
"""

import pymupdf
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# CONFIG
# ============================================================

DOC_PATH = "cosmos.pdf"

QDRANT_PATH = "./qdrant_db"

COLLECTION_NAME = "sift"


# ============================================================
# USER PREFERENCES
# ============================================================


def get_user_preferences():

    print("\nHey alien 👽")
    print("Let's configure Sift.\n")

    while True:
        answer = input("Is your PDF entirely in English? (yes/no): ").strip().lower()

        if answer in ("yes", "no"):
            break

        print("Please enter yes or no.")

    return {"english": answer == "yes"}


# ============================================================
# PDF
# ============================================================


def get_pdf_page_count(pdf_path):

    with pymupdf.open(pdf_path) as pdf:
        return len(pdf)


def get_pdf_pages(pdf_path):
    """
    Lazily yields one PDF page at a time.
    """

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            yield Document(
                page_content=page.get_text(),
                metadata={
                    "source": pdf_path,
                    "page": page.number,
                },
            )


# ============================================================
# BATCH STRATEGY
# ============================================================


def get_batch_size(total_pages):

    if total_pages <= 100:
        return None

    elif total_pages <= 300:
        return 30

    else:
        return 50


# ============================================================
# EMBEDDINGS
# ============================================================
def get_embeddings(english, device):

    # ========================================================
    # GPU → BGE-M3
    # ========================================================

    if device == "cuda":
        model_name = "BAAI/bge-m3"

        print("\n🚀 GPU detected → using BGE-M3")
        print("Quality mode")

    # ========================================================
    # CPU → ASK USER
    # ========================================================

    else:
        print("\n💻 Running on CPU.")

        print("\nChoose your embedding preference:")

        print("\n[1] Faster")
        print("[2] Better Quality")

        while True:
            choice = input("\nChoose: ").strip()

            if choice == "1":
                if english:
                    model_name = "BAAI/bge-small-en-v1.5"

                    print("\n⚡ Using BGE-small-en-v1.5")

                    print("Mode: Faster")

                else:
                    model_name = "BAAI/bge-m3"

                    print("\n🌍 Non-English PDF → using BGE-M3")

                break

            if choice == "2":
                model_name = "BAAI/bge-m3"

                print("\n🎯 Using BGE-M3")

                print("Mode: Better Quality")

                break

            print("\n❌ Invalid choice.")
            print("Please enter 1 or 2.")

    print(f"\nEmbedding model: {model_name}")

    print(f"Device: {device}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# VECTOR STORE
# ============================================================


def create_vector_store(chunks, embeddings):

    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )


# ============================================================
# INDEXING
# ============================================================


def store_qdrant(pdf_path):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")

        print(f"CUDA: {torch.version.cuda}")

    else:
        print("\n💻 Using CPU")

    # --------------------------------------------------------
    # PDF information
    # --------------------------------------------------------

    total_pages = get_pdf_page_count(pdf_path)

    print(f"\n📄 PDF: {pdf_path}")
    print(f"📑 Pages: {total_pages}")

    # --------------------------------------------------------
    # User preferences
    # --------------------------------------------------------

    preferences = get_user_preferences()

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embeddings = get_embeddings(
        english=preferences["english"],
        device=device,
    )

    # --------------------------------------------------------
    # Text splitter
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # --------------------------------------------------------
    # Determine loading strategy
    # --------------------------------------------------------

    batch_size = get_batch_size(total_pages)

    # ========================================================
    # SMALL PDF
    # ========================================================

    if batch_size is None:
        print("\n📦 Strategy: Load entire PDF")

        pages = list(get_pdf_pages(pdf_path))

        chunks = splitter.split_documents(pages)

        print(f"Pages loaded: {len(pages)}")

        print(f"Chunks created: {len(chunks)}")

        create_vector_store(
            chunks=chunks,
            embeddings=embeddings,
        )

        print("\n✅ Indexing complete!")

        return

    # ========================================================
    # LARGE PDF
    # ========================================================

    print(f"\n📦 Strategy: Lazy load → {batch_size} pages/batch")

    vector_store = None

    page_batch = []

    processed_pages = 0

    # --------------------------------------------------------
    # Lazy loading
    # --------------------------------------------------------

    for page in get_pdf_pages(pdf_path):
        page_batch.append(page)

        if len(page_batch) == batch_size:
            chunks = splitter.split_documents(page_batch)

            if vector_store is None:
                vector_store = create_vector_store(
                    chunks=chunks,
                    embeddings=embeddings,
                )

            else:
                vector_store.add_documents(chunks)

            processed_pages += len(page_batch)

            print(
                f"Processed "
                f"{processed_pages}/{total_pages} pages "
                f"→ {len(chunks)} chunks"
            )

            page_batch.clear()

    # ========================================================
    # REMAINING PAGES
    # ========================================================

    if page_batch:
        chunks = splitter.split_documents(page_batch)

        if vector_store is None:
            vector_store = create_vector_store(
                chunks=chunks,
                embeddings=embeddings,
            )

        else:
            vector_store.add_documents(chunks)

        processed_pages += len(page_batch)

        print(f"Processed {processed_pages}/{total_pages} pages → {len(chunks)} chunks")

    print("\n✅ Indexing complete!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    store_qdrant(DOC_PATH)
