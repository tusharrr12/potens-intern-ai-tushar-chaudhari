from langchain_community.vectorstores import Chroma

from backend.config import settings
from backend.models import get_embeddings

def get_vector_store():
    """
    Create or connect to the persistent ChromaDB collection.
    """

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="bike_manuals",
        embedding_function=embeddings,
        persist_directory=str(settings.VECTOR_DB_DIR),
    )

    return vector_store

def reset_vector_store():
    """
    Delete the existing Chroma collection so a fresh
    index can be created.
    """

    vector_store = get_vector_store()

    try:
        vector_store.delete_collection()
        print("[INFO] Existing vector collection deleted.")
    except Exception:
        print("[INFO] No existing collection found.")

def store_chunks(chunks):
    """
    Store all document chunks in ChromaDB.
    """

    vector_store = get_vector_store()

    texts = []
    metadatas = []

    for chunk in chunks:

        texts.append(chunk["text"])

        metadatas.append(chunk["metadata"])

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
    )

    vector_store.persist()

    print(f"[INFO] Stored {len(texts)} chunks in ChromaDB.")