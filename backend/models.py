from typing import Literal

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from .config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load (and cache) the local HuggingFace sentence-transformer model."""
    model_name = settings.EMBEDDING_MODEL_NAME
    return HuggingFaceEmbeddings(model_name=model_name)


def get_llm(which: Literal["primary", "fallback"] = "primary") -> ChatGroq:
    """Return configured Groq LLM client."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please configure it in your environment.")

    if which == "primary":
        model_name = settings.GROQ_PRIMARY_MODEL
    else:
        model_name = settings.GROQ_FALLBACK_MODEL

    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=model_name,
        temperature=0.2,
        max_tokens=None,
    )

