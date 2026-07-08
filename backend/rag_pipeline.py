from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma

from .config import settings
from .models import get_embeddings, get_llm


def get_vectorstore() -> Chroma:
    """Connect to persistent ChromaDB."""

    embeddings = get_embeddings()

    return Chroma(
        collection_name="bike_manuals",
        embedding_function=embeddings,
        persist_directory=str(settings.VECTOR_DB_DIR),
    )


def build_prompt() -> ChatPromptTemplate:
    system_template = """
You are an AI Document Assistant.

Rules:

1. Use ONLY the retrieved context.

2. Never use outside knowledge.

3. If the answer is not found, reply:

"I couldn't find this information in the uploaded documents."

4. Detect the language of the user's question.

5. Answer in the SAME language as the user.

6. The retrieved documents may be in English.
Translate the final answer into the user's language if needed.

7. Preserve all technical values exactly:
- Numbers
- Units
- Page numbers
- Chunk IDs
- File names

8. Never hallucinate.

9. Always include citations.
"""

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            (
                "human",
                """
                User Question:

                {question}

                Retrieved Context:

                {context}

                Answer in the SAME language as the question.
                """
            ),
        ]
    )


def format_docs(docs: List[Any]) -> str:
    """Prepare retrieved documents for the LLM."""

    context = []

    for doc in docs:
        meta = doc.metadata

        context.append(
            f"""
Source File : {meta.get("source_file")}
Company     : {meta.get("company")}
Model       : {meta.get("model")}
Page        : {meta.get("page")}
Chunk ID    : {meta.get("chunk_id")}

Content:
{doc.page_content}
"""
        )

    return "\n\n-----------------------------\n\n".join(context)


def collect_sources(docs: List[Any]) -> List[Dict]:
    """Return structured citations."""

    sources = []

    for doc in docs:
        meta = doc.metadata

        sources.append(
            {
                "source_file": meta.get("source_file"),
                "company": meta.get("company"),
                "model": meta.get("model"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "snippet": doc.page_content[:200] + "...",
            }
        )

    return sources


def answer_question(
    question: str,
    company: str,
    model: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> Dict[str, Any]:

    vector_store = get_vectorstore()

    raw_results = vector_store.similarity_search_with_score(
        question,
        k=settings.TOP_K * 5,
    )

    docs = []
    best_score = float("inf")

    for doc, score in raw_results:

        if score < best_score:
            best_score = score

        meta = doc.metadata

        if (
            meta.get("company", "").lower() == company.lower()
            and meta.get("model", "").lower() == model.lower()
        ):
            docs.append(doc)

        if len(docs) >= settings.TOP_K:
            break

    if not docs:
        return {
            "answer": "I couldn't find this information in the uploaded documents.",
            "sources": [],
            "confidence": 0.0,
            "model_used": None,
            "from_context": False,
        }

    confidence = max(0.0, min(1.0, 1 - best_score / 2))

    # Extra hallucination guard
    if confidence < 0.35:
        return {
            "answer": "I couldn't find enough relevant information in the uploaded documents to answer confidently.",
            "sources": collect_sources(docs),
            "confidence": round(confidence, 2),
            "model_used": None,
            "from_context": False,
        }

    context = format_docs(docs)

    prompt = build_prompt()

    messages = prompt.format_messages(
        question=question,
        context=context,
        chat_history=chat_history or [],
    )

    try:

        llm = get_llm("primary")

        response = llm.invoke(messages)

        answer = response.content

        model_used = "primary"

    except Exception:

        llm = get_llm("fallback")

        response = llm.invoke(messages)

        answer = response.content

        model_used = "fallback"

    return {
        "answer": answer.strip(),
        "sources": collect_sources(docs),
        "confidence": round(confidence, 2),
        "model_used": model_used,
        "from_context": True,
    }

def compare_documents(
    topic: str,
    document1: str,
    document2: str,
):
    """
    Compare two documents on a given topic.
    """

    vector_store = get_vectorstore()

    # Retrieve relevant chunks
    results = vector_store.similarity_search(topic, k=20)

    doc1_chunks = []
    doc2_chunks = []

    for doc in results:

        model = doc.metadata.get("model", "").lower()

        if model == document1.lower():
            doc1_chunks.append(doc)

        if model == document2.lower():
            doc2_chunks.append(doc)

    if not doc1_chunks:
        return {
            "error": f"No evidence found for {document1}"
        }

    if not doc2_chunks:
        return {
            "error": f"No evidence found for {document2}"
        }

    evidence1 = "\n\n".join(
        chunk.page_content for chunk in doc1_chunks[:3]
    )

    evidence2 = "\n\n".join(
        chunk.page_content for chunk in doc2_chunks[:3]
    )

    prompt = f"""
You are comparing two technical manuals.

Topic:
{topic}

------------------------

Document 1 ({document1})

{evidence1}

------------------------

Document 2 ({document2})

{evidence2}

------------------------

Answer ONLY in JSON.

{{
    "conflict": true/false,
    "reason": "...",
    "document1_summary": "...",
    "document2_summary": "..."
}}

Do not invent information.
Use only the provided evidence.
"""

    llm = get_llm("primary")

    response = llm.invoke(prompt)

    return {
        "comparison": response.content,
        "document1_evidence": evidence1[:500],
        "document2_evidence": evidence2[:500],
    }