from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ingest import discover_pdfs
from .rag_pipeline import (
    answer_question,
    compare_documents,
)


# --------------------------
# Request Models
# --------------------------

class AskRequest(BaseModel):
    company: str
    model: str
    question: str
    history: Optional[List[Dict[str, Any]]] = None


class ContradictRequest(BaseModel):
    document1: str
    document2: str
    topic: str


# --------------------------
# FastAPI
# --------------------------

app = FastAPI(
    title="Document RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Health
# --------------------------

@app.get("/")
def root():
    return {"message": "Potens AI Assignment API"}


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------
# Available Manuals
# --------------------------

@app.get("/documents")
def get_documents():

    docs = {}

    pdfs = discover_pdfs()

    for pdf in pdfs:

        company = pdf.parent.name

        docs.setdefault(company, [])

        model = (
            pdf.stem
            .replace("_", " ")
            .replace("-", " ")
        )

        print(f"DEBUG -> {pdf.stem} ---> {model}")

        docs[company].append(model)

    print("RETURNING:", docs)

    return docs


# --------------------------
# ASK
# --------------------------

@app.post("/ask")
def ask(request: AskRequest):

    result = answer_question(
        question=request.question,
        company=request.company,
        model=request.model,
        chat_history=None,
    )

    return result


# --------------------------
# CONTRADICT
# --------------------------

@app.post("/contradict")
def contradict(request: ContradictRequest):

    try:

        result = compare_documents(
            topic=request.topic,
            document1=request.document1,
            document2=request.document2,
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------
# Run
# --------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )