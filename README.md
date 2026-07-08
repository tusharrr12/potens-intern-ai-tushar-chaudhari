# Potens AI Internship Assignment – Document RAG Assistant

RAG system built for the Potens AI/ML Internship Assignment 2026 (Q1: Document Q&A with Citations).

The app answers questions over a set of technical manuals with source-backed citations, refuses to answer when the documents don't cover a question, and can compare two documents to flag contradictions on a given topic.

## What it does

- Ingests multiple PDF manuals and chunks them for retrieval
- Embeds chunks with SentenceTransformers and stores them in ChromaDB
- `/ask` — answers a question using only retrieved context, with citations
- `/contradict` — compares two documents on a topic and reports whether they conflict
- Returns an explicit fallback response when there is insufficient evidence in the retrieved documents
- Streamlit UI so it can be tried without Postman

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI |
| LLM | Groq (Llama 3.1) |
| Embeddings | Sentence Transformers |
| Vector DB | ChromaDB |
| Frontend | Streamlit |
| PDF parsing | PyMuPDF (fitz) |

## Architecture

```
PDF Manuals → Text Extraction → Recursive Character Splitter → Sentence Embeddings → ChromaDB
                                                                          │
                                                        ┌─────────────────┴─────────────────┐
                                                        ▼                                   ▼
                                                    /ask API                          /contradict API
                                                        │                                   │
                                                        └───────────────┬───────────────────┘
                                                                        ▼
                                                                Groq Llama 3.1
                                                                        │
                                                                        ▼
                                                              Streamlit Frontend
```

## Folder Structure

```
backend/
    config.py
    ingest.py
    main.py
    models.py
    rag_pipeline.py
    services/
frontend/
    app.py
data/
    manuals/
vectordb/
```

## Chunking Strategy

Using LangChain's `RecursiveCharacterTextSplitter`:

- Chunk size: 1000 characters
- Chunk overlap: 150 characters

These numbers were picked empirically — small enough that citations point to a specific, readable snippet, large enough that a chunk doesn't get cut off mid-procedure (manuals have a lot of multi-sentence steps, e.g. torque specs followed by a caveat in the next sentence). 150-character overlap was enough to stop a sentence from being split across two chunks in testing, without bloating the vector store much.

## Retrieval Flow

1. User submits a question.
2. Question is embedded and ChromaDB returns the top-K most relevant chunks.
3. Chunks are formatted with their source metadata (file, page, chunk ID).
4. Before generation, retrieved chunks are filtered using document metadata (company and model) to ensure answers come only from the selected manual.
5. Groq Llama generates an answer using *only* the retrieved chunks.
6. Response is returned with citations and a confidence score.

## API Endpoints

### `GET /health`
Health check.

### `GET /documents`
Returns available manuals.

### `POST /ask`
```json
{
  "company": "bajaj",
  "model": "Dominar 400",
  "question": "What engine oil is recommended?"
}
```
Returns a grounded answer with citations (source file, page number, chunk ID, supporting snippet) and a confidence score.

### `POST /contradict`
```json
{
  "document1": "Dominar 400",
  "document2": "TVS Apache RTR 310",
  "topic": "Engine oil capacity"
}
```
The endpoint retrieves evidence from both documents, then prompts the LLM to compare the retrieved passages and determine whether they contain conflicting information.

## Multilingual Support (Partial)

The app accepts queries in languages other than English. However, retrieval currently relies on English-centric document embeddings, so non-English queries may retrieve less relevant context, or fail to retrieve the right chunks at all. The LLM itself is capable of generating a multilingual response when it's given relevant context — but the end-to-end multilingual pipeline (query → correct retrieval → answer in the same language) is not fully implemented yet.

This is a known gap, not a hidden one: the brief allows a translation-at-the-boundary approach for the 24-hour version, and that's the direction I'd take next (see below) rather than switching to a multilingual embedding model outright, since that would mean re-embedding the entire corpus.

## Hallucination Prevention

The system does not answer outside retrieved context. If there isn't enough evidence, it explicitly returns:

> "I couldn't find this information in the uploaded documents."

A heuristic confidence score is computed from retrieval similarity before response generation. It is intended as an indicator of retrieval quality rather than a calibrated probability, and is used as a gate for this fallback.

## What's Broken / Unfinished

- **Multilingual retrieval** is only partially done — see above. Right now a non-English query will often retrieve the wrong (or no) chunks, even though the LLM could answer correctly if it had the right context.
- **PDF-only.** No OCR, so scanned/image-based manuals aren't supported.
- **No reranking.** Retrieval is single-pass dense search; on larger document sets this will start missing relevant chunks that don't rank in the initial top-K.
- **No automated eval set.** Retrieval and answer quality have been checked manually, not against a fixed set of Q&A pairs with ground truth.
- **Contradiction detection is only as good as retrieval** — if the relevant chunks on a topic aren't retrieved from both documents, `/contradict` can miss a real conflict rather than falsely report one.

## What I'd Build Next

1. Add a translation-at-the-boundary step (detect query language → translate to English for retrieval → translate the answer back) to close the multilingual gap without re-embedding the corpus.
2. Add a cross-encoder reranker on top of the initial dense retrieval.
3. Build a small eval set (~10 Q&A pairs with ground truth) and score retrieval at top-k, so chunking/retrieval changes can be measured instead of eyeballed.
4. OCR support for scanned manuals.
5. Docker Compose for one-command setup.

## How to Run

Python 3.11+ recommended.

```bash
git clone https://github.com/tusharrr12/potens-intern-ai-tushar-chaudhari.git
cd potens-intern-ai-tushar-chaudhari

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_PRIMARY_MODEL=llama-3.1-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
```

### Build the Vector Database and Run

```bash
python -m backend.ingest       # parses the PDF manuals and builds the Chroma vector database
python -m uvicorn backend.main:app --reload --port 8000   # backend
python -m streamlit run frontend/app.py                    # frontend
```

**Tested on:** Windows 11, Python 3.12

## Design Decisions

- **FastAPI** for a lightweight, typed REST layer.
- **ChromaDB** for local vector storage — no external service to stand up for a 24-hour build.
- **RecursiveCharacterTextSplitter** over naive fixed-size splitting, to reduce mid-sentence/mid-step cuts in the manuals.
- **SentenceTransformers** for embeddings — fast enough to run locally without a GPU.
- **Groq Llama 3.1** for generation — low latency, free tier, good enough for grounded Q&A over short contexts.
- **Streamlit** for the UI — fastest way to get something clickable without building a frontend from scratch.
- **ChromaDB stores embeddings locally on disk**, allowing the vector database to persist across application restarts without requiring re-ingestion.

## AI Use Log

| Tool | Approx. Usage | Purpose |
|---|---|---|
| ChatGPT | ~300–400 messages (~200k–300k tokens, estimated) | Architecture planning, debugging, RAG pipeline design, FastAPI & Streamlit integration, prompt engineering, code review, README drafting |
| Groq (Llama 3.1) | ~300–500 API requests (~80k–150k tokens, estimated) | Answer generation, RAG inference, contradiction comparison, testing |
| GitHub Copilot | Minimal — boilerplate completions only | Code completion for repetitive boilerplate |

AI tools were used to accelerate development — architecture discussions, debugging, prompt refinement, and documentation. All implementation decisions, integration, testing, and validation were done manually. Usage counts above are approximate estimates, as exact message/token counts weren't tracked during development.


## Assignment Coverage

| Requirement | Status |
|-------------|--------|
| Document ingestion | ✅ |
| Vector database | ✅ |
| Question answering | ✅ |
| Source citations | ✅ |
| Contradiction detection | ✅ |
| Streamlit UI | ✅ |
| Hallucination fallback | ✅ |
| Confidence score (bonus) | ✅ |
| Multilingual support | ⚠️ Partial |

## Author

Tushar Chaudhari — Potens Internship Assignment 2026
