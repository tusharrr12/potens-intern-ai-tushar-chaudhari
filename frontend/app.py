import json
from typing import Dict, List

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"


@st.cache_data
def fetch_documents() -> Dict[str, List[str]]:
    response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Potens Document RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Potens Document RAG Assistant")
st.write(
    "Ask questions from your uploaded manuals. "
    "Answers are generated using Retrieval-Augmented Generation (RAG) with citations."
)

try:
    documents = fetch_documents()
except Exception as e:
    st.error(f"Unable to connect to backend.\n\n{e}")
    st.stop()

# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("📂 Select Document")

    companies = sorted(documents.keys())

    company = st.selectbox(
        "Company",
        companies,
    )

    model = st.selectbox(
        "Model",
        sorted(documents[company]),
    )

    st.divider()

    st.caption(f"Backend : {BACKEND_URL}")

# ---------------- Chat History ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- User Question ----------------

question = st.chat_input("Ask anything from the manual...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    payload = {
        "company": company,
        "model": model,
        "question": question,
        "history": [],
    }

    with st.chat_message("assistant"):

        with st.spinner("Searching manual..."):

            try:

                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json=payload,
                    timeout=60,
                )

                response.raise_for_status()

                data = response.json()

            except Exception as e:

                st.error(f"Backend Error\n\n{e}")
                st.stop()

            answer = data.get("answer", "")

            confidence = data.get("confidence", 0)

            model_used = data.get("model_used", "-")

            sources = data.get("sources", [])

            st.markdown(answer)

            st.progress(float(confidence))

            st.caption(f"Confidence Score : {confidence:.2f}")

            st.caption(f"LLM : {model_used}")

            with st.expander("📄 Source Citations", expanded=False):

                if not sources:
                    st.info("No citations available.")

                for i, src in enumerate(sources, start=1):

                    st.markdown(f"### Citation {i}")

                    st.write(f"**Source File:** {src.get('source_file','-')}")
                    st.write(f"**Company:** {src.get('company','-')}")
                    st.write(f"**Model:** {src.get('model','-')}")
                    st.write(f"**Page:** {src.get('page','-')}")
                    st.write(f"**Chunk ID:** {src.get('chunk_id','-')}")

                    st.write("**Snippet:**")

                    st.code(
                        src.get("snippet", ""),
                        language="text",
                    )

                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )