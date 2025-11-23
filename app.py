import streamlit as st

import os
import json
import tempfile
from typing import List
from bs4 import BeautifulSoup
import fitz 
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(layout="wide", page_title="Autonomous QA Agent")

def extract_text_from_pdf(path: str) -> str:
    text = []
    doc = fitz.open(path)
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def extract_text_from_html(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for s in soup(["script", "style"]):
        s.decompose()
    return soup.get_text(separator="\n")


def extract_text_from_json(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return json.dumps(obj, indent=2)


def read_file_to_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if name.endswith(".pdf"):
        return extract_text_from_pdf(tmp_path)
    elif name.endswith(".html") or name.endswith(".htm"):
        return extract_text_from_html(uploaded_file.getvalue().decode("utf-8"))
    elif name.endswith(".json"):
        return extract_text_from_json(tmp_path)
    else:
        return uploaded_file.getvalue().decode("utf-8")


@st.cache_resource
def get_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@st.cache_resource
def get_chroma_client():
    # Use the NEW Chroma PersistentClient (replaces deprecated Client)
    client = chromadb.PersistentClient(path="./chroma_db")  # directory where embeddings are stored
    return client


def embed_texts(model, texts: List[str]):
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


st.title("Phase 1: Knowledge Base Ingestion")

with st.expander("Upload Files"):
    uploaded_docs = st.file_uploader(
        "Upload support documents (md/txt/json/pdf). Hold CTRL to multi-select",
        accept_multiple_files=True
    )

    html_file = st.file_uploader("Upload checkout.html", type=["html", "htm"])

    pasted_html = st.text_area("OR paste checkout.html content here", height=150)

    col1, col2 = st.columns([1, 1])
    with col1:
        chunk_size = st.number_input(
            "Chunk size (chars)", min_value=20, max_value=2000,
            value=800, step=50
        )
    with col2:
        chunk_overlap = st.number_input(
            "Chunk overlap (chars)", min_value=0, max_value=400,
            value=100, step=10
        )

    if st.button("Build Knowledge Base"):
        if not uploaded_docs and not html_file and pasted_html.strip() == "":
            st.warning("Please upload at least one document or HTML file.")
        else:
            docs = []
            for f in uploaded_docs:
                try:
                    txt = read_file_to_text(f)
                    docs.append({"source": f.name, "text": txt})
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")

            html_text = ""
            html_name = None

            if html_file:
                html_text = read_file_to_text(html_file)
                html_name = html_file.name
                docs.append({"source": html_name, "text": html_text})
            elif pasted_html.strip():
                html_text = pasted_html
                html_name = "checkout_pasted.html"
                docs.append({"source": html_name, "text": html_text})

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = []
            for d in docs:
                raw_chunks = text_splitter.split_text(d["text"])
                for i, c in enumerate(raw_chunks):
                    chunks.append({
                        "id": f"{d['source']}_chunk_{i}",
                        "text": c,
                        "metadata": {"source": d["source"], "chunk": i}
                    })

            st.write(f"Generated {len(chunks)} chunks. Creating embeddings...")

            model = get_embedding_model()
            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(model, texts)

            client = get_chroma_client()
            collection_name = "qa_agent_collection"

            collection = client.get_or_create_collection(name=collection_name)

            ids = [c["id"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings.tolist()
            )

            st.success("Knowledge Base built successfully!")
            st.session_state["kb_collection"] = collection_name
            st.session_state["html_text"] = html_text
            st.session_state["html_name"] = html_name
            st.write("Stored in ChromaDB:", collection_name)
