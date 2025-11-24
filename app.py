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

from groq import Groq
import re

st.set_page_config(layout="wide", page_title="Autonomous QA Agent")


def is_valid_test_case(tc):
    test_id = tc.get("Test_ID", "")
    pattern = r"^TC-\d+$"
    return bool(re.match(pattern, test_id))

@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


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
    client = chromadb.PersistentClient(path="./chroma_db")
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

    pasted_html = st.text_area("Paste checkout.html content here", height=150)

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
        if not uploaded_docs and pasted_html.strip() == "":
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

            if pasted_html.strip():
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

            collection = client.get_or_create_collection(name=collection_name, embedding_function=None)


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
    

st.title("Phase 2: Autonomous QA Agent")

with st.expander("Generate Test Cases with QA Agent"):
    user_query = st.text_area(
        "Ask the QA Agent (e.g., 'Generate positive and negative test cases for discount code feature')",
        height=120
    )

    top_k = st.number_input("Number of knowledge chunks to retrieve", 1, 10, 4)

    if st.button("Generate Test Cases"):
        if "kb_collection" not in st.session_state:
            st.error("Please build the Knowledge Base in Phase 1 first.")
        else:
            client = get_chroma_client()
            collection = client.get_collection(name=st.session_state["kb_collection"])

            model = get_embedding_model()

            q_emb = model.encode([user_query], convert_to_numpy=True)[0]

            results = collection.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k
            )

            retrieved_docs = results["documents"][0]
            retrieved_meta = results["metadatas"][0]

            context_blocks = []
            for i, text in enumerate(retrieved_docs):
                src = retrieved_meta[i].get("source", "unknown")
                context_blocks.append(
                    f"---\nSource: {src}\nChunk:\n{text}\n"
                )
            context = "\n".join(context_blocks)

            rag_prompt = f"""
            You are an Autonomous QA Agent.  
            Your job is to generate structured software test cases grounded ONLY in the provided context.

            ### CONTEXT (only use this information)
            {context}

            ### USER REQUEST
            {user_query}

            ### REQUIREMENTS
            1. Output MUST be in JSON list format or Markdown tables.
            2. Each test case MUST include:
            - Test_ID (must follow format TC-XXX, e.g., TC-001, TC-002) 
            - Feature  
            - Test_Scenario  
            - Expected_Result  
            - Grounded_In (source document name)
            3. Only use facts from the context—do not hallucinate.
            4. Include both positive and negative test cases if the user asks.

            ### NOW GENERATE TEST CASES:
            If the context does NOT contain enough information, reply:
            "Insufficient context to generate test cases."
            """
            
            client_g = get_groq_client()
            completion = client_g.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": rag_prompt}]
            )
            answer = completion.choices[0].message.content.strip()

            st.markdown(answer)
            def parse_markdown_table(md_text):
                lines = md_text.splitlines()
                lines = [l.strip() for l in lines if l.strip()]
                
                table_lines = [l for l in lines if "|" in l]
                if len(table_lines) < 2:
                    return None

                header = [h.strip() for h in table_lines[0].split("|") if h.strip()]
                rows = table_lines[2:]  

                parsed = []
                for row in rows:
                    cols = [c.strip() for c in row.split("|") if c.strip()]
                    if len(cols) == len(header):
                        parsed.append(dict(zip(header, cols)))

                return parsed

            parsed_cases_list = None

            try:
                st.session_state["generated_test_cases"] = json.loads(answer)
            except:
                md_cases = parse_markdown_table(answer)
                if md_cases:
                    st.session_state["generated_test_cases"] = md_cases
                else:
                    st.error("Output is neither valid JSON nor Markdown table.") 



st.title("Phase 3: Selenium Script Generation Agent")

if "script" in st.session_state:
    st.subheader("Generated Selenium Script")
    st.code(st.session_state["script"], language="python")

if "generated_test_cases" not in st.session_state:
    st.info("Generate test cases in Phase 2 first.")
else:
    # Get raw test cases
    test_cases_raw = st.session_state.get("generated_test_cases", [])

    # Normalize and filter valid Test_IDs
    valid_test_cases = []
    for tc in test_cases_raw:
        # Normalize keys by removing spaces
        tc_norm = {k.replace(" ", "_").strip(): v for k, v in tc.items()}

        test_id = tc_norm.get("Test_ID") or tc_norm.get("TestId") or tc_norm.get("Test_Id")
        if test_id and re.match(r"^TC-\d+$", str(test_id).strip()):
            tc_norm["Test_ID"] = str(test_id).strip()
            valid_test_cases.append(tc_norm)

    if not valid_test_cases:
        st.write("No valid test cases available.")
    else:
        # Display as JSON
        st.subheader("Previously Generated Test Cases (JSON)")
        st.json(valid_test_cases)

        # Selectbox from valid test cases
        select_options = [
            f"{tc['Test_ID']} - {tc.get('Test_Scenario', '')}" for tc in valid_test_cases
        ]
        selected_case = st.selectbox("Select a Test Case", options=select_options)

        if st.button("Generate Selenium Script"):
            chosen = next(
                tc for tc in valid_test_cases
                if f"{tc['Test_ID']} - {tc.get('Test_Scenario', '')}" == selected_case
            )

            client = get_chroma_client()
            collection = client.get_collection(name=st.session_state["kb_collection"])
            model = get_embedding_model()

            q_emb = model.encode([chosen["Feature"]], convert_to_numpy=True)[0]
            results = collection.query(query_embeddings=[q_emb.tolist()], n_results=3)
            context_docs = "\n\n".join(results["documents"][0])

            html_content = st.session_state["html_text"]
            html_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkout.html")
            test_case_json_str = json.dumps(chosen, indent=2)
            escaped_json_str = json.dumps(test_case_json_str)


            selenium_prompt = f"""
You are an expert Selenium QA Automation Engineer.
Generate a fully executable Python Selenium script that runs locally on Windows or Linux. Output ONLY valid Python code—no markdown, no explanations, no extra text.

CONTEXT:
TEST CASE JSON: {escaped_json_str}
Relevant docs/context: {context_docs}
checkout.html {pasted_html}

TASK REQUIREMENTS:
- Access checkout.html locally in the same directory as the script. The script must create the file using the content provided in the `checkout.html content` section and a constant variable.
- Use only the path:
This path is where the operations will be performed, the code in the file should not be changed
    html_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkout.html")
    file_url = f"file:///{html_file_path.replace(os.sep, '/')}"
    driver.get(file_url)
- Do NOT open any other URLs or files.
- Open Selenium Chrome driver (visible, not headless). Use only chromedriver_autoinstaller.
- Dynamically interact with the page according to the provided test case JSON:(in this order)
    - Add 1–3 products to cart or remove items (Logic must ensure removal doesn't exceed added quantity).
    - Apply discount code if present.
    - Generate test data dynamically:
        Name: "Test User <timestamp>"
        Email: "test<timestamp>@example.com"
        Address: "123 Test St <timestamp>"
    - Select shipping option if specified.
    - Select a payment method
- Compute expected total based on:
    - Added products
    - Discount applied
    - Shipping charges 
- Read displayed total and assert equality with expected total (round to 2 decimal places). Raise AssertionError if mismatch.
- Fail fast if any required element is missing (RuntimeError("Element not found: <description>")).
- Keep browser open for 300 seconds for inspection.
- Use only the IDs and class names present in checkout.html. Do not invent selectors.
- Selector priority: id → name → XPath. Use XPath for product buttons based on `data-product` attribute.
- All interactions must use WebDriverWait. Tiny sleeps (e.g., time.sleep(0.1)) allowed only after clicks.
- Use only the inline test case JSON provided. Do NOT read any external JSON file.

STRICT RULES:

Use only these imports:

import chromedriver_autoinstaller
chromedriver_autoinstaller.install()
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import json
import random
from datetime import datetime


options = Options()
driver = webdriver.Chrome(options=options)

OUTPUT:

- Fully working Python script executable locally.
- Dynamically handles all actions (add/remove products, discounts, shipping) based on TEST CASE JSON (Use the provided JSON string for logic, but generate a random instance for dynamic testing).
- Keeps browser open for inspection.
- Uses chromedriver_autoinstaller only.
- Works with no errors.
- Syntax errors must be avoided


"""


            client_g = get_groq_client()
            completion = client_g.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": selenium_prompt}]
            )
            script = completion.choices[0].message.content

            
            st.session_state["script"] = script


if "script" in st.session_state:
    st.subheader("Generated Selenium Script")
    st.code(st.session_state["script"], language="python")



