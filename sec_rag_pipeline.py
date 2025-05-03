import os
import tempfile
import json
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

import openai
from openai import OpenAIError

from dotenv import load_dotenv
from pathlib import Path

# LangChain community imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.llms import OpenAI as CommunityOpenAI

# Load environment variables
load_dotenv(dotenv_path=".env")

# Configuration: set your OpenAI API key via environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY env var is required")

# SEC Downloader credentials
SEC_EDGAR_COMPANY_NAME = os.getenv('SEC_EDGAR_COMPANY_NAME')
SEC_EDGAR_EMAIL_ADDRESS = os.getenv('SEC_EDGAR_EMAIL_ADDRESS')
if not SEC_EDGAR_COMPANY_NAME or not SEC_EDGAR_EMAIL_ADDRESS:
    raise RuntimeError("SEC_EDGAR_COMPANY_NAME and SEC_EDGAR_EMAIL_ADDRESS env vars are required for EDGAR Downloader")

# Default chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def fetch_edgar_filing(cik: str, form_type: str = "10-K") -> str:
    """
    Download plain text of the most recent EDGAR filing for a given CIK and form type.
    """
    from sec_edgar_downloader import Downloader

    tmpdir = tempfile.mkdtemp()
    dl = Downloader(SEC_EDGAR_COMPANY_NAME, SEC_EDGAR_EMAIL_ADDRESS, tmpdir)
    dl.get(form_type, cik, download_details=False)
    files = list(Path(tmpdir).rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No filings found for CIK {cik} and type {form_type}")
    return files[0].read_text(encoding='utf-8', errors='ignore')


def build_vectorstore(text: str) -> Chroma:
    """
    Chunk the text and build a Chroma vectorstore using OpenAI embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in texts]
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(docs, embeddings)


def retrieve_relevant_docs(store: Chroma, query: str, k: int = 5) -> List[Document]:
    """
    Retrieve top-k relevant chunks for the query.
    """
    return store.similarity_search(query, k=k)


def answer_query(query: str, docs: List[Document]) -> str:
    """
    Use OpenAI LLM to generate an answer based on retrieved document chunks.
    Output should be pure JSON matching the requested schema.
    """
    llm = CommunityOpenAI(temperature=0)
    # Combine docs into context
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        f"You are a specialized SEC filings assistant. "
        f"Use the following document excerpts to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Please answer in valid JSON array format only."
    )
    return llm(prompt)


def extract_structured(answer: str, features: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Parse the LLM-generated answer (in JSON) into a DataFrame with specified schema.
    """
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        raise ValueError("LLM output is not valid JSON: " + answer)

    df = pd.json_normalize(data)
    for col, spec in features.items():
        if spec.get('required') and df.get(col).isnull().any():
            raise ValueError(f"Required column {col} contains nulls")
        dtype = spec.get('type')
        if dtype == 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif dtype == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype == 'enum':
            # Do not enforce enum validation; accept LLM-provided values
            df[col] = df[col].astype(str)
    return df


def run_rag(cik: str, query: str, features: Dict[str, Dict[str, Any]], form_type: str = "10-K", k: int = 5) -> pd.DataFrame:
    """
    End-to-end RAG: fetch the latest SEC filing, build vectorstore,
    retrieve docs, answer the query, and extract structured DataFrame.
    """
    raw = fetch_edgar_filing(cik, form_type)
    store = build_vectorstore(raw)
    docs = retrieve_relevant_docs(store, query, k=k)

    # Build JSON schema prompt fragment
    schema_parts = [f'"{col}": {spec["description"]}' for col, spec in features.items()]
    # Ask the model to output JSON matching schema
    answer = answer_query(
        query + "\nSchema: {" + ", ".join(schema_parts) + "}",
        docs
    )
    return extract_structured(answer, features)
