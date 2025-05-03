import pytest
from langchain.docstore.document import Document
from sec_rag_pipeline import build_vectorstore, retrieve_relevant_docs

def test_retrieval_returns_list_of_documents():
    # Build a simple vector store from repetitive text
    text = "foo bar baz " * 100
    store = build_vectorstore(text)
    docs = retrieve_relevant_docs(store, "bar", k=3)

    # Should get back a list of Document objects
    assert isinstance(docs, list)
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)

def test_retrieval_documents_are_relevant():
    # Create text with distinct keywords in different chunks
    text = "apple " * 500 + "banana " * 500 + "cherry " * 500
    store = build_vectorstore(text)

    # Query for 'banana' and expect chunks containing 'banana'
    docs = retrieve_relevant_docs(store, "banana", k=2)
    assert len(docs) == 2
    for d in docs:
        assert "banana" in d.page_content.lower()

def test_retrieval_respects_k_parameter():
    text = "lorem ipsum dolor sit amet consectetur adipiscing elit"
    store = build_vectorstore(text)

    # Request more docs than chunks: k > possible chunks
    docs = retrieve_relevant_docs(store, "ipsum", k=10)
    # Should return at most the number of chunks available
    assert len(docs) <= 10
    assert all(isinstance(d, Document) for d in docs)