def test_vectorstore_build_and_search():
    from sec_rag_pipeline import build_vectorstore, retrieve_relevant_docs
    text = "foo bar baz " * 500
    store = build_vectorstore(text)
    docs = retrieve_relevant_docs(store, "bar", k=2)
    assert len(docs) == 2
    assert all("bar" in d.page_content for d in docs)
