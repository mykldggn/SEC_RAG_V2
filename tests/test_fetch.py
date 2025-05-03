from sec_rag_pipeline import fetch_edgar_filing
def test_fetch(tmp_path):
    text = fetch_edgar_filing("0000320193", "10-K")
    assert "dummy EDGAR text" in text
