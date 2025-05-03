from sec_rag_pipeline import run_rag
def test_run_rag_end_to_end():
    features = {
      "Date": {"description":"…","type":"datetime64[ns]","required":True},
      "Ticker": {"description":"…","type":"string","required":True},
      "Direction": {"description":"…","type":"enum","enum":["raised","lowered"],"required":True}
    }
    df = run_rag("0000320193", "mention AAPL expanding into ASIA?", features)
    assert df.shape[0] == 1
    assert "AAPL" in df["Ticker"].values
