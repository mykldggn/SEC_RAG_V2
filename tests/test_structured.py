import pandas as pd
from sec_rag_pipeline import extract_structured
def test_extract_valid():
    ans = '[{"Date":"2025-01-01","Ticker":"AAPL","Direction":"raised"}]'
    features = {
      "Date": {"description":"…","type":"datetime64[ns]","required":True},
      "Ticker": {"description":"…","type":"string","required":True},
      "Direction": {"description":"…","type":"enum","enum":["raised","lowered"],"required":True}
    }
    df = extract_structured(ans, features)
    assert isinstance(df, pd.DataFrame)
    assert df["Direction"].iloc[0] == "raised"
