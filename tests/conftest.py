import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch, tmp_path):
    # 1) Stub out the Downloader that fetch_edgar_filing uses
    class DummyDownloader:
        def get(self, form, cik, download_details, output_dir):
            # write a dummy file into the output_dir so fetch_edgar_filing sees it
            output_path = Path(output_dir) / cik / form / f"{cik}.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("Dummy EDGAR text about AAPL expanding into Asia.")

    # Replace sec_rag_pipeline.Downloader with our DummyDownloader factory
    monkeypatch.setattr(
        "sec_rag_pipeline.Downloader",
        lambda *args, **kwargs: DummyDownloader()
    )

    # 2) Stub out OpenAI so run_rag’s ChatCompletion never actually calls the API
    def fake_chat_create(*args, **kwargs):
        class Choice:
            message = {"content": '[{"Date":"2025-01-01","Ticker":"AAPL","Direction":"raised"}]'}
        return {"choices": [Choice()]}

    monkeypatch.setattr("openai.ChatCompletion", "create", fake_chat_create)

    yield