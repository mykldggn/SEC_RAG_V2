# SEC RAG Pipeline

A Retrieval‑Augmented‑Generation (RAG) backend for extracting structured data from SEC EDGAR filings using OpenAI and LangChain.

---

## 📝 Description

This service fetches the latest SEC filing (e.g., 10‑K), splits it into chunks, embeds and indexes with a vector store, retrieves relevant passages for your question, and uses an LLM to extract structured JSON which is normalized into a pandas `DataFrame`.

---

## 🚀 Prerequisites

* Python 3.9+
* A valid OpenAI API key
* (Optional) A GitHub account for version control

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/mykldggn/SEC_RAG_V2.git
   cd SEC_RAG_V2
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🔑 Environment Variables

Create a file named `.env` in the project root with the following keys:

```ini
OPENAI_API_KEY=sk-...                  # Your OpenAI secret key
SEC_EDGAR_COMPANY_NAME=YourOrgName     # Any identifier for EDGAR downloader
SEC_EDGAR_EMAIL_ADDRESS=you@domain.com # A real email address
```

> **Note:** Add `.env` to your `.gitignore` to keep secrets out of source control.

---

## 📥 Input Specification

Your main entry point is the function:

```python
df = run_rag(
    cik: str,
    query: str,
    features: Dict[str, Dict[str, Any]],
    form_type: str = "10-K",
    k: int = 5
) -> pandas.DataFrame
```

### `features` schema

* **Type:** `Dict[str, Dict[str, Any]]`
* **Keys:** column names you want in the output DataFrame.
* **Values:** a dictionary with:

  * `description` (`str`): human‑readable description of the field.
  * `type` (`str`): one of `"string"`, `"datetime64[ns]`, `"float"`, or `"enum"`.
  * `enum` (`List[str]`, optional): allowed values (only for `"enum"` types).
  * `required` (`bool`): whether the column must be non‑null.

**Example:**

```python
features = {
    "Date": {
        "description": "The date when the update was made",
        "type": "datetime64[ns]",
        "required": True
    },
    "Ticker": {
        "description": "The stock ticker",
        "type": "string",
        "required": True
    },
    "Direction": {
        "description": "Price target change",
        "type": "enum",
        "enum": ["raised", "lowered"],
        "required": True
    }
}
```

---

## 💡 Usage Example

Run a one‑liner in your shell (make sure `.env` is present):

```bash
python - <<'PYCODE'
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from sec_rag_pipeline import run_rag

# define features as above...

df = run_rag(
    cik="0000320193",
    query="Did any filing mention AAPL expanding into ASIA?",
    features=features
)
print(df)
PYCODE
```

Or drop it in `examples/run_example.py` and run:

```bash
python examples/run_example.py
```

---

## 🧪 Testing

All functions are covered by pytest fixtures. Run:

```bash
pytest --maxfail=1 --disable-warnings -q
```

---

## 📄 License

[MIT](LICENSE)

```
```

