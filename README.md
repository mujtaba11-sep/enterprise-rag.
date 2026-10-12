# Enterprise RAG — Local Hackathon Prototype

A local-first document question-answering prototype with a clean interface, subtle animations, hybrid search, and source-page citations. No paid API keys.

## Features
- Upload searchable PDF, TXT, and Markdown files
- Extract page-aware text and chunk it with overlap
- Local embeddings with Sentence Transformers
- FAISS semantic vector search
- BM25 keyword search
- Reciprocal Rank Fusion hybrid retrieval
- Evidence passages with document names and page numbers
- Optional local answer generation with Ollama
- Subtle entrance and hover transitions, with reduced-motion support

## Windows setup (PowerShell)
Open PowerShell in the extracted project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

This avoids PowerShell's script activation policy. The app usually opens at `http://localhost:8501`.

## Windows alternative (CMD)
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

## Optional local LLM
The app works without Ollama and returns evidence passages. For natural-language answers generated locally:
1. Install Ollama from https://ollama.com
2. Run `ollama pull llama3.2`
3. Ensure Ollama is running.
4. Enable **Generate answer with local Ollama** in the sidebar.

No paid API key is required. First run needs internet to install packages and download the embedding model. After downloads, core retrieval runs locally.

## Demo flow
1. Upload several policy PDFs.
2. Click **Process documents and build index**.
3. Ask a question covered by the files.
4. Expand source evidence to show the exact passage, filename, and page.
5. Ask something absent from the files to demonstrate insufficient-evidence handling.

## Limitations
- Scanned PDFs need OCR; not included.
- Each indexing run replaces the existing index. Select the full collection when rebuilding.
- Thousands of files require benchmarking and may need significant RAM, disk, and indexing time.
- Retrieval score is a heuristic, not a calibrated confidence score.
- Citation presence does not itself guarantee factual correctness.
- No authentication, access control, encryption at rest, audit logs, or tenant isolation.
- Not production-ready for confidential enterprise data.
