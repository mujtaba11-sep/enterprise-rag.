import os, re, json
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import fitz
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_FILE = INDEX_DIR / "chunks.json"
FAISS_FILE = INDEX_DIR / "vectors.faiss"
META_FILE = INDEX_DIR / "index_meta.json"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="Enterprise RAG", page_icon="📚", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1180px; padding-top:1.7rem; padding-bottom:3rem;}
.rag-hero {padding:1.65rem 1.8rem; border:1px solid #e5eaf2; border-radius:18px;
background:linear-gradient(115deg,#f1f5ff 0%,#fff 58%,#f0fbf6 100%); animation:enter .4s ease-out both; margin-bottom:1.15rem;}
.rag-kicker {font-size:.76rem; letter-spacing:.13em; text-transform:uppercase; font-weight:750; color:#3659d9;}
.rag-heading {font-size:clamp(1.8rem,4vw,2.65rem); line-height:1.12; letter-spacing:-.04em; font-weight:800; color:#172033; margin:.5rem 0 .65rem;}
.rag-description {color:#667085; max-width:740px; font-size:1rem;}
.rag-tags {display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1rem;}
.rag-tag {background:#fff; border:1px solid #e1e7f0; border-radius:999px; padding:.32rem .65rem; font-size:.78rem; color:#344054;}
div.stButton > button {border-radius:10px; transition:transform .16s ease,box-shadow .16s ease;}
div.stButton > button:hover {transform:translateY(-1px); box-shadow:0 5px 15px rgba(23,32,51,.1);}
[data-testid="stExpander"] {border-radius:12px; border-color:#e3e8f0;}
[data-testid="stMetric"] {border:1px solid #e3e8f0; border-radius:12px; padding:.75rem 1rem;}
@keyframes enter {from {opacity:0; transform:translateY(7px)} to {opacity:1; transform:translateY(0)}}
@media (prefers-reduced-motion:reduce) {* {animation-duration:.01ms!important; transition-duration:.01ms!important;}}
</style>
<div class="rag-hero">
<div class="rag-kicker">Evidence-first AI · Local document search</div>
<div class="rag-heading">Find answers inside your documents.</div>
<div class="rag-description">Search policies, manuals, and internal PDFs in plain language. Review source passages and page references before trusting an answer.</div>
<div class="rag-tags"><span class="rag-tag">Hybrid search</span><span class="rag-tag">Page-level citations</span><span class="rag-tag">Local-first</span><span class="rag-tag">No paid API key</span></div>
</div>
""", unsafe_allow_html=True)

def normalize(text: str) -> str:
    text = text.replace("\\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pages(uploaded_file) -> List[Dict[str, Any]]:
    name = Path(uploaded_file.name).name
    raw = uploaded_file.getvalue()
    pages = []
    if Path(name).suffix.lower() == ".pdf":
        try:
            doc = fitz.open(stream=raw, filetype="pdf")
            for number, page in enumerate(doc, 1):
                text = normalize(page.get_text("text"))
                if text:
                    pages.append({"source": name, "page": number, "text": text})
            doc.close()
        except Exception as exc:
            st.warning(f"Could not read {name}: {exc}")
    else:
        text = normalize(raw.decode("utf-8", errors="ignore"))
        if text:
            pages.append({"source": name, "page": 1, "text": text})
    return pages

def make_chunks(pages, chunk_words=220, overlap=40):
    chunks = []
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            snippet = " ".join(words[start:end]).strip()
            if snippet:
                chunks.append({"source": page["source"], "page": page["page"], "text": snippet, "chunk_id": len(chunks)})
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
    return chunks

@st.cache_resource(show_spinner="Loading local embedding model...")
def get_embedder():
    return SentenceTransformer(MODEL_NAME)

def build_index(chunks, embedder):
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(FAISS_FILE))
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    META_FILE.write_text(json.dumps({"embedding_model": MODEL_NAME, "chunk_count": len(chunks), "vector_dim": int(vectors.shape[1])}, indent=2), encoding="utf-8")

def load_index():
    if not (FAISS_FILE.exists() and CHUNKS_FILE.exists()):
        return None, None
    try:
        return json.loads(CHUNKS_FILE.read_text(encoding="utf-8")), faiss.read_index(str(FAISS_FILE))
    except Exception:
        return None, None

def tokens(text):
    return re.findall(r"\b[a-zA-Z0-9_'-]+\b", text.lower())

def retrieve(question, chunks, index, embedder, k):
    q = np.asarray(embedder.encode([question], normalize_embeddings=True), dtype=np.float32)
    candidate_count = min(max(k * 4, k), len(chunks))
    dense_scores, dense_ids = index.search(q, candidate_count)
    dense_rank = {int(idx): (rank, float(score)) for rank, (score, idx) in enumerate(zip(dense_scores[0], dense_ids[0])) if idx >= 0}
    bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
    raw_scores = bm25.get_scores(tokens(question))
    bm_ids = np.argsort(raw_scores)[::-1][:candidate_count]
    bm_rank = {int(idx): (rank, float(raw_scores[idx])) for rank, idx in enumerate(bm_ids)}
    candidates = set(dense_rank) | set(bm_rank)
    fused = {}
    for idx in candidates:
        fused[idx] = (1 / (61 + dense_rank[idx][0]) if idx in dense_rank else 0) + (1 / (61 + bm_rank[idx][0]) if idx in bm_rank else 0)
    ordered = sorted(fused, key=fused.get, reverse=True)[:k]
    max_score = max((fused[i] for i in ordered), default=0)
    result = []
    for idx in ordered:
        item = dict(chunks[idx])
        item["retrieval_score"] = fused[idx] / max_score if max_score else 0
        item["semantic_score"] = dense_rank.get(idx, (None, None))[1]
        item["keyword_score"] = bm_rank.get(idx, (None, 0))[1]
        result.append(item)
    return result

def ollama_answer(question, passages, model):
    import requests
    context = "\n\n".join(f"[Source: {p['source']}, page {p['page']}]\n{p['text']}" for p in passages)
    prompt = f"""Answer using ONLY the CONTEXT below. Do not use outside knowledge. Cite factual claims with [filename, p. X]. Never invent sources or page numbers. If evidence is insufficient, say: I could not find sufficient evidence in the indexed documents to answer this reliably.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""
    response = requests.post("http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()

def evidence_answer(passages):
    if not passages:
        return "I could not find sufficient evidence in the indexed documents to answer this reliably."
    lines = []
    for p in passages[:3]:
        excerpt = p["text"]
        if len(excerpt) > 650:
            excerpt = excerpt[:647].rsplit(" ", 1)[0] + "..."
        lines.append(f"- {excerpt} **[{p['source']}, p. {p['page']}]**")
    return "The strongest matching evidence in the indexed documents is:\n\n" + "\n\n".join(lines)

with st.sidebar:
    st.markdown("### Search settings")
    st.caption("Embedding model")
    st.code(MODEL_NAME)
    top_k = st.slider("Evidence passages", 3, 12, 6)
    threshold = st.slider("Minimum relative retrieval score", 0.0, 1.0, 0.12, 0.01)
    use_ollama = st.toggle("Generate answer with local Ollama", value=False)
    ollama_model = st.text_input("Ollama model", value="llama3.2", disabled=not use_ollama)
    st.caption("First run downloads the embedding model. No paid API key is required.")

tab_ask, tab_upload, tab_status = st.tabs(["⌕  Ask documents", "＋  Upload & index", "▤  Index status"])

with tab_upload:
    st.subheader("Build your knowledge base")
    st.write("Upload searchable PDFs, TXT, or Markdown files. Scanned/image-only PDFs need OCR, which this prototype does not include.")
    uploaded = st.file_uploader("Drop files here or browse", type=["pdf", "txt", "md"], accept_multiple_files=True)
    c1, c2 = st.columns(2)
    with c1:
        chunk_words = st.number_input("Words per chunk", min_value=100, max_value=500, value=220, step=20)
    with c2:
        overlap = st.number_input("Overlap words", min_value=0, max_value=100, value=40, step=10)
    if uploaded:
        st.caption(f"{len(uploaded)} file(s) selected.")
        for file in uploaded[:8]:
            st.write(f"📄 {file.name} · {file.size / 1024:.1f} KB")
        if len(uploaded) > 8:
            st.caption(f"and {len(uploaded) - 8} more...")
    if st.button("Process documents and build index", type="primary", disabled=not uploaded, use_container_width=True):
        pages = []
        seen = set()
        progress = st.progress(0, text="Reading documents...")
        for i, file in enumerate(uploaded):
            signature = hash(file.getvalue())
            if signature not in seen:
                seen.add(signature)
                pages.extend(extract_pages(file))
            progress.progress((i + 1) / max(len(uploaded), 1), text=f"Reading documents: {i+1}/{len(uploaded)}")
        chunks = make_chunks(pages, int(chunk_words), int(overlap))
        if not chunks:
            st.error("No readable text found. If your PDFs are scanned images, OCR is required.")
        else:
            embedder = get_embedder()
            with st.spinner(f"Creating embeddings and indexing {len(chunks):,} passages..."):
                build_index(chunks, embedder)
            progress.empty()
            st.success(f"Index ready: {len(uploaded)} files · {len(pages):,} pages/sections · {len(chunks):,} chunks.")
    if st.button("Delete local index", use_container_width=True):
        for path in [CHUNKS_FILE, FAISS_FILE, META_FILE]:
            if path.exists():
                path.unlink()
        st.success("Local index deleted.")

with tab_ask:
    chunks, index = load_index()
    if chunks is None:
        st.info("Your knowledge base is empty. Open **Upload & index** to add documents.")
    else:
        a, b = st.columns([3, 1])
        with a:
            st.metric("Indexed passages", f"{len(chunks):,}")
        with b:
            st.metric("Search mode", "Hybrid")
        question = st.text_area("Ask a question about your documents", placeholder="e.g. What is the procedure for reporting a workplace accident?", height=95)
        if st.button("Search evidence", type="primary", disabled=not question.strip(), use_container_width=True):
            embedder = get_embedder()
            with st.spinner("Searching semantic and keyword indexes..."):
                hits = retrieve(question.strip(), chunks, index, embedder, int(top_k))
            evidence = [hit for hit in hits if hit["retrieval_score"] >= threshold]
            if not evidence:
                st.warning("I could not find sufficient evidence in the indexed documents to answer this reliably.")
            else:
                st.markdown("### Answer")
                if use_ollama:
                    try:
                        st.markdown(ollama_answer(question.strip(), evidence, ollama_model.strip()))
                    except Exception as exc:
                        st.error(f"Ollama could not be reached. Check that it is running and the model is installed. Details: {exc}")
                        st.markdown(evidence_answer(evidence))
                else:
                    st.markdown(evidence_answer(evidence))
                    st.caption("Evidence mode shows retrieved passages. Enable Ollama in the sidebar for locally generated prose.")
                st.markdown("### Source evidence")
                for i, hit in enumerate(evidence, 1):
                    with st.expander(f"{i}. {hit['source']} · page {hit['page']} · match {hit['retrieval_score']:.2f}", expanded=(i == 1)):
                        st.write(hit["text"])
                        st.caption(f"Semantic score: {hit['semantic_score'] if hit['semantic_score'] is not None else 'not ranked'} · Keyword score: {hit['keyword_score']:.3f}")

with tab_status:
    chunks, index = load_index()
    st.subheader("Knowledge base status")
    if chunks is None:
        st.info("No index exists yet.")
    else:
        st.metric("Indexed passages", f"{len(chunks):,}")
        if META_FILE.exists():
            st.json(json.loads(META_FILE.read_text(encoding="utf-8")))
        st.write(f"Index storage: `{INDEX_DIR}`")
        st.warning("Prototype limitations: no authentication, role-based permissions, encryption at rest, or multi-user isolation. Do not use for confidential production data.")
