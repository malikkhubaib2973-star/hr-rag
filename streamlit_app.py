"""Company Policy RAG Assistant - Streamlit UI
Login -> 1. Files save karo -> 2. Embeddings banao -> 3. Sawal poochein (retrieval test ke sath)
Chalane ke liye:  python -m streamlit run streamlit_app.py
"""
import os
import secrets
import time
import streamlit as st
from rag import ask, add_file, list_files, _col  # rag.py .env khud load kar leta hai

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE, "data", "company_docs")
os.makedirs(DOCS_DIR, exist_ok=True)
SUPPORTED = (".pdf", ".docx", ".txt", ".md")
APP_USER = os.getenv("APP_USER", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

st.set_page_config(page_title="Company Policy RAG", page_icon="📚")


def same(a, b):
    return bool(b) and secrets.compare_digest(a.encode(), b.encode())


def doc_files():
    return sorted(n for n in os.listdir(DOCS_DIR) if n.lower().endswith(SUPPORTED))


def retrieve(question, source=None, k=5):
    """Retrieval test: sab se milte julte hissay. Distance jitni chhoti, match utna behtar."""
    where = {"source": source} if source else None
    res = _col.query(query_texts=[question], n_results=k, where=where)
    out = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"source": meta["source"], "page": meta.get("page"), "distance": dist, "text": text})
    return out


# ================= AUTHENTICATION =================
if not st.session_state.get("auth"):
    st.title("🔐 Company Policy RAG")
    st.caption("Aage barhne ke liye login karein.")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        go = st.form_submit_button("Login")
    if go:
        if same(u, APP_USER) and same(p, APP_PASSWORD):
            st.session_state.auth = True
            st.session_state.user = u
            st.rerun()
        else:
            time.sleep(1)  # galat password par thora rukna
            st.error("Username ya password galat hai")
    if not APP_USER or not APP_PASSWORD:
        st.warning(".env mein APP_USER aur APP_PASSWORD likhna zaroori hai.")
    st.stop()

# ================= SIDEBAR =================
with st.sidebar:
    st.success("Authenticated")
    st.caption(f"User: {st.session_state.get('user', '')}")
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
    st.divider()
    st.subheader("Document folder")
    st.code(os.path.join("data", "company_docs"))
    st.caption("Supported files: PDF, DOCX, TXT")

# ================= MAIN =================
st.title("Company Policy RAG Assistant")
st.caption("Search HR policies and other approved company documents.")

# ---- 1. Files add karo ----
st.header("1. Add company files")
uploads = st.file_uploader(
    "Upload HR policies, employee handbooks, or other approved documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)
if st.button("Save uploaded files"):
    if not uploads:
        st.warning("Pehle upar se file chunein.")
    else:
        for f in uploads:
            with open(os.path.join(DOCS_DIR, os.path.basename(f.name)), "wb") as out:
                out.write(f.getbuffer())
        st.session_state.last_file = os.path.basename(uploads[-1].name)
        st.success(f"{len(uploads)} file(s) save ho gayi.")

files = doc_files()
st.write("Files available for indexing: " + (", ".join(files) if files else "abhi koi nahi"))

# ---- 2. Embeddings banao ----
st.header("2. Embed files and build the searchable index")
if st.button("Build / refresh embeddings", type="primary"):
    if not files:
        st.warning("Pehle file save karein (step 1).")
    else:
        bar = st.progress(0.0, text="Shuru ho raha hai...")
        for i, name in enumerate(files):
            bar.progress(i / len(files), text=f"{name} embed ho rahi hai... ({i + 1}/{len(files)}) - bari file mein 1-3 minute lag sakte hain")
            try:
                n = add_file(os.path.join(DOCS_DIR, name), name)
                if n == 0:
                    st.error(f"{name}: text nahi nikla (shayad scanned/image wali file hai)")
                else:
                    st.success(f"{name}: {n} chunks embed ho gaye")
            except Exception as e:
                st.error(f"{name}: error - {type(e).__name__}: {e}")
        bar.progress(1.0, text="Mukammal")

indexed = list_files()
if indexed:
    st.caption("Indexed: " + ", ".join(f"{k} ({v} chunks)" for k, v in indexed.items()))
else:
    st.info("Abhi koi index nahi bana. Upar 'Build / refresh embeddings' dabayein.")

# ---- 3. Sawal poochein ----
st.header("3. Ask questions and test retrieval")
st.caption("Example: How many annual leave days are allowed?")

options = ["All files"] + list(indexed)
last = st.session_state.get("last_file")
scope = st.selectbox("Search in", options, index=options.index(last) if last in options else 0)

with st.form("ask"):
    q = st.text_input("Your question")
    go = st.form_submit_button("Ask")

if go and q.strip():
    if not indexed:
        st.warning("Pehle embeddings banayein (step 2).")
    else:
        src = None if scope == "All files" else scope
        with st.spinner("Soch raha hoon..."):
            try:
                st.session_state.result = {
                    "q": q.strip(),
                    "r": ask(q.strip(), source=src),
                    "passages": retrieve(q.strip(), source=src),
                }
            except Exception as e:
                st.session_state.result = {"q": q.strip(), "r": {"answer": f"Error: {type(e).__name__}: {e}", "sources": []}, "passages": []}

res = st.session_state.get("result")
if res:
    st.subheader("Answer")
    st.write(res["r"]["answer"])
    if res["r"]["sources"]:
        st.caption("Source: " + " | ".join(res["r"]["sources"]))
    with st.expander("Retrieved passages (retrieval test)"):
        for p in res["passages"]:
            page = f"page {p['page']}" if p["page"] else "page -"
            st.markdown(f"**{p['source']}, {page}**  (distance: {p['distance']:.3f})")
            st.text(p["text"])
            st.divider()