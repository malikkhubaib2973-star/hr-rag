"""RAG core: embed files, retrieve relevant passages for a question, answer with Groq."""
import os
import chromadb
from dotenv import load_dotenv
from ingest import make_chunks

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))

DB_DIR = os.path.join(BASE, "db")
COLLECTION = "company_docs"
MODEL = "openai/gpt-oss-120b"

_client = chromadb.PersistentClient(path=DB_DIR)
_col = _client.get_or_create_collection(COLLECTION)


def add_file(path, name):
    """Read the file, make chunks, embed and save. Returns the number of chunks."""
    try:
        _col.delete(where={"source": name})  # avoid duplicates on re-upload
    except Exception:
        pass
    items = make_chunks(path)
    if not items:
        return 0
    ids, docs, metas = [], [], []
    for i, (text, page) in enumerate(items):
        ids.append(f"{name}-{i}")
        docs.append(text)
        m = {"source": name}
        if page:
            m["page"] = page
        metas.append(m)
    for s in range(0, len(ids), 64):
        _col.upsert(ids=ids[s:s + 64], documents=docs[s:s + 64], metadatas=metas[s:s + 64])
    return len(ids)


def list_files():
    """{file_name: chunks} - which files are currently embedded."""
    counts = {}
    for m in _col.get(include=["metadatas"])["metadatas"]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return counts


def _sources(metas):
    pages = {}
    for m in metas:
        pages.setdefault(m["source"], set())
        if "page" in m:
            pages[m["source"]].add(m["page"])
    out = []
    for name, ps in sorted(pages.items()):
        out.append(f"{name} (page {', '.join(map(str, sorted(ps)))})" if ps else name)
    return out


def _fallback(chunks, metas, note):
    """If the LLM key is missing or an error occurs, show the closest passages directly."""
    parts = []
    for c, m in list(zip(chunks, metas))[:3]:
        parts.append(f"[Page {m.get('page', '-')}] {c[:600]}...")
    return note + "\n\nClosest passages from the file:\n\n" + "\n\n".join(parts)


def ask(question, k=5, source=None):
    """source=None searches all files, otherwise only that file (e.g. 'Employee_Handbook_2018.pdf')."""
    if _col.count() == 0:
        return {"answer": "No file has been uploaded yet. Please attach a file first.", "sources": []}

    where = {"source": source} if source else None
    res = _col.query(query_texts=[question], n_results=k, where=where)
    if not res["documents"][0]:
        return {"answer": f"Nothing found in '{source}'. Please upload the file again and retry.", "sources": []}
    chunks, metas = res["documents"][0], res["metadatas"][0]
    sources = _sources(metas)

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        note = "(GROQ_API_KEY is missing in .env, so showing only the matching passages.)"
        return {"answer": _fallback(chunks, metas, note), "sources": sources}

    context = "\n\n---\n\n".join(
        f"[{m['source']}, page {m.get('page', '-')}]\n{c}"
        for c, m in zip(chunks, metas)
    )
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are an HR policy assistant. Answer ONLY from the given context. "
                               "If the answer is not in the context, say clearly that it is not in the file. "
                               "Always answer in English, even if the question is in another language. "
                               "Keep the answer clear and well structured.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        note = f"(Groq error: {e}. Showing only the matching passages.)"
        return {"answer": _fallback(chunks, metas, note), "sources": sources}

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    while True:
        q = input("\nQuestion (type exit to quit): ").strip()
        if q.lower() == "exit":
            break
        r = ask(q)
        print("\nAnswer:", r["answer"])
        print("Sources:", ", ".join(r["sources"]))