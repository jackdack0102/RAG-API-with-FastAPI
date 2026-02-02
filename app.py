import os
import uuid

from fastapi import FastAPI
import chromadb
import ollama

# ---- Config ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

# ---- Clients ----
ollama_client = ollama.Client(host=OLLAMA_HOST)

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")


@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results.get("documents") else ""

    answer = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
    )

    return {"answer": answer.get("response", "")}


@app.post("/add")
def add_knowledge(text: str):
    doc_id = str(uuid.uuid4())
    collection.add(documents=[text], ids=[doc_id])
    return {"status": "success", "message": "Content added to knowledge base", "id": doc_id}

print(f"Using OLLAMA_HOST={OLLAMA_HOST}, OLLAMA_MODEL={OLLAMA_MODEL}", flush=True)
