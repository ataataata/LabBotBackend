# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from ollama import Client
import numpy as np
import json, os, traceback

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

# ---- Config ----
# Examples:
#   qwen3:0.6b-instruct
#   qwen3:1.8b-instruct
MODEL  = os.getenv("OLLAMA_MODEL", "qwen3:0.6b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

K = 3
THRESH = 0.30  # cosine similarity floor after row-normalization

GEN_OPTS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "num_ctx": 2048,
    "stop": ["\nQ:", "\nA:", "Q:", "A:"],
    # some Qwen builds support this flag; harmless if ignored
    "thinking": False
}

SYS_MSG = (
    "You are the website chatbot for our demo.\n"
    "You MUST ONLY answer using the Q-A pairs shown below.\n"
    "If the answer is not present, reply exactly: \"I don't know.\""
)

# ---- App + Data ----
app = FastAPI()
oll = Client()

def _load_data():
    qa = json.load(open("qa.json")) if os.path.exists("qa.json") else []
    vecs = np.load("vecs.npy").astype(np.float32) if os.path.exists("vecs.npy") else None
    # Expect row-normalized vectors (build_index.py does this), but normalize again defensively
    if vecs is not None and vecs.ndim == 2:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms
    print(f"Loaded {len(qa)} Q-A pairs; vecs: {None if vecs is None else vecs.shape}")
    return qa, vecs

qa, vecs = _load_data()

def embed(text: str) -> np.ndarray:
    try:
        e = oll.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
        v = np.asarray(e, dtype=np.float32)
        n = np.linalg.norm(v) + 1e-9
        return v / n
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.zeros((1,), dtype=np.float32)

def retrieve(query: str, k: int = K):
    if not qa or vecs is None or vecs.ndim != 2 or vecs.shape[0] != len(qa):
        print("Retrieval skipped: missing or mismatched qa/vecs.")
        return [], np.array([])
    qv = embed(query)
    if qv.ndim != 1 or qv.shape[0] != vecs.shape[1]:
        print("Retrieval skipped: embedding dim mismatch.")
        return [], np.array([])
    sims = vecs @ qv  # row-normalized -> cosine
    top_idx = np.argsort(sims)[-k:][::-1]
    top_ctx = [qa[i] for i in top_idx]
    print("Top-3 questions:", [c["q"] for c in top_ctx])
    return top_ctx, sims[top_idx]

def build_messages(user_msg: str):
    ctx, sims = retrieve(user_msg)
    if len(ctx) > 0 and np.max(sims) >= THRESH:
        kb = "\n".join(f"Q: {c['q']}\nA: {c['a']}" for c in ctx)
        system_content = f"{SYS_MSG}\n\n{kb}"
    else:
        system_content = f"{SYS_MSG}\n\nNo relevant context."
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_msg},
    ]

def strip_think(text: str) -> str:
    # Remove optional <think> blocks if any model leaks them
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        print(f"Received message: {req.message}")
        messages = build_messages(req.message)
        res = oll.chat(model=MODEL, messages=messages, stream=False, options=GEN_OPTS)
        content = (res or {}).get("message", {}).get("content", "") if isinstance(res, dict) else ""
        answer = strip_think(content)
        if not answer:
            answer = "I don't know."
        # Enforce policy one last time:
        if "No relevant context." in messages[0]["content"] and answer.lower() != "i don't know.":
            answer = "I don't know."
        return ChatResponse(answer=answer)
    except Exception as e:
        print("Backend error:", e)
        print(traceback.format_exc())
        return ChatResponse(answer=f"Backend error: {e}")

@app.get("/health")
async def health_check():
    try:
        models = oll.list().get("models", [])
        return {"status": "healthy", "models_available": [m.get('name') for m in models]}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting backend server… MODEL:", MODEL)
    uvicorn.run(app, host="0.0.0.0", port=8001)
