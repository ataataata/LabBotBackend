# build_index.py
import json, numpy as np, ollama, sys

EMBED_MODEL = "nomic-embed-text"  # keep in sync with backend

def embed(client, text: str) -> np.ndarray:
    e = client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
    v = np.asarray(e, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def main(path="qa.json", out="vecs.npy"):
    client = ollama.Client()
    qa = json.load(open(path, "r"))
    if not qa or not isinstance(qa, list):
        print("qa.json is empty or invalid list.")
        sys.exit(1)

    qs = [row["q"] for row in qa if row.get("q")]
    if not qs:
        print("No 'q' fields found in qa.json.")
        sys.exit(1)

    vecs = np.vstack([embed(client, q) for q in qs])
    np.save(out, vecs.astype(np.float32))
    print("Saved", vecs.shape, "→", out)

if __name__ == "__main__":
    main()
