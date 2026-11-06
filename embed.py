"""Embedding model (multilingual-e5-small) and off-site step 2: embed every chunk.

Run from fieldmed/:  python -m pipeline.embed
"""
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import DB_PATH, EMB_PATH, EMBED_MODEL

torch.set_num_threads(max(1, torch.get_num_threads()))


class Embedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        model = AutoModel.from_pretrained(EMBED_MODEL).eval()
        # int8 linear layers: ~2x faster on CPU, negligible quality loss
        self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int = 320) -> np.ndarray:
        b = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        h = self.model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(v, dim=-1).numpy()

    def query(self, q: str) -> np.ndarray:
        return self.encode(["query: " + q])[0]


def main(batch: int = 32):
    db = sqlite3.connect(DB_PATH)
    rows = db.execute("""SELECT c.id, d.title, c.heading, c.text FROM chunks c
                         JOIN docs d ON d.id = c.doc_id ORDER BY c.id""").fetchall()
    assert rows[-1][0] == len(rows), "chunk ids must be 1..N so row i maps to id i+1"
    emb = Embedder()
    # Checkpoint every 20 batches so an interrupted run resumes instead of starting over.
    part = EMB_PATH.with_suffix(".partial.npy")
    out = [np.load(part)] if part.exists() else []
    start = len(out[0]) if out else 0
    t0 = time.time()
    for i in range(start, len(rows), batch):
        texts = [f"passage: {t}. {h}\n{x}" for _, t, h, x in rows[i:i + batch]]
        out.append(emb.encode(texts).astype(np.float16))
        if i // batch % 20 == 0:
            np.save(part, np.vstack(out))
            done = i + len(texts)
            eta = (time.time() - t0) / (done - start) * (len(rows) - done)
            print(f"{done}/{len(rows)} chunks, ETA {eta / 60:.1f} min", flush=True)
    np.save(EMB_PATH, np.vstack(out))
    part.unlink(missing_ok=True)
    print(f"saved {EMB_PATH} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
