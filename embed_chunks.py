import os, json, time
import numpy as np
from dotenv import load_dotenv, dotenv_values
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=dotenv_values(".env")["GEMINI_API_KEY"])

MODEL = "gemini-embedding-001"
DIM = 768
BATCH = 50
PARTIAL = "data/embeddings_partial.npy"
OUT = "data/embeddings.npy"

chunks = json.load(open("data/chunks.json"))
texts = [c["text"].strip() or " " for c in chunks]

# resume if a previous run stopped halfway
vectors = list(np.load(PARTIAL)) if os.path.exists(PARTIAL) else []
print(f"Starting at chunk {len(vectors)} of {len(texts)}")

while len(vectors) < len(texts):
    start = len(vectors)
    batch = texts[start:start + BATCH]
    for attempt in range(6):
        try:
            res = client.models.embed_content(
                model=MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=DIM,
                ),
            )
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print("Error:", str(e)[:150], f"-> waiting {wait}s")
            time.sleep(wait)
    else:
        raise SystemExit("Too many errors. Run the script again to resume.")

    for emb in res.embeddings:
        v = np.array(emb.values, dtype="float32")
        vectors.append(v / np.linalg.norm(v))   # normalise for cosine search
    np.save(PARTIAL, np.array(vectors))
    print(f"Embedded {len(vectors)} / {len(texts)}")
    time.sleep(1)

np.save(OUT, np.array(vectors))
print("Done. Saved", OUT, np.array(vectors).shape)
