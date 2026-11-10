import os, json
from pypdf import PdfReader

PAPERS_DIR = "data/papers"
OUT_FILE = "data/chunks.json"
CHUNK_SIZE = 1000      # characters per chunk
OVERLAP = 200          # characters shared between neighbouring chunks

def extract_text(path):
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return chunks

all_chunks = []
for name in sorted(os.listdir(PAPERS_DIR)):
    if not name.endswith(".pdf"):
        continue
    try:
        text = extract_text(os.path.join(PAPERS_DIR, name))
    except Exception as e:
        print("FAILED", name, e)
        continue
    pieces = chunk_text(text)
    for i, piece in enumerate(pieces):
        all_chunks.append({"paper": name, "chunk_id": i, "text": piece})
    print(f"{name}: {len(text)} characters, {len(pieces)} chunks")

with open(OUT_FILE, "w") as f:
    json.dump(all_chunks, f)
print("Total chunks:", len(all_chunks))
