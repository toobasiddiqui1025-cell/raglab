# Inquira 🔍

### *Ask. Explore. Understand.*

> **Your AI Research Companion.**
>
> Inquira is an AI-powered research assistant built using Retrieval-Augmented Generation (RAG), with a scientific evaluation framework to measure and improve answer quality.

---

## 🎨 Brand Identity & Design System

Inquira is built around a modern, scientific visual identity designed for clarity, depth, and precision:

| Color Token | Hex Code | Visual Preview | Role |
| :--- | :--- | :--- | :--- |
| **Midnight Navy** | `#101D35` | `■ #101D35` | Primary Canvas & Deep Surface Contrast |
| **Teal** | `#65D6D0` | `■ #65D6D0` | Primary Accent / Interactive Highlights & Evidence |
| **Violet** | `#8B7CF6` | `■ #8B7CF6` | Secondary Accent / AI Synthesis & Citations |
| **Cloud White** | `#F4F6FA` | `■ #F4F6FA` | Light Neutral / Surface Layer & Readable Content |

- **Tagline**: *Ask. Explore. Understand.*
- **Mission**: Empower researchers and ML practitioners to interrogate academic literature with mathematically grounded citations, zero tolerance for hallucinations, and empirically benchmarked retrieval quality.

---

## Architecture

```mermaid
graph TD
    User([Researcher Query]) --> UI[Inquira UI: Streamlit / FastAPI]
    UI --> Retriever[Hybrid Retriever]
    Retriever --> Dense[Dense Vector Search: ChromaDB]
    Retriever --> Sparse[Sparse Lexical Search: BM25]
    Dense --> Fusion[Reciprocal Rank Fusion - RRF]
    Sparse --> Fusion
    Fusion --> Reranker[Cross-Encoder Reranker]
    Reranker --> Prompt[Grounded Prompt Assembly & Citation Enforcer]
    Prompt --> LLM[LLM Engine: Gemini Client]
    LLM --> Answer([Cited Answer [doc_id, p.N]])

    subgraph Evaluation Harness
        Gold[eval/gold_questions.json] --> Runner[Benchmark Runner]
        Runner --> PageMetrics[Page-Level Recall@k, MRR, nDCG@k]
        Runner --> Judge[LLM-as-a-Judge: Correctness & Faithfulness]
        Runner --> Results[eval/results/]
    end
```

## Features

- **Page-Aware Ingestion**: PDF extraction and chunking preserving exact document IDs and page numbers for source attribution.
- **Hybrid Retrieval**: Combines semantic embeddings (Sentence-Transformers) and keyword search (BM25) fused with Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking**: Precision rescoring of candidate chunks before prompt augmentation.
- **Strict Grounding & Citation**: Structured prompting enforcing page-level citations (`[doc_id, p.N]`) and explicit refusals on out-of-context queries.
- **Evaluation Harness**: Automated evaluation recording Recall@k, MRR, nDCG@k, and LLM-as-a-judge faithfulness/correctness scores.
- **Production APIs & UI**: FastAPI endpoints (`/ask`, `/health`) and interactive Streamlit analytics dashboard.

## Setup & Quickstart

```bash
# Clone and enter repository
git clone https://github.com/YOUR_USERNAME/raglab.git
cd raglab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .

# Configure environment variables
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY

# Run Inquira Streamlit App
streamlit run app/streamlit_app.py
```

## Project Structure

```text
raglab/
├── src/raglab/
│   ├── config.py              # Settings via Pydantic + YAML
│   ├── ingestion/
│   │   ├── loader.py          # PDF parsing -> page objects (PyMuPDF)
│   │   └── chunker.py         # Fixed-size and recursive chunkers
│   ├── indexing/
│   │   ├── embedder.py        # Dense embeddings (Sentence Transformers)
│   │   ├── vector_store.py    # Persistent vector store (ChromaDB)
│   │   └── bm25_index.py      # Sparse lexical index (BM25)
│   ├── retrieval/
│   │   ├── dense.py           # Dense semantic search
│   │   ├── sparse.py          # BM25 keyword search
│   │   ├── hybrid.py          # Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py        # Cross-encoder rescoring
│   ├── generation/
│   │   ├── llm.py             # LLM abstractions (GeminiClient, MockLLMClient)
│   │   ├── prompts.py         # Prompt templates and citation enforcement
│   │   └── answerer.py        # End-to-end grounded answer pipeline
│   ├── evaluation/
│   │   ├── dataset.py         # Gold evaluation dataset schema & loader
│   │   ├── question_prompts.py   # Prompt templates for draft question generation
│   │   ├── retrieval_metrics.py  # Recall@k, MRR, nDCG@k
│   │   ├── generation_metrics.py # LLM-as-a-judge (faithfulness, correctness)
│   │   ├── runner.py          # Automated benchmark runner
│   │   └── report.py          # Markdown tables and Plotly visualizations
│   └── api/
│       └── main.py            # Inquira FastAPI REST service (/ask, /health)
├── app/
│   └── streamlit_app.py       # Inquira Streamlit Research & Evaluation App
├── configs/                   # Experiment configurations (YAML)
├── scripts/
│   └── draft_questions.py     # Generates UNVERIFIED draft gold questions
├── data/papers/               # ArXiv PDF corpus
├── eval/
│   ├── gold_questions.json    # Verified gold benchmark questions
│   ├── drafts/                # Unverified AI drafts (gold_questions_draft.json)
│   └── results/               # Experiment benchmark outputs
├── tests/                     # Unit and integration test suite
├── docs/
│   └── DECISIONS.md           # Architecture Decision Records (ADRs)
├── Dockerfile
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Phase 3: Gold Evaluation Dataset

> **Status:** the tooling below is implemented, but `eval/gold_questions.json` currently holds only one seed question. The 50-question benchmark still has to be written and verified by hand.

### What a gold evaluation dataset is

A gold evaluation dataset is a fixed set of questions, each with a reference answer and the exact places in the corpus that support it. The benchmark runner compares the system's retrieved pages and generated answers against these labels, so Recall@k, MRR, nDCG@k and answer-quality scores are measured against ground truth rather than judged by feel. The labels are only as good as the people who verified them.

### Why page-level labels

Each question is labelled with the document and page numbers that contain its evidence, not with chunk IDs.

- Page numbers do not change when the chunker, chunk size or overlap changes, so the same gold set can be reused across chunking experiments.
- Page numbers are easy for a person to check in the PDF.
- Page numbers are 1-indexed and match `Page.page_number` from the PDF loader. `doc_id` is the PDF filename stem (e.g. `1706.03762`).

### Question categories

| Type | Meaning | Evidence |
|---|---|---|
| `factoid` | A single short fact, normally from one page | typically one document, one page |
| `numeric_detail` | A specific number, hyperparameter or metric, with exact value and unit | typically one document, one page |
| `multi_chunk` | An answer that needs information from several pages, possibly several documents | normally two or more pages |
| `unanswerable` | The corpus does not contain the answer | none (`is_answerable: false`) |

### Target distribution (50 questions)

| Type | Target |
|---|---|
| `factoid` | 20 |
| `numeric_detail` | 12 |
| `multi_chunk` | 10 |
| `unanswerable` | 8 |

This is a target for authoring. The loader does not enforce these counts, nor the typical page counts per type in the table above (for example, it accepts a `multi_chunk` question with a single evidence page), so check those by hand.

### Schema

`eval/gold_questions.json` is a JSON array of questions (`src/raglab/evaluation/dataset.py`):

```json
{
  "id": "q001",
  "type": "factoid",
  "question": "What optimizer and learning rate schedule was used in Attention Is All You Need?",
  "reference_answer": "The Adam optimizer ... with a warmup schedule of 4000 warmup steps.",
  "is_answerable": true,
  "gold_evidence": [
    {"doc_id": "1706.03762", "pages": [7]}
  ]
}
```

- **`GoldQuestion`**: `id`, `type`, `question`, `reference_answer`, `is_answerable`, `gold_evidence` (a list of `GoldEvidence`).
- **`GoldEvidence`**: `doc_id` and `pages` (a list of 1-indexed page numbers).
- `pages` must be real integers and `is_answerable` a real boolean; values like `"3"`, `true` or `"yes"` are rejected rather than coerced.
- Unknown fields are rejected, so a typo such as `gold_evidnce` is an error rather than being silently ignored.

### How validation works

Validation has two layers. Every problem is reported with the question ID and field, and problems are collected so you see them all at once.

1. **Per question (Pydantic):**
   - `id`, `question` and `reference_answer` must be non-empty.
   - `type` must be one of the four categories.
   - `pages` must be non-empty with every page >= 1. Duplicates are removed and the list is sorted.
   - A `doc_id` may appear only once per question. Merge page lists into one entry.
   - An `unanswerable` question must have `is_answerable: false`. Unanswerable questions must have empty `gold_evidence`, and answerable ones must have at least one entry.
2. **Whole dataset (`load_gold_dataset`):**
   - Question IDs must be unique.
   - If a corpus registry is passed, every `doc_id` must exist in the corpus and every page must be within that document's page count. The registry is built from the actual PDFs with `build_corpus_registry`; no document IDs or page counts are hardcoded.

Without a registry the corpus checks are skipped. Always pass one for the real dataset.

### Draft question generation

`scripts/draft_questions.py` helps with authoring by proposing candidate questions. **Its output is a set of drafts only. LLM-generated questions are never verified, and they are never written to `eval/gold_questions.json`.**

How it works:

1. Loads the PDFs in `data/papers/` with the existing loader and keeps pages with enough text (at least 500 characters).
2. Randomly samples pages with a fixed seed (`--seed`, default 42), so a run is reproducible. `multi_chunk` samples groups of two pages, from one document or from two.
3. Sends the pages to the configured LLM (`GeminiClient`) using the prompt in `src/raglab/evaluation/question_prompts.py`. Page text is fenced as untrusted data, and the prompt tells the model to ignore instructions inside it and to return no question when the text does not support one.
4. Validates the JSON response. A candidate is dropped if it has the wrong type, cites a page that was not supplied, or has the wrong number of pages for its type. Malformed responses are retried once, and API errors are logged and skipped.
5. Writes `eval/drafts/gold_questions_draft.json` with a `WARNING` field. Each draft is marked `"verified": false` and `"status": "UNVERIFIED_AI_DRAFT"` and includes a `reason` explaining the cited evidence.

Safety behavior: the script refuses any output path named `gold_questions.json`, and it will not overwrite an existing draft file unless you pass `--force`.

The script generates only `factoid`, `numeric_detail` and `multi_chunk` drafts. **`unanswerable` questions are not generated and must be written by hand.**

### Generating candidates

Set `GEMINI_API_KEY` in `.env` (see Setup above). The model and token limit come from `configs/baseline.yaml`. Real runs call the Gemini API and use your quota.

```bash
# Preview which pages would be sampled (no LLM calls, nothing written)
python scripts/draft_questions.py --dry-run --seed 42

# Generate drafts (counts are the number of pages or page groups to sample)
python scripts/draft_questions.py --seed 42 --factoid 5 --numeric 5 --multi-chunk 3

# Other options: --output, --per-page, --temperature, --retries, --force, --papers-dir, --config
```

Loading the whole corpus takes about a minute per run.

### Manually verifying candidates

Every question in the final gold dataset must be verified by a person. For each draft:

1. Open the cited PDF at the cited page(s). Check that the page numbers are the PDF's page positions (1 = first page), not the printed numbers on the page.
2. Confirm the answer is stated in the cited text, with the exact number and unit for numeric questions.
3. Confirm the question is unambiguous and can be understood without the draft context.
4. For `multi_chunk`, confirm the answer really needs every cited page, and that no needed page is missing.
5. Fix or discard anything that fails. Edit freely; the draft is only a starting point.
6. Copy each accepted question into `eval/gold_questions.json` by hand, using the gold schema: assign an `id`, set `is_answerable: true`, and drop the draft-only fields (`draft_id`, `status`, `verified`, `reason`).
7. Write the `unanswerable` questions yourself (`is_answerable: false`, `gold_evidence: []`, and a reference answer saying why the corpus has no answer). Check that the corpus really does not contain the answer.

### Loading and validating the gold dataset

```bash
python - <<'PY'
from pathlib import Path
from raglab.evaluation.dataset import build_corpus_registry, load_gold_dataset

registry = build_corpus_registry(Path("data/papers"))
questions = load_gold_dataset(Path("eval/gold_questions.json"), registry)
print(f"{len(questions)} questions valid")
PY
```

On failure a `DatasetValidationError` lists every problem, for example `[q007] type: Input should be 'factoid', ...`. Validation checks structure and corpus consistency only. It cannot tell whether an answer is correct, which is why manual verification is required.

### Phase 3 tests

```bash
pytest tests/test_dataset.py tests/test_evaluation_dataset.py tests/test_draft_questions.py \
       tests/test_question_prompts.py tests/test_llm.py
```

These tests use mocked LLM responses and temporary files; they make no API calls and do not touch the real gold dataset.
