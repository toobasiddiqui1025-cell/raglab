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
│   │   ├── retrieval_metrics.py  # Recall@k, MRR, nDCG@k
│   │   ├── generation_metrics.py # LLM-as-a-judge (faithfulness, correctness)
│   │   ├── runner.py          # Automated benchmark runner
│   │   └── report.py          # Markdown tables and Plotly visualizations
│   └── api/
│       └── main.py            # Inquira FastAPI REST service (/ask, /health)
├── app/
│   └── streamlit_app.py       # Inquira Streamlit Research & Evaluation App
├── configs/                   # Experiment configurations (YAML)
├── data/papers/               # ArXiv PDF corpus
├── eval/
│   ├── gold_questions.json    # Benchmark questions
│   └── results/               # Experiment benchmark outputs
├── tests/                     # Unit and integration test suite
├── docs/
│   └── DECISIONS.md           # Architecture Decision Records (ADRs)
├── Dockerfile
├── pyproject.toml
├── .env.example
└── .gitignore
```