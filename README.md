# RAGLab 🔬

A modular, highly-evaluable Retrieval-Augmented Generation (RAG) system over machine learning research papers with rigorous offline benchmarking.

## Architecture

```mermaid
graph TD
    User([User Query]) --> API[FastAPI / Streamlit]
    API --> Retriever[Hybrid Retriever]
    Retriever --> Dense[Dense Vector Search: ChromaDB]
    Retriever --> Sparse[Sparse Lexical Search: BM25]
    Dense --> Fusion[Reciprocal Rank Fusion]
    Sparse --> Fusion
    Fusion --> Reranker[Cross-Encoder Reranker]
    Reranker --> Prompt[Prompt Assembly & Citation Enforcer]
    Prompt --> LLM[LLM: Gemini Client]
    LLM --> Answer([Cited Answer [doc_id, p.N]])

    subgraph Evaluation Harness
        Gold[eval/gold_questions.json] --> Runner[Evaluation Runner]
        Runner --> PageMetrics[Recall@k, MRR, nDCG@k]
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
│       └── main.py            # FastAPI REST service (/ask, /health)
├── app/
│   └── streamlit_app.py       # Interactive Q&A & evaluation dashboard
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