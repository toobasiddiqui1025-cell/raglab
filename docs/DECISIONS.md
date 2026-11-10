# Architectural Decision Records (ADRs)

Documenting key technical choices, alternatives considered, and rationale.

---

## ADR 001: Project Architecture and Modular Pipeline Design

- **Status**: Accepted
- **Date**: 2026-11-10
- **Context**: Setting up initial repository layout for RAGLab to support both iterative development and rigorous evaluation across diverse retrieval and generation strategies.
- **Choice**: Separate modules into dedicated packages under `src/raglab/` (`ingestion`, `indexing`, `retrieval`, `generation`, `evaluation`, `api`) orchestrated by configuration schemas.
- **Rejected**: Monolithic scripts or notebook-driven pipelines.
- **Why**: High modularity enables isolated ablation testing (e.g. swapping retrievers or chunkers) and facilitates automated CI without coupling components to external network calls or single LLM vendors.
