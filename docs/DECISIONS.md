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

---

## ADR 002: Product Identity, Branding, and Visual Design System

- **Status**: Accepted
- **Date**: 2026-11-10
- **Context**: Defining user-facing product identity, nomenclature, and design system for interactive application surfaces (Streamlit dashboard, OpenAPI documentation, and user-facing artifacts).
- **Choice**:
  - **Product Name**: Inquira
  - **Tagline**: *Ask. Explore. Understand.*
  - **Description**: *Your AI Research Companion.* An AI-powered research assistant built using Retrieval-Augmented Generation (RAG) over ML papers with a scientific evaluation framework.
  - **Brand Palette**:
    - Midnight Navy (`#101D35`): Deep primary surface and dark contrast.
    - Teal (`#65D6D0`): Primary accent for exploration, evidence tags, and active states.
    - Violet (`#8B7CF6`): Secondary accent for AI synthesis, reasoning indicators, and citations.
    - Cloud White (`#F4F6FA`): Crisp neutral for cards, panels, and text readability.
- **Rejected**: Generic internal titles (such as "RAGLab Streamlit App" or generic dashboard themes).
- **Why**: Establishes a professional, memorable product identity for users and portfolio evaluators while cleanly maintaining the internal Python engine structure (`raglab`) intact without namespace collisions or breaking imports.
