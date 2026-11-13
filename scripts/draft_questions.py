#!/usr/bin/env python3
"""Draft gold-question generator for Inquira / RAGLab.

Randomly samples pages from the PDF corpus (via the existing loader) and asks
the configured generation LLM to PROPOSE candidate evaluation questions of
type 'factoid', 'numeric_detail' and 'multi_chunk'.

IMPORTANT: every output item is an UNVERIFIED AI-generated draft. Nothing is
verified until a human has checked the question, the reference answer and the
cited pages against the PDF. The output is written ONLY to
eval/drafts/gold_questions_draft.json; this script never writes to, or
overwrites, eval/gold_questions.json. Verified drafts must be copied over by
hand (dropping the draft-only fields) and then checked with load_gold_dataset().

Requirements to run for real: the project's configured provider is Gemini
(generation.model_name in configs/baseline.yaml), which needs GEMINI_API_KEY
set in the environment or in .env (see .env.example).

Usage:
    python scripts/draft_questions.py --seed 42 --factoid 5 --numeric 5 --multi-chunk 3
    python scripts/draft_questions.py --dry-run          # no LLM; show sampled pages
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from raglab.config import AppConfig, load_config
from raglab.evaluation.question_prompts import (
    PROMPT_VERSION,
    QUESTION_GENERATION_SYSTEM_PROMPT,
    build_question_generation_prompt,
)
from raglab.generation.llm import (
    LLMAuthenticationError,
    LLMClient,
    LLMConfigurationError,
)
from raglab.ingestion.loader import Page, load_directory

logger = logging.getLogger("draft_questions")

DEFAULT_OUTPUT = Path("eval/drafts/gold_questions_draft.json")
GOLD_DATASET_FILENAME = "gold_questions.json"
MIN_PAGE_CHARS = 500
MULTI_CHUNK_PAGES = 2

DraftType = Literal["factoid", "numeric_detail", "multi_chunk"]

VERIFICATION_WARNING = (
    "WARNING: ALL ITEMS BELOW ARE UNVERIFIED AI-GENERATED DRAFTS. Questions, "
    "reference answers and page citations may be wrong or hallucinated. Each "
    "item MUST be manually verified against the source PDF before being copied "
    "into eval/gold_questions.json."
)

class LLMUnavailableError(RuntimeError):
    """Raised when no usable generation LLM is configured or implemented."""


class DraftEvidence(BaseModel):
    """Evidence cited by a draft: one document and its 1-indexed pages."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    pages: List[int]

    @field_validator("pages")
    @classmethod
    def _pages_non_empty(cls, v: List[int]) -> List[int]:
        if not v or any(p < 1 for p in v):
            raise ValueError("pages must be a non-empty list of integers >= 1")
        return sorted(set(v))


class DraftCandidate(BaseModel):
    """One LLM-proposed question, as returned by the model (pre-checks)."""

    model_config = ConfigDict(extra="ignore")

    question: str
    type: DraftType
    reference_answer: str
    evidence: List[DraftEvidence]
    reason: str

    @field_validator("question", "reference_answer", "reason")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("evidence")
    @classmethod
    def _evidence_non_empty(cls, v: List[DraftEvidence]) -> List[DraftEvidence]:
        if not v:
            raise ValueError("evidence must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

PageRef = Tuple[str, int]  # (doc_id, page_number)


def eligible_pages(pages: Sequence[Page], min_chars: int = MIN_PAGE_CHARS) -> List[Page]:
    """Keep pages with enough text to support a question (drops blank/figure pages)."""
    return [p for p in pages if len(p.text.strip()) >= min_chars]


def sample_single_pages(pages: Sequence[Page], n: int, rng: random.Random) -> List[Page]:
    """Randomly sample up to n distinct pages."""
    return rng.sample(list(pages), min(n, len(pages)))


def sample_page_groups(
    pages: Sequence[Page],
    n_groups: int,
    rng: random.Random,
    group_size: int = MULTI_CHUNK_PAGES,
    cross_doc_probability: float = 0.3,
) -> List[List[Page]]:
    """Sample groups of pages for multi_chunk questions.

    Each group is either pages from one document or, with probability
    cross_doc_probability, pages from different documents.
    """
    by_doc: Dict[str, List[Page]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_id, []).append(p)
    multi_page_docs = [d for d, ps in by_doc.items() if len(ps) >= group_size]

    groups: List[List[Page]] = []
    seen: set = set()
    for _ in range(n_groups * 10):  # bounded retries to avoid duplicate groups
        if len(groups) >= n_groups:
            break
        cross = rng.random() < cross_doc_probability and len(by_doc) >= group_size
        if cross or not multi_page_docs:
            if len(by_doc) < group_size:
                break
            docs = rng.sample(sorted(by_doc), group_size)
            group = [rng.choice(by_doc[d]) for d in docs]
        else:
            doc = rng.choice(sorted(multi_page_docs))
            group = rng.sample(by_doc[doc], group_size)
        group.sort(key=lambda p: (p.doc_id, p.page_number))
        key = tuple((p.doc_id, p.page_number) for p in group)
        if key not in seen:
            seen.add(key)
            groups.append(group)
    return groups


# ---------------------------------------------------------------------------
# Prompting and parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Parse a JSON object from LLM output, tolerating markdown code fences.

    Raises:
        ValueError: If no valid JSON can be parsed.
    """
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError("response is not valid JSON") from None


def parse_candidates(
    text: str, q_type: DraftType, source_pages: Sequence[Page]
) -> List[DraftCandidate]:
    """Parse and validate an LLM response into draft candidates.

    Individual items are dropped (with a warning) if they fail schema validation,
    have the wrong type, cite pages that were not supplied, or (for single-page
    types) cite more than one page.

    Raises:
        ValueError: If the response as a whole is malformed.
    """
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError("response must be an object with a 'questions' list")

    allowed: set = {(p.doc_id, p.page_number) for p in source_pages}
    accepted: List[DraftCandidate] = []
    for i, item in enumerate(data["questions"]):
        try:
            cand = DraftCandidate.model_validate(item)
        except ValidationError as exc:
            logger.warning("Dropping item %d (invalid schema): %s", i, exc.errors()[0]["msg"])
            continue
        cited = {(e.doc_id, p) for e in cand.evidence for p in e.pages}
        if cand.type != q_type:
            logger.warning("Dropping item %d: type '%s' != requested '%s'.", i, cand.type, q_type)
        elif not cited <= allowed:
            logger.warning("Dropping item %d: cites pages not supplied: %s", i, sorted(cited - allowed))
        elif q_type != "multi_chunk" and len(cited) != 1:
            logger.warning("Dropping item %d: %s must cite exactly one page.", i, q_type)
        elif q_type == "multi_chunk" and len(cited) < 2:
            logger.warning("Dropping item %d: multi_chunk must cite >= 2 pages.", i)
        elif len({e.doc_id for e in cand.evidence}) != len(cand.evidence):
            logger.warning("Dropping item %d: duplicate doc_id in evidence.", i)
        else:
            accepted.append(cand)
    return accepted


def generate_for_pages(
    client: LLMClient,
    q_type: DraftType,
    pages: Sequence[Page],
    n_questions: int,
    temperature: float,
    retries: int = 1,
) -> List[DraftCandidate]:
    """Call the LLM for one page/group; tolerate API errors and bad output.

    Returns an empty list if every attempt fails. Raises LLMUnavailableError
    only when the client is unusable (not implemented, bad credentials).
    """
    prompt = build_question_generation_prompt(q_type, pages, n_questions)
    label = ", ".join(f"{p.doc_id} p.{p.page_number}" for p in pages)
    for attempt in range(retries + 1):
        try:
            response = client.generate(
                prompt, system_instruction=QUESTION_GENERATION_SYSTEM_PROMPT, temperature=temperature
            )
        except NotImplementedError as exc:
            raise LLMUnavailableError(
                "The LLM client's generate() is not implemented."
            ) from exc
        except (LLMAuthenticationError, LLMConfigurationError) as exc:
            raise LLMUnavailableError(str(exc)) from exc
        except Exception as exc:  # provider/network errors vary by SDK
            logger.error("LLM call failed for [%s] (attempt %d): %s", label, attempt + 1, exc)
            continue
        try:
            return parse_candidates(response.text, q_type, pages)
        except ValueError as exc:
            logger.warning("Malformed response for [%s] (attempt %d): %s", label, attempt + 1, exc)
    logger.error("Giving up on [%s] (%s).", label, q_type)
    return []


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def assert_safe_output_path(path: Path) -> None:
    """Refuse any output path that is, or could be, the real gold dataset.

    Blocks by case-insensitive filename (macOS/Windows filesystems are
    case-insensitive) and by file identity, which also catches symlinks.
    """
    gold = DEFAULT_OUTPUT.parent.parent / GOLD_DATASET_FILENAME
    is_gold = path.name.lower() == GOLD_DATASET_FILENAME.lower()
    if not is_gold and path.exists() and gold.exists():
        is_gold = path.samefile(gold)
    if is_gold:
        raise ValueError(
            f"Refusing to write to '{path}': drafts must never be written to "
            f"{GOLD_DATASET_FILENAME}. Use {DEFAULT_OUTPUT}."
        )


def build_output(
    drafts: Sequence[Tuple[DraftCandidate, str]], seed: int, model_name: str
) -> Dict[str, Any]:
    """Assemble the draft file payload; every item is flagged unverified."""
    items = []
    for idx, (cand, _) in enumerate(drafts, start=1):
        items.append(
            {
                "draft_id": f"draft_{idx:03d}",
                "status": "UNVERIFIED_AI_DRAFT",
                "verified": False,
                "type": cand.type,
                "question": cand.question,
                "reference_answer": cand.reference_answer,
                "gold_evidence": [e.model_dump() for e in cand.evidence],
                "reason": cand.reason,
            }
        )
    return {
        "WARNING": VERIFICATION_WARNING,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "seed": seed,
        "model": model_name,
        "drafts": items,
    }


def write_drafts(payload: Dict[str, Any], path: Path, force: bool = False) -> None:
    """Write the payload to path, never replacing an existing file unless forced."""
    assert_safe_output_path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"'{path}' already exists (it may contain hand-edited drafts). "
            "Pass --force to overwrite or choose another --output."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_llm_client(config: AppConfig) -> LLMClient:
    """Construct the project's configured generation client (Gemini).

    Raises:
        LLMUnavailableError: If GEMINI_API_KEY is not set.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("GEMINI_API_KEY"):
        raise LLMUnavailableError(
            "No generation LLM is available. The configured provider is Gemini "
            f"(generation.model_name='{config.generation.model_name}'); set "
            "GEMINI_API_KEY in your environment or .env (see .env.example)."
        )
    from raglab.generation.llm import GeminiClient

    try:
        return GeminiClient(
            model_name=config.generation.model_name,
            max_output_tokens=config.generation.max_output_tokens,
            json_mode=True,
        )
    except LLMConfigurationError as exc:
        raise LLMUnavailableError(str(exc)) from exc


def run_generation(
    client: LLMClient,
    pages: Sequence[Page],
    seed: int,
    counts: Dict[DraftType, int],
    per_page: int,
    temperature: float,
    retries: int = 1,
) -> List[Tuple[DraftCandidate, str]]:
    """Sample pages and collect validated drafts for each requested type.

    `counts` is the number of pages (or page groups) to sample per type.
    """
    rng = random.Random(seed)
    usable = eligible_pages(pages)
    if not usable:
        raise ValueError("No pages with enough text were found in the corpus.")

    results: List[Tuple[DraftCandidate, str]] = []
    for q_type in ("factoid", "numeric_detail", "multi_chunk"):
        n = counts.get(q_type, 0)  # type: ignore[arg-type]
        if n <= 0:
            continue
        if q_type == "multi_chunk":
            samples = sample_page_groups(usable, n, rng)
        else:
            samples = [[p] for p in sample_single_pages(usable, n, rng)]
        for group in samples:
            for cand in generate_for_pages(client, q_type, group, per_page, temperature, retries):  # type: ignore[arg-type]
                results.append((cand, q_type))
        logger.info("%s: %d sample(s) processed.", q_type, len(samples))
    return results


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Generate UNVERIFIED draft gold questions.")
    p.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    p.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--factoid", type=int, default=5, help="Pages to sample for factoid drafts.")
    p.add_argument("--numeric", type=int, default=5, help="Pages to sample for numeric_detail drafts.")
    p.add_argument("--multi-chunk", type=int, default=3, help="Page groups to sample for multi_chunk drafts.")
    p.add_argument("--per-page", type=int, default=1, help="Max questions requested per sample.")
    p.add_argument("--temperature", type=float, default=None, help="Override config temperature.")
    p.add_argument("--retries", type=int, default=1, help="Retries after a malformed/failed response.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing draft file.")
    p.add_argument("--dry-run", action="store_true", help="Show sampled pages; make no LLM calls and write nothing.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, client: Optional[LLMClient] = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_arguments(argv)
    try:
        assert_safe_output_path(args.output)
        config = load_config(args.config)
        pages = load_directory(args.papers_dir)
        if not pages:
            logger.error("No pages loaded from '%s'.", args.papers_dir)
            return 1
        counts: Dict[DraftType, int] = {
            "factoid": args.factoid,
            "numeric_detail": args.numeric,
            "multi_chunk": args.multi_chunk,
        }
        temperature = config.generation.temperature if args.temperature is None else args.temperature

        if args.dry_run:
            rng = random.Random(args.seed)
            usable = eligible_pages(pages)
            for p in sample_single_pages(usable, args.factoid + args.numeric, rng):
                logger.info("single sample: %s p.%d", p.doc_id, p.page_number)
            for g in sample_page_groups(usable, args.multi_chunk, rng):
                logger.info("group sample: %s", [(p.doc_id, p.page_number) for p in g])
            return 0

        if client is None:
            client = build_llm_client(config)
        drafts = run_generation(
            client, pages, args.seed, counts, args.per_page, temperature, args.retries
        )
        if not drafts:
            logger.error("No valid drafts were produced; nothing written.")
            return 1
        payload = build_output(drafts, args.seed, config.generation.model_name)
        write_drafts(payload, args.output, force=args.force)
    except LLMUnavailableError as exc:
        logger.error("%s", exc)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Wrote %d UNVERIFIED draft(s) to %s", len(drafts), args.output)
    logger.warning(VERIFICATION_WARNING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
