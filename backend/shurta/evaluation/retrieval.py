"""Measuring retrieval: recall at 1, 3 and 5, and mean reciprocal rank.

A chunk counts as relevant when its service is one the question names, which is
the right granularity: an answer is assembled from several sections of a service.

Keyword only, vector only and the two fused are measured on the same questions,
so what each contributes is visible.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from shurta.chunking import Chunk
from shurta.config import REPOSITORY_ROOT, Settings
from shurta.evaluation.dataset import Question, load_questions
from shurta.llm.ollama import OllamaClient
from shurta.models import Language
from shurta.retrieval.hybrid import HybridRetriever
from shurta.retrieval.store import SearchIndex

CUTOFFS = (1, 3, 5)
RESULTS_DIR = REPOSITORY_ROOT / "eval"

METHODS = ("bm25", "dense", "hybrid")

METHOD_NAMES = {
    "bm25": "Keyword only (BM25)",
    "dense": "Vector only",
    "hybrid": "Hybrid, fused",
}

GROUP_NAMES = {
    "overall": "Overall",
    "en": "English",
    "ar": "Arabic (Modern Standard)",
    "gulf": "Gulf dialect",
    "code_switched": "Code-switched",
}


@dataclass
class Scores:
    """Accumulated metrics for one method over one group of questions."""

    total: int = 0
    hits: dict[int, int] = field(default_factory=lambda: dict.fromkeys(CUTOFFS, 0))
    reciprocal_ranks: list[float] = field(default_factory=list)

    def add(self, ranked_service_ids: Sequence[str], expected: Sequence[str]) -> None:
        self.total += 1
        wanted = set(expected)

        first = next(
            (
                position
                for position, service_id in enumerate(ranked_service_ids)
                if service_id in wanted
            ),
            None,
        )
        for cutoff in CUTOFFS:
            if first is not None and first < cutoff:
                self.hits[cutoff] += 1
        self.reciprocal_ranks.append(0.0 if first is None else 1.0 / (first + 1))

    def recall(self, cutoff: int) -> float:
        return self.hits[cutoff] / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return (
            sum(self.reciprocal_ranks) / len(self.reciprocal_ranks)
            if self.reciprocal_ranks
            else 0.0
        )

    def to_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {"questions": self.total}
        for cutoff in CUTOFFS:
            result[f"recall@{cutoff}"] = round(self.recall(cutoff), 4)
        result["mrr"] = round(self.mrr, 4)
        return result


def _service_order(chunks: Sequence[Chunk]) -> list[str]:
    """The services the chunks point at, in rank order and without repeats.

    Several chunks of one service may be retrieved; for recall the service's
    best position is what counts, so duplicates are collapsed.
    """
    seen: list[str] = []
    for chunk in chunks:
        if chunk.service_id not in seen:
            seen.append(chunk.service_id)
    return seen


def _group(question: Question) -> str:
    return question.dialect or question.language.value


async def evaluate_retrieval(
    retriever: HybridRetriever,
    questions: Sequence[Question],
    *,
    depth: int = max(CUTOFFS),
) -> dict[str, dict[str, Scores]]:
    """Score each method, overall and per language group."""
    scores: dict[str, dict[str, Scores]] = {method: {"overall": Scores()} for method in METHODS}

    for question in questions:
        if not question.is_retrieval_item:
            continue

        language = Language(question.language)
        rankings = {
            "bm25": _service_order(
                await retriever.search_lexical_only(question.question, top_k=depth * 3)
            ),
            "dense": _service_order(
                await retriever.search_dense_only(question.question, top_k=depth * 3)
            ),
            "hybrid": _service_order(
                [
                    retrieved.chunk
                    for retrieved in (
                        await retriever.search(
                            question.question, top_k=depth * 3, language=language
                        )
                    ).chunks
                ]
            ),
        }

        group = _group(question)
        for method, ranked in rankings.items():
            buckets = scores[method]
            buckets.setdefault(group, Scores())
            buckets["overall"].add(ranked, question.expected_service_ids)
            buckets[group].add(ranked, question.expected_service_ids)

    return scores


def as_json(
    scores: dict[str, dict[str, Scores]], *, index: SearchIndex, question_count: int
) -> dict[str, object]:
    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "embedding_model": index.metadata.embedding_model,
        "chunks_indexed": index.metadata.chunk_count,
        "questions_scored": question_count,
        "methods": {
            method: {group: bucket.to_dict() for group, bucket in groups.items()}
            for method, groups in scores.items()
        },
    }


def as_markdown(
    scores: dict[str, dict[str, Scores]], *, index: SearchIndex, question_count: int
) -> str:
    groups = ["overall", *sorted(set(scores["hybrid"]) - {"overall"})]

    lines = [
        "# Retrieval evaluation",
        "",
        f"{question_count} questions with a known answer in the corpus, against "
        f"{index.metadata.chunk_count} indexed chunks "
        f"(embeddings: `{index.metadata.embedding_model}`).",
        "",
        "A result counts as a hit when the retrieved chunk belongs to a service the",
        "question names. Relevance is measured at the service level because an answer",
        "is assembled from several sections of the same service.",
        "",
    ]

    interpretation = _interpretation(scores)

    for group in groups:
        rows = [(method, scores[method][group]) for method in METHODS if group in scores[method]]
        if not rows or rows[0][1].total == 0:
            continue
        lines += [
            f"## {GROUP_NAMES.get(group, group)} ({rows[0][1].total} questions)",
            "",
            "| Method | recall@1 | recall@3 | recall@5 | MRR |",
            "|---|---:|---:|---:|---:|",
        ]
        for method, bucket in rows:
            lines.append(
                f"| {METHOD_NAMES[method]} | {bucket.recall(1):.3f} | {bucket.recall(3):.3f} "
                f"| {bucket.recall(5):.3f} | {bucket.mrr:.3f} |"
            )
        lines.append("")

    if interpretation:
        lines += ["## Reading these numbers", "", interpretation, ""]

    return "\n".join(lines)


def _single_methods_both_ahead(
    scores: dict[str, dict[str, Scores]],
) -> list[tuple[str, Scores, Scores, Scores]]:
    """Groups where the fused ranking is below keyword *and* vector search.

    The shipped ranking being the worst of the three somewhere is the thing a
    reader most needs told, so it is found from the numbers rather than
    remembered from the run that first showed it.
    """
    losing: list[tuple[str, Scores, Scores, Scores]] = []
    for group, hybrid in scores["hybrid"].items():
        if group == "overall" or not hybrid.total:
            continue
        bm25 = scores["bm25"].get(group)
        dense = scores["dense"].get(group)
        if bm25 is None or dense is None:
            continue
        if hybrid.recall(1) < bm25.recall(1) and hybrid.recall(1) < dense.recall(1):
            losing.append((group, hybrid, bm25, dense))
    return losing


def _rivals(bm25: Scores, dense: Scores) -> str:
    """The two single methods' recall@1, said once when they agree."""
    if bm25.recall(1) == dense.recall(1):
        return f"{bm25.recall(1):.3f} for both"
    return f"{bm25.recall(1):.3f} for keyword search and {dense.recall(1):.3f} for vector search"


def _interpretation(scores: dict[str, dict[str, Scores]]) -> str:
    """Say plainly which method won, including where the default loses.

    Generated from the measured numbers rather than written once by hand, so it
    cannot drift away from the table above it.
    """
    dense = scores["dense"]["overall"]
    hybrid = scores["hybrid"]["overall"]
    if not dense.total:
        return ""

    paragraphs: list[str] = []
    gap = dense.recall(1) - hybrid.recall(1)
    if gap > 0:
        questions = round(gap * dense.total)
        paragraphs.append(
            f"Vector search alone is ahead at rank 1: {dense.recall(1):.3f} against the fused "
            f"{hybrid.recall(1):.3f}, which is {questions} "
            f"question{'s' if questions != 1 else ''} in {dense.total}."
        )

    switched = scores.get("hybrid", {}).get("code_switched")
    switched_dense = scores.get("dense", {}).get("code_switched")
    if switched and switched_dense and switched.recall(1) > switched_dense.recall(1):
        paragraphs.append(
            "The fused ranking is kept because keyword matching holds on to exact terms the "
            "embedding can miss: a service name typed in Latin script inside an Arabic "
            f"sentence, or a figure. It reaches recall@1 of {switched.recall(1):.3f} on the "
            f"code-switched group, {switched.total} questions, where vector search alone "
            f"reaches {switched_dense.recall(1):.3f}."
        )

    losing = _single_methods_both_ahead(scores)
    if len(losing) == 1:
        group, fused, bm25, single = losing[0]
        paragraphs.append(
            f"{GROUP_NAMES.get(group, group)} is the one group where fusion scores below both "
            f"methods on their own: recall@1 {fused.recall(1):.3f} against "
            f"{_rivals(bm25, single)}, over {fused.total} questions, recovering to "
            f"{fused.recall(5):.3f} by rank 5. One question moves that group by "
            f"{1 / fused.total:.3f}, so it is a direction to look at rather than a measured "
            "conclusion."
        )
    elif losing:
        listed = "; ".join(
            f"{GROUP_NAMES.get(group, group)} {fused.recall(1):.3f} against "
            f"{_rivals(bm25, single)} over {fused.total} questions"
            for group, fused, bm25, single in losing
        )
        paragraphs.append(
            f"Fusion scores below both methods on their own in {len(losing)} groups: {listed}."
        )

    paragraphs.append(
        "Weighted fusion is a possible next step rather than a measured one. The weights are "
        "equal and were not tuned, because there is no held-out set to tune them on: tuning "
        f"them on the same {dense.total} questions these figures come from would fit the "
        "weights to the numbers published beside them."
    )

    return "\n\n".join(paragraphs)


def main(settings: Settings) -> int:
    """Run the retrieval evaluation and write the results."""
    try:
        index = SearchIndex.load(settings.index_path)
    except RuntimeError as error:
        print(f"Cannot evaluate retrieval: {error}")
        return 1

    questions = load_questions()
    scored = [question for question in questions if question.is_retrieval_item]

    async def run() -> dict[str, dict[str, Scores]]:
        client = OllamaClient(
            settings.ollama_base_url,
            generation_model=settings.generation_model,
            embedding_model=settings.embedding_model,
            embedding_timeout=settings.embedding_timeout,
            keep_alive=settings.keep_alive,
        )

        async def embed(query: str) -> list[float]:
            return (await client.embed([query]))[0]

        try:
            return await evaluate_retrieval(HybridRetriever(index, embed), questions)
        finally:
            await client.close()

    scores = asyncio.run(run())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "retrieval-results.json"
    markdown_path = RESULTS_DIR / "retrieval-results.md"
    payload = as_json(scores, index=index, question_count=len(scored))
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    markdown_path.write_text(
        as_markdown(scores, index=index, question_count=len(scored)), encoding="utf-8"
    )

    overall = scores["hybrid"]["overall"]
    print(f"Scored {len(scored)} questions on {index.metadata.chunk_count} chunks.")
    for method in METHODS:
        bucket = scores[method]["overall"]
        print(
            f"  {METHOD_NAMES[method]:24} recall@1 {bucket.recall(1):.3f}  "
            f"recall@3 {bucket.recall(3):.3f}  recall@5 {bucket.recall(5):.3f}  "
            f"MRR {bucket.mrr:.3f}"
        )
    print(f"Hybrid recall@5 {overall.recall(5):.3f} -> {markdown_path}")
    return 0
