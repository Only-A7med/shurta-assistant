"""Reciprocal rank fusion.

BM25 scores and cosine similarities are not comparable, so the scores are thrown
away and only the positions kept: a document scores ``1 / (k + rank)`` in each
ranking it appears in, and the contributions are added.

``k`` flattens the curve near the top; at the standard 60, first and second place
differ by about 1.6 per cent. Weights default to equal — see
docs/architecture.md for the sweep that kept them there.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Fuse rankings of document indexes into one, best first.

    Each input is a sequence of document indexes in rank order. ``weights``
    scales each ranking's contribution and defaults to equal weighting. The
    result is ``(index, fused score)`` pairs, highest first, with ties broken by
    the document index so the output is deterministic.
    """
    ordered = list(rankings)
    if weights is None:
        weights = [1.0] * len(ordered)
    elif len(weights) != len(ordered):
        raise ValueError(f"got {len(weights)} weights for {len(ordered)} rankings; they must match")

    totals: dict[int, float] = {}
    for ranking, weight in zip(ordered, weights, strict=True):
        for position, document in enumerate(ranking):
            totals[document] = totals.get(document, 0.0) + weight / (k + position + 1)

    fused = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return fused[:limit] if limit is not None else fused
