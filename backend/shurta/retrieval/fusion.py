"""Reciprocal rank fusion.

Two rankings have to be combined, and their scores are not comparable: BM25
returns an unbounded sum of term weights, cosine similarity a number between
minus one and one. Normalising either onto the other's scale means inventing a
mapping, and the mapping would change with the corpus.

Reciprocal rank fusion sidesteps that by throwing the scores away and keeping
only the positions. A document scores ``1 / (k + rank)`` in each ranking it
appears in, and the contributions are added. So a passage both methods place
near the top wins, a passage only one method found still places, and the
arithmetic does not care that the two methods measure different things.

``k`` flattens the curve near the top. At the standard 60, first and second
place differ by about 1.6 per cent, so one ranking being slightly more confident
does not override agreement between the two.

Each ranking can also carry a weight. The default is equal weighting, and
measurement is the reason it stayed that way. Vector search is the stronger
ranking overall on this corpus — recall@1 of 0.91 against keyword search's 0.81
— which suggested down-weighting the keyword side. Sweeping the weight showed
the opposite: lowering it changed the overall figures by less than one question
in a hundred, and it cost the code-switched questions outright, where keyword
search is the only method that finds a Latin-script service name typed inside an
Arabic sentence. So the weights exist, and they are set to one.
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
