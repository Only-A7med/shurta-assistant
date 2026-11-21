# Retrieval evaluation

110 questions with a known answer in the corpus, against 372 indexed chunks (embeddings: `bge-m3`).

A result counts as a hit when the retrieved chunk belongs to a service the
question names. Relevance is measured at the service level because an answer
is assembled from several sections of the same service.

## Overall (110 questions)

| Method | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword only (BM25) | 0.827 | 0.909 | 0.945 | 0.878 |
| Vector only | 0.918 | 0.955 | 0.973 | 0.942 |
| Hybrid, fused | 0.900 | 0.945 | 0.964 | 0.931 |

## Arabic (Modern Standard) (45 questions)

| Method | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword only (BM25) | 0.822 | 0.956 | 0.956 | 0.881 |
| Vector only | 0.956 | 0.956 | 0.978 | 0.960 |
| Hybrid, fused | 0.911 | 0.956 | 0.956 | 0.940 |

## Code-switched (5 questions)

| Method | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword only (BM25) | 1.000 | 1.000 | 1.000 | 1.000 |
| Vector only | 0.800 | 0.800 | 0.800 | 0.829 |
| Hybrid, fused | 1.000 | 1.000 | 1.000 | 1.000 |

## English (45 questions)

| Method | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword only (BM25) | 0.800 | 0.844 | 0.933 | 0.851 |
| Vector only | 0.911 | 0.956 | 0.978 | 0.939 |
| Hybrid, fused | 0.911 | 0.933 | 0.956 | 0.933 |

## Gulf dialect (15 questions)

| Method | recall@1 | recall@3 | recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword only (BM25) | 0.867 | 0.933 | 0.933 | 0.910 |
| Vector only | 0.867 | 1.000 | 1.000 | 0.933 |
| Hybrid, fused | 0.800 | 0.933 | 1.000 | 0.872 |

## Reading these numbers

Vector search alone is ahead at rank 1: 0.918 against the fused 0.900, which is 2 questions in 110.

The fused ranking is kept because keyword matching holds on to exact terms the embedding can miss: a service name typed in Latin script inside an Arabic sentence, or a figure. It reaches recall@1 of 1.000 on the code-switched group, 5 questions, where vector search alone reaches 0.800.

Gulf dialect is the one group where fusion scores below both methods on their own: recall@1 0.800 against 0.867 for both, over 15 questions, recovering to 1.000 by rank 5. One question moves that group by 0.067, so it is a direction to look at rather than a measured conclusion.

Weighted fusion is a possible next step rather than a measured one. The weights are equal and were not tuned, because there is no held-out set to tune them on: tuning them on the same 110 questions these figures come from would fit the weights to the numbers published beside them.
