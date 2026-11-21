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

Vector search alone has the better overall recall@1: 0.918 against the fused 0.900, a difference of 2 questions out of 110.

The fused ranking earns its place on the code-switched questions, where it reaches recall@1 of 1.000 against vector search's 0.800: a service name typed in Latin script inside an Arabic sentence is matched by keyword search and missed by the embedding. Losing a whole class of question is worse than losing a point of overall accuracy, so the fused ranking is the one the assistant uses.

The weakest group is Gulf dialect, at recall@1 0.800 over 15 questions, recovering to 1.000 by rank 5. The sample is small enough that each question moves the figure a long way, so this is a direction to investigate rather than a measured conclusion.
