# Fixture corpus

This directory holds an **invented** service catalogue for a fictional city police force in a
fictional emirate, written for this repository. None of it is real, and none of it comes from any
official website.

It exists because the real knowledge base is not part of this repository — it is built on your own
machine from the official public site with `python -m shurta ingest`, and is deliberately not
redistributed here. Without a committed corpus, the test suite and continuous integration would
have nothing to run against, and a fresh clone could not be checked end to end.

So everything that needs a corpus but must not depend on official content uses this one:

- the chunking, indexing and retrieval unit tests,
- the fixture-based end-to-end evaluation that runs in continuous integration,
- any local experiment where the real content is not the point.

The shape is identical to `data/processed/services.json`, so anything that works here works against
the real corpus. The numbers, durations and document lists are chosen to be obviously fictional
while still exercising the same fields: fees in several tiers, nested document conditions, instant
and multi-day processing times, and a service with fields genuinely left empty.
