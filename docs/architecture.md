# Architecture

This explains how a question becomes a cited answer, and why each part is built the way it is.
It is written for somebody reading the repository for the first time.

## The shape of the problem

A reader asks, in Arabic or English, what a public service costs, what documents it needs, or how
long it takes. The answer exists on an official page. The difficulty is not generating prose — it is
making sure the answer contains the *right fee*, says so in the *right language*, and shows where it
came from, while running on an ordinary machine with no graphics card and a few gigabytes of memory
for the whole stack.

Those constraints drive almost every decision below.

## The pipeline

```mermaid
flowchart TD
    Q["Question<br/>(Arabic or English)"] --> G{"Guardrails<br/>before any model call"}

    G -->|emergency| E["Fixed response:<br/>999 / 998 / 997"]
    G -->|"asks about own record"| P["Decline, name the<br/>official channel"]
    G -->|ordinary| N["Detect language"]

    N --> R["Hybrid retrieval"]
    R --> B["BM25 over<br/>Arabic-normalised text"]
    R --> D["Cosine similarity<br/>over bge-m3 vectors"]
    B --> F["Reciprocal rank fusion"]
    D --> F
    F --> T{"Closest passage<br/>similar enough?"}

    T -->|no| X["Fixed response:<br/>not in my information"]
    T -->|yes| C["Top 4 passages,<br/>numbered"]
    C --> M["Generation model<br/>streams the answer"]
    M --> A["Answer + citations<br/>+ timings"]

    E --> OUT["Server-sent events"]
    P --> OUT
    X --> OUT
    A --> OUT
```

Three of the five possible outcomes never reach a model at all. That is deliberate: each one is a
case where a model could only make things worse.

## Collection

The official service pages are rendered by a browser-side application. Fetching a page route
returns a shell — 44 kilobytes of markup yielding about 300 characters of visible text — so reading
the pages directly is useless without a headless browser.

The site serves the same content through its own public content interface as structured JSON, in
both languages. Reading that is better on every axis: no browser automation to install or maintain,
a fraction of the bandwidth, far less load on the site, and the content arrives already separated
into description, fees, documents, channels and so on, rather than needing to be parsed back out of
rendered HTML.

The collector is deliberately slow and forgetful of nothing:

- it identifies itself in its user agent, with a link to the project;
- it waits at least two seconds between requests, no matter how fast responses come back;
- it writes every response to `data/raw/` and reads from there next time, so an interrupted run
  resumes and developing the normalisation step costs no further requests;
- it refuses a response that is not JSON, because the site answers requests its edge rejects with
  an HTML error page under a `200` status, which would otherwise be stored as though it were content.

**The repository contains no collected content.** The official terms reserve all rights over it, so
the knowledge base is built on your own machine by `python -m shurta ingest` and both `data/raw/`
and `data/processed/` are untracked. What is committed is the collector, and a small invented
catalogue for a fictional city police force that the tests and continuous integration run against.

### What the schema leaves empty

Each service is normalised into a fixed shape: names, category, description, eligibility, required
documents, fees, processing time, channels, steps, per-language source URLs, and a collection date.
A field the source does not state is left **empty** rather than filled in. `python -m shurta ingest`
writes a coverage report next to the catalogue so the gaps are visible: in the current collection,
descriptions and processing times are complete, required documents cover about two services in five,
fees about one in four, and `steps` is empty for every service because no step-by-step procedure is
published for any of them.

That last one is worth stating plainly: a permanently empty field looks like a bug. It is not. The
alternative — approximating steps from the channel text — would be inventing procedure, which is
the one thing this project must not do.

## Arabic normalisation

Arabic writes the same word several ways, and none of the differences carry meaning:

| Written | Also written | What differs |
|---|---|---|
| `الإقامة` | `الاقامة` | alef with hamza, or bare alef |
| `شهادة` | `شهاده` | ta marbuta, or ha |
| `دعوى` | `دعوي` | alef maqsura, or ya |
| `شــهادة` | `شهادة` | tatweel, a justification stretch |
| `شَهَادَة` | `شهادة` | vowel marks |
| `٥٠٠` | `500` | Arabic-Indic digits |

A reader treats all of these as the same word. Exact token matching does not. So both the indexed
text and the query pass through one normaliser that removes diacritics and tatweel, unifies the
alef forms, folds ta marbuta onto ha and alef maqsura onto ya, converts Arabic-Indic digits to
Western ones, and strips the definite article from the front of a token so that `الشهادة` matches
`شهادة`.

The transformation is lossy, and it is applied **only** to the text used for matching. What the
reader sees and what goes into the prompt is the source's own wording, untouched.

Two details are easy to get wrong and are tested:

- The article is only stripped when enough of the word remains. Taking `ال` off a three-letter word
  leaves a meaningless stub.
- Matching happens on *tokens*, not substrings. An early version matched Arabic phrases as
  substrings, and the word for "thief" (`لص`) is a substring of the word for "validity"
  (`الصلاحية`) — which turned a question about an expired identity card into an emergency.

Language detection uses the same normaliser and leans deliberately towards Arabic: a question is
treated as Arabic once a fifth of its letters are Arabic script. A code-switched question such as
"ابغى أسوي police clearance certificate" is mostly Latin letters by count, but a reader who writes
any Arabic expects an Arabic answer.

## Chunking

One chunk is **one section of one service in one language** — the fees of the police clearance
certificate in English, its required documents in Arabic.

Coarser would be wasteful: putting a whole service in one chunk means a question about a fee drags
the eligibility conditions, the channel list and the opening hours into the prompt with it. On
processor-only inference that is paid for twice, in memory and in time to first token.

Finer would be useless: a lone fee line, "Resident: AED 200", cannot be answered from without
knowing which service and which fee it belongs to.

Every chunk carries its service's name **in both languages**, its section heading in its own
language, its source URL and its collection date. The bilingual naming costs a few tokens and is
what lets an Arabic question land on an English passage.

## Retrieval

Two methods run, and each covers the other's failure.

**BM25** matches the words actually typed. It is exact about a figure, a service name or an
identifier, and it is the only one of the two that reliably finds a rare term. It cannot match a
question to a passage that says the same thing in different words, and it cannot cross languages at
all: Arabic `الرسوم` shares no token with English "fees".

**Cosine similarity over `bge-m3` embeddings** crosses both gaps, because the model places text
with the same meaning near each other regardless of language. It is weaker on a specific string — it
will happily rank a passage about a different certificate close to the right one.

BM25 is written out in the repository rather than taken from a library, for one reason: the
tokeniser has to be the Arabic-normalising one above, on both the passages and the query, or Arabic
keyword search barely works. Once the tokeniser is yours, the rest of BM25 is about forty lines, and
keeping it here makes it directly testable.

### Why reciprocal rank fusion

The two rankings have incomparable scores: BM25 returns an unbounded sum of term weights, cosine
similarity a number between minus one and one. Normalising either onto the other's scale means
inventing a mapping, and the mapping would change with the corpus.

Reciprocal rank fusion throws the scores away and keeps only the positions. A passage scores
`1 / (k + rank)` in each ranking it appears in, and the contributions are added. A passage both
methods rank highly wins; a passage only one found still places; and the arithmetic does not care
that the two methods measure different things. With `k` at the standard 60, first and second place
differ by under two per cent, so one method being slightly more confident does not override
agreement between both.

### The trade-off that was measured, not assumed

Fusion is not free. Measured over 100 questions with a known answer:

| Method | recall@1 | recall@5 | Code-switched recall@1 |
|---|---:|---:|---:|
| Keyword only | 0.810 | 0.940 | 1.000 |
| Vector only | 0.910 | 0.970 | 0.800 |
| Fused | 0.890 | 0.960 | 1.000 |

Vector search alone is **better overall** — by two questions in a hundred. The fused ranking is
still the default because it is the only configuration that answers every code-switched question
correctly, where vector search alone misses one in five. A service name typed in Latin script inside
an Arabic sentence is found by keyword search and missed by the embedding. Losing a whole class of
question is worse than losing a point of overall accuracy.

Weighting the fusion towards the stronger vector ranking was tried and reverted: sweeping the weight
moved the overall figures by less than one question in a hundred and cost the code-switched case
outright. The weights exist in the code, and they are set to one.

A small preference is applied for passages in the reader's own language. Both languages are always
searched — a question in one may be answered by a passage in the other — but when two passages say
the same thing, the one in the reader's language is the better citation.

### Why no vector database

The corpus is a few hundred passages. The dense side is a single NumPy array of about a megabyte,
and an exhaustive cosine similarity over it is a matrix multiply that finishes in well under a
millisecond. A vector database server would add a process to run, a protocol to speak and a failure
mode to handle, in exchange for approximate search the corpus is far too small to need.

An index is three files: `vectors.npy` (unit-length rows, so cosine similarity is a dot product),
`chunks.json` in the same order, and `index.json` holding the lexical index and the build metadata.
The metadata records which embedding model built it, and `doctor` refuses an index built with a
different one — vectors from two models are not comparable, and the resulting nonsense would be
silent.

End to end, retrieval takes about 50 to 70 milliseconds once the embedding model is resident. All
of the perceptible wait is generation.

## Guardrails

Two kinds of question are answered without a model, because a model could only make them worse.

An **emergency** needs a telephone number in the first line. On processor-only inference the first
token can be ten seconds away and the answer could be wrong, so an emergency intent short-circuits
the entire pipeline and returns a fixed response naming 999, 998 and 997. The detector is tuned to
fire readily: a false positive costs a reader one unnecessary line of emergency numbers, which is a
far smaller harm than a missed emergency. What stops it firing constantly is an override for
phrasings that ask *about* a service — "how do I report a robbery?" names a robbery but is a
question about a reporting service.

A **personal-case lookup** — "what is the status of my fine?" — cannot be answered from published
information by anybody, because the answer lives in a record this project has no access to and
should not have. These are declined, with the official channel named. The detection needs two things
together: a reference to the reader's own record *and* a request for a value from it. Either alone
is ordinary, which is why "can I pay my traffic fines in instalments?" is answered normally — it
mentions the reader's own fines but asks how the service works.

All 120 evaluation questions, including the emergency and personal-case ones, are classified as
intended.

## The relevance floor

When nothing retrieved is close enough, a fixed "not in my information" response is returned instead
of asking a model to work from an empty context — which is where a model of this size starts
inventing.

Choosing the signal for this took a measurement and a correction. The obvious candidate, the fused
retrieval score, **does not work**: it is built from rank positions alone, so the top passage scores
near the maximum whether or not it is relevant. Measured, out-of-scope questions scored *higher* on
it than most answerable ones.

Raw cosine similarity is an absolute measure and does separate them:

| | lowest | median | highest |
|---|---:|---:|---:|
| Answerable questions (100) | 0.526 | 0.697 | 0.804 |
| Out-of-scope questions (6) | 0.355 | 0.471 | 0.566 |

The distributions overlap, and honestly so — "how do I renew my driving licence" really is close to
the traffic services in the corpus. So the floor is set at 0.50, *below* every answerable question
rather than between the two groups. It rejects nothing the catalogue can answer and catches four of
the six out-of-scope questions before any model runs. The remaining two are the genuinely adjacent
government services, and those are left to the prompt, which can see the passages and say they cover
something else. A higher floor would start refusing questions the catalogue does answer, which is
the worse failure.

## Generation

The prompt has one job: get a small model to restate retrieved text accurately, in the reader's
language, with a citation against each claim, and to say it does not know rather than fill a gap.

Three things in it are shaped by the model being small:

- **The rules are few and ordered by importance.** A four-billion-parameter model follows the first
  few lines of an instruction block and drifts from the rest, so "use only the passages", "answer in
  this language" and "cite" come first.
- **The passages are numbered and those numbers are what it is asked for.** Asking for a citation it
  would have to invent invites invention; asking it to copy a number that is in front of it is a
  copying task, which these models are reliable at.
- **The language instruction appears twice**, once at the top and once immediately before the answer.
  Otherwise an Arabic question gets an English answer whenever the retrieved passages are English.

### Reasoning, and why the default model does not do any

Some instruction models reason before answering. The provider reports that reasoning in a separate
field and the answer in another — but its `think` request parameter does **not** switch the reasoning
off. It switches off the *separation*, after which the reasoning arrives in the answer field and is
indistinguishable from the answer. Sending `think: false` to a model whose reasoning cannot be
disabled makes things actively worse: answers opened with "Okay, the user is asking…" and hit the
token cap before saying anything useful.

So the client never sends that parameter. It lets the provider separate the two and uses only the
answer — correct for a model that reasons, a no-op for one that does not.

That leaves the real lever: choosing a model whose reasoning does not have to be paid for. Both
candidates produce the same correct answer to "how much does a police clearance certificate cost for
a resident?" — `AED 200 [1]`. One takes about eleven seconds to the first token; the other takes
about fifty, spending the difference on reasoning the reader never sees. `eval/results.md` has the
full comparison and the measured basis for the default.

### The rest of the budget

Everything else about generation exists to keep a small machine responsive: the top four passages
only, a 4096-token context, a capped answer length, tokens streamed so the reader sees text as it
arrives rather than after the whole answer, the model warmed at start-up so the first question does
not pay to load it, and `keep_alive` set so it stays resident between questions.

## The interface

`POST /api/chat` streams server-sent events in a fixed order: a `start` event carrying the detected
language and the citations — sent *before* any generation, so the interface can set its text
direction and show the right state immediately — then `token` events, then a `done` event with the
timings, or an `error` event carrying whatever was produced before the failure.

The event framing splits a payload across several `data:` lines. Without that, every answer would
truncate at its first newline, and answers are full of newlines because every list has them.

`GET /api/health` reports each part separately — provider reachable, each model pulled, index
loaded, catalogue present — and names the command that fixes whatever is missing. Start-up
deliberately does not fail when the index or the provider is absent: somebody who has just cloned
the repository should get an API that starts and a health page that explains itself, not a stack
trace at boot.

The model client is reached only through a small interface, which is how the whole API is tested
against a stand-in with no provider running, and how a second provider would arrive.

## Privacy

Nothing a reader types is stored. There is no session, no identifier and no history. The question is
held in memory for the length of the request and is not written to any log unless `LOG_QUESTIONS` is
turned on deliberately, which it is not by default. A test asserts that the response does not even
echo the question back. Generation and embedding both run on the machine the assistant is installed
on, so no question leaves it.

## What this design is not good at

- **Freshness.** The catalogue is a snapshot. A fee that changed on the official page yesterday is
  wrong here until the next `ingest`. Every answer carries a collection date for that reason.
- **Anything the source does not state.** Procedure steps are published for no service, so the
  assistant cannot walk somebody through an application.
- **Gulf dialect, relative to the other groups.** It is the weakest group measured, recovering by
  rank 5 but not reliably at rank 1. The sample is five questions, so this is a direction to
  investigate rather than a measured conclusion.
- **Reasoning across services.** Retrieval finds passages; it does not compare two services or work
  out a consequence. A question whose answer requires combining three services will get the closest
  passages and an answer limited to them.
