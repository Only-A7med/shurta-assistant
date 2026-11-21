# Answer evaluation

Measured on 2026-11-21 over 372 indexed passages, on processor-only inference.

Every figure is measured by rule, not judged by a model. Amounts, durations and
document phrases are compared against what the cited service states, on
Arabic-normalised text so that a fee in Arabic-Indic digits matches one in
Western digits.

## Accuracy

All 2 models below marked with a plain question count were scored on the **same 130 questions**, so their figures are directly comparable. A row marked *(subset)* was scored on fewer and is not comparable with them.

| Model | Questions | Key facts recalled | All facts present | Citations valid | Citations present | Answer language | Behaviour |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma3:4b` | 130 | 0.726 | 0.583 | 1.000 | 0.955 | 1.000 | 0.992 |
| `qwen3:4b-instruct-2507-q4_K_M` | 130 | 0.663 | 0.450 | 1.000 | 0.973 | 0.985 | 0.992 |
| `qwen3:4b` | 20 (subset) | 0.262 | 0.250 | 1.000 | 0.400 | 0.500 | 1.000 |

## Speed

| Model | Prompt tokens p50 | Prompt processing p50 | First token p50 | First token p95 | Full answer p50 | Full answer p95 | Tokens per second |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma3:4b` | 604 | 9.2 s | 9.2 s | 18.7 s | 13.0 s | 28.3 s | 10.2 |
| `qwen3:4b-instruct-2507-q4_K_M` | 656 | 9.7 s | 9.7 s | 20.2 s | 13.1 s | 30.6 s | 7.9 |
| `qwen3:4b` | 662 | 10.1 s | 39.8 s | 67.5 s | 66.9 s | 92.4 s | 8.7 |

- `qwen3:4b`: measured on the 20-question latency subset only, so not comparable with the full runs above: it is a reasoning-only build whose reasoning cannot be switched off, and half its answers came back empty because the whole token budget went on reasoning the reader never sees.

## Time to first token: what was tried

All of the perceptible wait is the model reading the prompt. On the final
streamed event the provider reports how many prompt tokens it processed and how
long that took, and prompt processing turned out to be **99 per cent of the time
to first token**: 706 prompt tokens taking about 15 seconds, while loading the
model costs 17 milliseconds and retrieval 50 to 70 milliseconds.

Every change below was measured on the same fixed 20-question subset — mixed
English, Modern Standard Arabic, Gulf dialect and code-switched, every item
answerable and carrying checkable facts — with `gemma3:4b`:

```
python -m shurta eval answers --model gemma3:4b --subset latency --force
```

| Change | Prompt tokens p50 | First token p50 | Key facts | Citations | Language | Kept |
|---|---:|---:|---:|---:|---:|:--|
| Baseline | 706 | 15.0 s | 0.571 | 1.000 | 1.000 | — |
| Baseline, repeated | 706 | 18.3 s | 0.571 | 1.000 | 1.000 | — |
| Passages capped at 500 characters | 638 | 8.0 s | 0.524 | 1.000 | 1.000 | no |
| `top_k` 3 instead of 4 | 576 | 8.2 s | 0.500 | 1.000 | 1.000 | no |
| 4 generation threads | 706 | 20.2 s | 0.595 | 1.000 | 1.000 | no |
| 8 generation threads | 706 | 17.3 s | 0.571 | 1.000 | 1.000 | no |
| 12 generation threads | 706 | 13.6 s | 0.595 | 1.000 | 1.000 | optional |
| 16 generation threads | 706 | 13.9 s | 0.571 | 1.000 | 1.000 | no |
| Cap 500 + `top_k` 3 + 12 threads | 529 | 11.3 s | 0.500 | 1.000 | 1.000 | no |
| Cap 500 + 12 threads | 638 | 11.8 s | 0.500 | 1.000 | 1.000 | no |
| **Service named once per group** | **617** | **13.6 s** | **0.571** | **1.000** | **1.000** | **yes** |
| Service named once + 12 threads | 617 | 11.4 s | 0.571 | 1.000 | 1.000 | optional |

### What was kept, and what was reverted

**Kept: the service is named once per group of passages, not once per passage.**
A chunk's text repeats both of its service's names and its category, which is
what lets an Arabic question retrieve an English passage. In the prompt that is
redundant, so passages are now grouped by service and the service is named once,
in the reader's language. The median prompt falls by 89 tokens with no
information lost and no measured change in accuracy.

**Kept: an unchanged prompt prefix.** The rules block was already first and
byte-identical between requests in one language, which lets the provider reuse
its cached prefix. Measured directly against the provider with a 170-token
prefix: 9.2 seconds with the prefix first against 13.1 seconds with the same
block moved to the end, and 0.5 seconds when a prompt is repeated exactly. A
test now asserts the prefix is stable, because moving anything variable above
the rules would silently cost seconds per question.

**Kept: the models stay resident.** `keep_alive` is sent with generation *and*
embedding requests, and start-up warms both. `ollama ps` confirms neither is
reloaded between questions, and the reported load time per question is about 17
milliseconds.

**Reverted: capping passage length.** A 500-character cap saved 68 tokens and
cost key-fact recall in both runs that used it (0.524 and 0.500 against 0.571
unchanged). The tail of a fee or document list is sometimes the part the question
was about. The cap is still available as `MAX_PASSAGE_CHARS`, unset by default.

**Reverted: `top_k` 3.** Two runs with three passages both scored 0.500 on key
facts against a mean of 0.571 with four. A question is often answered by two
sections of one service, and the fourth passage is what carries the second one.

**Optional, not a default: the generation thread count.** 12 threads was the
fastest setting measured and 4 the slowest, a difference of 6.6 seconds with no
change in accuracy. It is left unset so the provider chooses, because the right
number is the machine's physical core count and a value hard-coded here would be
wrong on a smaller machine. Set `NUM_THREAD` to tune it.

### Was the target reached?

**No.** The target was under 4 seconds to the first token, and the honest best
without losing accuracy is **13.6 seconds at the shipped defaults**, or **11.4
seconds** with the thread count tuned — down from 15.0 to 18.3 seconds.

The arithmetic says why. Prompt processing runs at roughly 45 to 50 tokens per
second here, so a 4-second budget buys about 200 tokens. The rules block alone is
about 230 tokens — free after the first question in a language, because it is
cached, but the four passages and the question are another 390. Reaching 200
uncached tokens means one passage, or a truncated one, and both were measured to
cost key-fact recall. Under 4 seconds is reachable on this inference setup only
by answering worse.

Two caveats on these numbers. Run-to-run variance is large: identical settings produced 15.0 and 18.3 seconds, and two
early runs reached 8.0 seconds at a prompt-processing rate never reproduced
afterwards, so differences under about 20 per cent are not evidence of anything.
And key-fact recall on 20 questions moves by one or two facts between identical
runs (0.500 to 0.595 observed), which is why a change was reverted only when it
cost recall in *every* run that used it.

## Why the wrong answers were wrong

Every answerable question that missed a key fact or carried no citation was
classified by cause. The answers and the passages they were written from stay on
the machine that produced them, so only the counts are published here.

Both models were scored on the same 130 questions, so the columns are comparable.

| Cause | `gemma3:4b` | `qwen3:4b-instruct` |
|---|---:|---:|
| **(a)** Retrieval did not return the right service in the top 4 | 2 | 3 |
| **(a)** The service was returned, but no retrieved passage states the fact | 6 | 6 |
| **(b)** A passage stated the fact and the answer left it out or changed it | 39 | 49 |
| **(b)** The answer carried no citation at all | 5 | 3 |
| **(c)** The grading rule was wrong — **fixed**, with a test for each | 5 classes | 5 classes |
| **(c)** Flagged, read by hand, and not a grading fault after all | 1 | 1 |
| **(d)** The source does not state the fact | 0 | 0 |

Retrieval accounts for very little: 8 of 53 failures for `gemma3:4b` and 9 of 62
for `qwen3:4b-instruct`. Nothing was caused by a gap in the source, because the
expected facts are derived from the source in the first place.

### Five grading rules were wrong, and are fixed

Each was found by reading the answers behind the counts rather than trusting the
counts, and each is pinned by a test. None of them loosens the grading in any
other respect — two of them make it stricter.

1. **A correct refusal was scored as a wrong answer.** The English phrase list
   recognised "does not cover" and "no information" but nothing containing
   "contain", so *"The passages do not contain information on how to renew a
   passport"* counted as a failure to decline.
2. **The Arabic phrase list was matched against un-normalised text**, so a
   refusal matched or not depending on which of several meaning-identical
   spellings the model produced. It is now matched on normalised text, written
   in normalised spelling, and extended to an equivalent of every English
   phrase.
3. **A document was graded on the condition it was grouped under.** A nested
   requirement is flattened to "condition — document". Matching the whole phrase
   scored one model 3/3 and another 0/3 *for the same three documents*, and —
   the same fault inverted — credited an answer that named the condition and no
   document at all. Grading the document alone fixes both.
4. **A surcharge named after what it funds was invisible.** The pattern needed
   the currency word next to the number, but these are written as figure, then
   name, then currency, so an answer stating one correctly was scored as having
   missed the figure it plainly states.
5. **The Arabic conjunction hid a document.** It attaches to the word it joins,
   so a document listed after "and" became a different token. It is now allowed
   on the answer's side rather than stripped from every token, because the same
   letter begins ordinary words such as "وثيقة".

### The largest remaining fault is in the measurement, not the models

`key_fact_recall` requires **every figure a field states**, including the ones
the question did not ask about.

`en-001` asks what a police clearance certificate costs *for a resident*.
`gemma3:4b` gave exactly the resident figure the source states, correctly cited,
and nothing else — and scored **1 of 5**. The fees section of that service also
states a different figure for citizens, two more for applicants outside the
country, and a fixed government surcharge, and every one of them is derived as a
required fact for the question.

Counted over the whole set, of the cause-(b) fact failures above:

| | `gemma3:4b` | `qwen3:4b-instruct` |
|---|---:|---:|
| A figure the question did not ask for | 30, across 14 questions | 29, across 16 questions |
| A genuine omission | **9** | **20** |

So **`key_fact_recall` is a lower bound**: it penalises a precisely scoped
answer. The genuine-omission counts, 9 against 20, separate the two models far
more sharply than the published 0.726 against 0.663 does.

This is not corrected here. The correction is not a grading rule but a change to
the question set — each question would have to record which figure it expects —
and inventing a looser rule to raise the score would be the wrong way round. It
penalises both models in the same direction, so the comparison and the choice of
default stand. It is the first thing worth fixing.

### What was tried against cause (b), and reverted

Cause (b) led the first count, so one prompt change was measured against it on
the fixed 20-question subset, against a baseline re-scored with the same rules:
a rule requiring every fee, duration and document the passages state for what
was asked. It changed key-fact recall by **nothing** (0.595 either way) and cost
citation presence (1.000 to 0.900), so it was reverted. That it moved recall by
exactly zero is what led to the measurement fault above.

A second change, telling the model to decline when the passages describe a
different service, was measured against the two out-of-scope questions that
reach the model. It did not fix either of them and cost key-fact recall
(0.571 to 0.548) and citation presence (1.000 to 0.950), so it was reverted too.
Those two questions are a known limitation rather than a solved problem.

## Composite score

One number per model so the choice is reproducible. Weights: key_fact_recall 30%, citation_validity 15%, language_match 15%, behaviour_accuracy 15%, speed 25%.

Speed is a quarter of the score rather than a tie-breaker: a model that is
accurate and takes half a minute to begin answering is not the better choice
for something a reader is waiting on.

The speed component is a linear scale on the **median time to first token**:

```
speed = clamp((SLOW - first_token_p50) / (SLOW - FAST), 0, 1)
  FAST = 5 s or less -> 1.0
  SLOW = 60 s or more -> 0.0
```

So 5 seconds or less scores 1.0, 60 seconds or more scores 0.0, and the scale is straight between them.
The bounds describe what processor-only inference can do, not what any candidate
scores: five seconds is about the floor for a four-billion-parameter model that
must read several hundred tokens of prompt before it can begin, and a minute is
long past the point a reader gives up. They were fixed before the models were
compared. An earlier version zeroed at twenty seconds on the 95th percentile,
which put every realistic candidate at exactly 0.000 and so quietly removed a
quarter of the score; the median is also the fairer statistic here, because
run-to-run variance is large and the tail measures the longest prompt rather
than how the model feels to use.

| Model | Composite | Speed component | First token p50 used |
|---|---:|---:|---:|
| `gemma3:4b` | 0.890 | 0.895 | 10.8 s *(clean)* |
| `qwen3:4b-instruct-2507-q4_K_M` | 0.871 | 0.902 | 10.4 s *(clean)* |
| `qwen3:4b` *(subset, not comparable)* | 0.545 | 0.367 | 39.8 s *(clean)* |

The speed component is taken from the `latency` subset for `gemma3:4b`, `qwen3:4b-instruct-2507-q4_K_M`, `qwen3:4b`, measured with nothing else running on the machine, because latency is the only score another process can move. Accuracy comes from the whole question set. Figures marked *(clean)* above come from that run.

## Default

**`gemma3:4b`** is the configured default, on the numbers above.

Change it by setting `GENERATION_MODEL` in `.env`. Every model compared here is
a roughly four-billion-parameter instruction model at four-bit quantisation; a
larger model is a supported configuration rather than the default, because the
default has to work on an ordinary machine.
