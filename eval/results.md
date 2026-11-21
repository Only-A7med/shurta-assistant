# Answer evaluation

Measured on 2026-11-21 over 372 indexed passages, on processor-only inference.

Every figure is measured by rule, not judged by a model. Amounts, durations and
document phrases are compared against what the cited service states, on
Arabic-normalised text so that a fee in Arabic-Indic digits matches one in
Western digits.

## Accuracy

Both instruction models were scored on the same 130 questions, so their figures are directly comparable. The subset row is not.

Key facts recalled is scored on the variant of a field each question asks about,
so a question asking a resident's fee is graded on the resident figure. The next
column is the same answers graded against every figure the field states, which is
what earlier runs reported. Both are shown so the change is visible. The composite
score uses the scoped figure.

| Model | Questions | Key facts recalled | Recalled, all figures | All facts present | Citations valid | Citations present | Answer language | Behaviour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemma3:4b` | 130 | 0.826 | 0.719 | 0.689 | 1.000 | 0.955 | 1.000 | 0.992 |
| `qwen3:4b-instruct-2507-q4_K_M` | 130 | 0.755 | 0.657 | 0.541 | 1.000 | 0.973 | 0.985 | 0.992 |
| `qwen3:4b` | 20 (subset) | — | — | — | — | — | — | — |

## Speed

| Model | Prompt tokens p50 | Prompt processing p50 | First token p50 | First token p95 | Full answer p50 | Full answer p95 | Tokens per second |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemma3:4b` | 604 | 9.2 s | 9.2 s | 18.7 s | 13.0 s | 28.3 s | 10.2 |
| `qwen3:4b-instruct-2507-q4_K_M` | 656 | 9.7 s | 9.7 s | 20.2 s | 13.1 s | 30.6 s | 7.9 |
| `qwen3:4b` | 662 | 10.1 s | 39.8 s | 67.5 s | 66.9 s | 92.4 s | 8.7 |

- `qwen3:4b`: measured on the 20-question latency subset only, so not comparable with the full runs above. It is a reasoning-only build whose reasoning cannot be switched off, so the whole token budget can go on reasoning the reader never sees. Only 10 of its 20 answers were non-empty, so accuracy is not reported.

## Time to first token

At the shipped defaults `gemma3:4b` reaches its first token in a median of 9.2
seconds over all 130 questions and 10.8 seconds on the 20-question latency
subset re-run with nothing else on the machine, both measured on 2026-11-21;
`qwen3:4b-instruct` gives 9.7 and 10.4 seconds. The speed table above carries
the percentiles for each.

All of the perceptible wait is the model reading the prompt. The provider reports
on its final streamed event how many prompt tokens it processed and how long that
took, and prompt processing is 99 per cent of the time to first token: a median
604 prompt tokens take 9.15 of the 9.21 seconds, against a model load of 17
milliseconds and retrieval of 50 to 70 milliseconds.

### What was tried

Measured on 2026-11-20 with `gemma3:4b` on the same fixed 20-question subset —
mixed English, Modern Standard Arabic, Gulf dialect and code-switched, every item
answerable and carrying checkable facts — on processor-only inference, with other
work running on the machine between runs. Kept as the record of what was tried;
the figures are from that day's conditions and are not the shipped-default
figures above.

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
| Service named once per group | 617 | 13.6 s | 0.571 | 1.000 | 1.000 | yes |
| Service named once + 12 threads | 617 | 11.4 s | 0.571 | 1.000 | 1.000 | optional |

### What was kept, and what was reverted

Kept: the service is named once per group of passages, not once per passage.
A chunk's text repeats both of its service's names and its category, which is
what lets an Arabic question retrieve an English passage. In the prompt that is
redundant, so passages are grouped by service and the service is named once, in
the reader's language. The median prompt falls by 89 tokens with no information
lost and no measured change in accuracy.

Kept: an unchanged prompt prefix. The rules block was already first and
byte-identical between requests in one language, which lets the provider reuse
its cached prefix. Measured directly against the provider with a 170-token
prefix: 9.2 seconds with the prefix first against 13.1 seconds with the same
block moved to the end, and 0.5 seconds when a prompt is repeated exactly. A test
asserts the prefix is stable, because moving anything variable above the rules
would silently cost seconds per question.

Kept: the models stay resident. `keep_alive` is sent with generation *and*
embedding requests, and start-up warms both. `ollama ps` confirms neither is
reloaded between questions, and the reported load time per question is about 17
milliseconds.

Reverted: capping passage length. A 500-character cap saved 68 tokens and
cost key-fact recall in both runs that used it (0.524 and 0.500 against 0.571
unchanged). The tail of a fee or document list is sometimes the part the question
was about. The cap is still available as `MAX_PASSAGE_CHARS`, unset by default.

Reverted: `top_k` 3. Two runs with three passages both scored 0.500 on key
facts against a mean of 0.571 with four. A question is often answered by two
sections of one service, and the fourth passage is what carries the second one.

Optional, not a default: the generation thread count. 12 threads was the
fastest setting measured and 4 the slowest, a difference of 6.6 seconds with no
change in accuracy. It is left unset so the provider chooses, because the right
number is the machine's physical core count and a value hard-coded here would be
wrong on a smaller machine. Set `NUM_THREAD` to tune it.

### Was the target reached?

No. The target was under 4 seconds to the first token, and the shipped defaults
give the 9.2 and 10.8 second medians above.

The arithmetic says why. Prompt processing runs at roughly 45 to 50 tokens per
second in these measurements, so a 4-second budget buys about 200 tokens. The
rules block alone is about 230 tokens — free after the first question in a
language, because it is cached — and the four passages and the question are
another 390. Reaching 200 uncached tokens means one passage, or a truncated one,
and both were measured to cost key-fact recall. Under 4 seconds is reachable
only by answering worse.

Two caveats. Run-to-run variance is large: identical settings produced 15.0 and
18.3 seconds, and two early runs reached 8.0 seconds at a prompt-processing rate
never reproduced afterwards, so differences under about 20 per cent are not
evidence of anything. And key-fact recall on 20 questions moves by one or two
facts between identical runs (0.500 to 0.595 observed), which is why a change was
reverted only when it cost recall in *every* run that used it.

## Why the wrong answers were wrong

Every answerable question that missed a key fact or carried no citation was
classified by cause, by reading the answer and the passages it was written from.
Those stay on the machine that produced them, so only the counts are published.

Both models were scored on the same 130 questions, so the columns are
comparable. The table is the classification made on 2026-11-21, before the
questions recorded which variant of a field they ask about; the current totals
are further down, recomputed from the scorecards.

| Cause | `gemma3:4b` | `qwen3:4b-instruct` |
|---|---:|---:|
| (a) Retrieval did not return the right service in the top 4 | 2 | 3 |
| (a) The service was returned, but no retrieved passage states the fact | 6 | 6 |
| (b) A passage stated the fact and the answer left it out or changed it | 39 | 49 |
| (b) The answer carried no citation at all | 5 | 3 |
| (c) The grading rule was wrong — fixed, with a test for each | 5 classes | 5 classes |
| (c) Flagged, read by hand, and not a grading fault after all | 1 | 1 |
| (d) The source does not state the fact | 0 | 0 |

Retrieval accounts for very little: 8 of 53 failures for `gemma3:4b` and 9 of 62
for `qwen3:4b-instruct`. Nothing was caused by a gap in the source, because the
expected facts are derived from the source in the first place.

### Six grading rules were wrong, and are fixed

Each was found by reading the answers behind the counts rather than trusting the
counts, and each is pinned by a test. None loosens the grading in any other
respect, and two make it stricter.

1. A correct refusal was scored as a wrong answer. The English phrase list
   recognised "does not cover" and "no information" but nothing containing
   "contain", so *"The passages do not contain information on how to renew a
   passport"* counted as a failure to decline.
2. The Arabic phrase list was matched against un-normalised text, so a
   refusal matched or not depending on which of several meaning-identical
   spellings the model produced. It is now matched on normalised text, written
   in normalised spelling, and extended to an equivalent of every English
   phrase.
3. A document was graded on the condition it was grouped under. A nested
   requirement is flattened to "condition — document". Matching the whole phrase
   scored one model 3/3 and another 0/3 *for the same three documents*, and —
   the same fault inverted — credited an answer that named the condition and no
   document at all. Grading the document alone fixes both.
4. A surcharge named after what it funds was invisible. The pattern needed
   the currency word next to the number, but these are written as figure, then
   name, then currency, so an answer stating one correctly was scored as having
   missed the figure it plainly states.
5. The Arabic conjunction hid a document. It attaches to the word it joins,
   so a document listed after "and" became a different token. It is now allowed
   on the answer's side rather than stripped from every token, because the same
   letter begins ordinary words such as "وثيقة".
6. A question was graded against figures it did not ask about — the largest
   of the six, and a fault in the question set rather than in the grader. It is
   described next.

### Questions now say which figure they ask for

A fee field states a figure per applicant category, one per line: citizen,
resident, from outside the country, and a fixed government surcharge. Key-fact
recall required every one of them, whatever the question asked.

`en-001` asks what a police clearance certificate costs *for a resident*.
`gemma3:4b` gave the resident figure the source states, correctly cited, and
nothing else — and scored 1 of 5.

The fix is in the question set, not the grading rule. A question may now record
which variant of a field it asks about, as a category label — `resident`,
`outside_country`, `in_person` — and `eval prepare` derives only that variant's
facts. A question that asks about the whole field ("what are the fees") records
nothing and keeps every figure, and so does a question whose wording does not
say which figure it wants. Twelve of the 130 questions are scoped; seven of them
derive fewer facts as a result. The labels never name a figure or quote the
source, so the committed question set still carries no collected content.

Both figures are published. Scored on the variant each question asks about, over
the same stored answers:

| | `gemma3:4b` | `qwen3:4b-instruct` |
|---|---:|---:|
| Key facts recalled, scoped | 0.826 | 0.755 |
| Key facts recalled, every figure | 0.719 | 0.657 |
| Facts missed, scoped | 27 of 155, in 19 questions | 38 of 155, in 28 questions |
| Facts missed, every figure | 50 of 178 | 61 of 178 |

Scoping removes 23 fact failures from each model, so it does not move them
relative to each other: the gap is 0.071 scoped and 0.062 unscoped, and the
default model is unchanged. The earlier pass flagged 30 unasked-for figures for
`gemma3:4b` and 29 for `qwen3:4b-instruct`; scoping accounts for 23 of each, and
the rest fall in questions left unscoped on purpose, where the wording does not
say which figure it wants.

Two notes on reading these numbers. The all-figures column is re-derived from the
current extractor and comes out at 0.719 and 0.657, where the figures published
before this change were 0.726 and 0.663: `eval prepare` had not been re-run
after grading rule 4 above, which adds one expected figure to the set. And the
cause table higher up was classified by hand against the facts as they were then
derived, so it is not a row-by-row reconciliation of the totals here.

### What was tried against the model, and reverted

Cause (b) led the first count, so one prompt change was measured against it on
the fixed 20-question subset, against a baseline re-scored with the same rules:
a rule requiring every fee, duration and document the passages state for what
was asked. It changed key-fact recall by nothing (0.595 either way) and cost
citation presence (1.000 to 0.900), so it was reverted. That it moved recall by
exactly zero is what led to the measurement fault above.

A second change, telling the model to decline when the passages describe a
different service, was measured against the two out-of-scope questions that
reach the model. It did not fix either of them and cost key-fact recall (0.571 to
0.548) and citation presence (1.000 to 0.950), so it was reverted too. Those two
questions are a known limitation rather than a solved problem.

## Composite score

One number per model so the choice is reproducible. Weights: key facts 30%, speed 25%, citation validity 15%, answer language 15%, behaviour 15%.

Speed is a quarter of the score rather than a tie-breaker: a model that is
accurate and takes half a minute to begin answering is not the better choice
for something a reader is waiting on. It is a linear scale on the median time
to first token.

```
speed = clamp((SLOW - first_token_p50) / (SLOW - FAST), 0, 1)
  FAST = 5 s or less -> 1.0
  SLOW = 60 s or more -> 0.0
```

The bounds describe what processor-only inference can do, not what any candidate
scores, and were fixed before the models were compared: five seconds is about the
floor for a four-billion-parameter model that must read several hundred tokens of
prompt before it can begin, and a minute is long past the point a reader gives up.
An earlier version zeroed at twenty seconds on the 95th percentile, which put
every realistic candidate at exactly 0.000 and so removed a quarter of the score.
The median rather than the tail, because run-to-run variance is large and the tail
measures the longest prompt rather than how the model feels to use.

| Model | Composite | Speed component | First token p50 used |
|---|---:|---:|---:|
| `gemma3:4b` | 0.920 | 0.895 | 10.8 s *(clean)* |
| `qwen3:4b-instruct-2507-q4_K_M` | 0.898 | 0.902 | 10.4 s *(clean)* |
| `qwen3:4b` *(subset, not comparable)* | — | 0.367 | 39.8 s *(clean)* |

Each model marked *(clean)* above takes its speed component from the `latency` subset, run with nothing else on the machine, because latency is the only score another process can move, while its accuracy comes from the whole question set.

## Default

**`gemma3:4b`** is the configured default, on the numbers above.

Change it by setting `GENERATION_MODEL` in `.env`. Every model compared here is
a roughly four-billion-parameter instruction model at four-bit quantisation; a
larger model is a supported configuration rather than the default, because the
default has to work on an ordinary machine.
