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
