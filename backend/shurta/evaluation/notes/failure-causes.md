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
6. **A question was graded against figures it did not ask about** — the largest
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
| Key facts recalled, scoped | **0.826** | **0.755** |
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
