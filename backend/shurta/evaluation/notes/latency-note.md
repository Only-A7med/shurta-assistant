## Time to first token

At the shipped defaults `gemma3:4b` reaches its first token in a median of
**9.2 seconds** over all 130 questions and **10.8 seconds** on the 20-question
latency subset re-run with nothing else on the machine, both measured on
2026-11-21; `qwen3:4b-instruct` gives 9.7 and 10.4 seconds. The speed table
above carries the percentiles for each.

All of the perceptible wait is the model reading the prompt. The provider reports
on its final streamed event how many prompt tokens it processed and how long that
took, and prompt processing is 99 per cent of the time to first token: 706 prompt
tokens taking about 15 seconds, against a model load of 17 milliseconds and
retrieval of 50 to 70 milliseconds.

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

**Kept: the service is named once per group of passages, not once per passage.**
A chunk's text repeats both of its service's names and its category, which is
what lets an Arabic question retrieve an English passage. In the prompt that is
redundant, so passages are grouped by service and the service is named once, in
the reader's language. The median prompt falls by 89 tokens with no information
lost and no measured change in accuracy.

**Kept: an unchanged prompt prefix.** The rules block was already first and
byte-identical between requests in one language, which lets the provider reuse
its cached prefix. Measured directly against the provider with a 170-token
prefix: 9.2 seconds with the prefix first against 13.1 seconds with the same
block moved to the end, and 0.5 seconds when a prompt is repeated exactly. A test
asserts the prefix is stable, because moving anything variable above the rules
would silently cost seconds per question.

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

No. The target was under 4 seconds to the first token, and the shipped defaults
give the 9.2 and 10.8 second medians above.

The arithmetic says why. Prompt processing runs at roughly 45 to 50 tokens per
second here, so a 4-second budget buys about 200 tokens. The rules block alone is
about 230 tokens — free after the first question in a language, because it is
cached — and the four passages and the question are another 390. Reaching 200
uncached tokens means one passage, or a truncated one, and both were measured to
cost key-fact recall. Under 4 seconds is reachable on this inference setup only
by answering worse.

Two caveats. Run-to-run variance is large: identical settings produced 15.0 and
18.3 seconds, and two early runs reached 8.0 seconds at a prompt-processing rate
never reproduced afterwards, so differences under about 20 per cent are not
evidence of anything. And key-fact recall on 20 questions moves by one or two
facts between identical runs (0.500 to 0.595 observed), which is why a change was
reverted only when it cost recall in *every* run that used it.
