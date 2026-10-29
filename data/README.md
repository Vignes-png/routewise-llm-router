# Routing benchmark dataset (v1)

`routing_benchmark.jsonl` holds 60 labelled prompts used by `scripts/benchmark_routing.py`
to compare the rule-based router with the semantic-plus-rules router.

**Status of the labels.** The labels are hand-designed expectations, written from the rubric
below *before* either router was run on the prompts. They are judgements, not ground truth,
and they have not yet been reviewed by a second person. Treat disagreements between a label
and a router as a prompt to re-read the label as much as the router.

## Record format (one JSON object per line)

| field | meaning |
|---|---|
| `id` | unique, stable identifier such as `cod-03`. Never reuse or renumber an id. |
| `prompt` | the text sent to both routers |
| `expected_category` | `coding`, `reasoning`, `summarization`, `creative_writing` or `general` |
| `expected_tier` | `cheap` or `strong` |
| `label_note` | one-line reason for the labels |
| `tags` | optional list describing the kind of prompt (see below) |

The set has 12 prompts per category: 60 in total (40 expected `cheap`, 20 expected `strong`).

## Rubric: expected_category

Label by the **main task being requested**, not by the topic words in the prompt.
A poem *about* a deadlock is `creative_writing`; a question that merely starts with "Why" is not
automatically `reasoning`.

* `coding` - writing, explaining, debugging or designing software, queries or data systems.
* `reasoning` - multi-step logic, maths, probability, argument or decision analysis.
* `summarization` - condensing, simplifying or extracting from supplied or well-known material.
* `creative_writing` - original imaginative text: stories, poems, songs, speeches, descriptions.
* `general` - everyday facts, advice, chit-chat, translation and everything else.

## Rubric: expected_tier

`strong` if a good answer needs **at least one** of:

1. **S1 multi-step reasoning** where intermediate steps matter (proofs, probability, estimation chains).
2. **S2 diagnosis or design** of a non-trivial technical system with interacting causes or constraints.
3. **S3 open-ended analysis or decision** with competing considerations and no single correct answer.
4. **S4 high-stakes fidelity**: a wrong or sloppy answer is costly (security, health, legal text).
5. **S5 demanding creative output**: strict forms, sustained voice, planted structure, or long-form writing.

Otherwise `cheap`: lookups, definitions, standard code patterns, short rewrites or compressions,
brief creative pieces, and everyday advice with many acceptable answers.

Two rules keep the labels independent of keyword routing: prompt **length alone** never makes a
prompt `strong`, and a hard-sounding **word alone** never makes a prompt `strong`.
The label is about what a good answer requires.

## Tags

| tag | meaning |
|---|---|
| `simple` | obvious category and easy answer |
| `moderate` | standard task with a little care needed |
| `difficult` | clearly demanding |
| `ambiguous` | a reasonable reviewer could pick another category |
| `short-but-difficult` | very short prompt, hard answer |
| `misleading-keyword` | contains a word a keyword router could misread |
| `keyword-free` | contains none of the obvious keywords for its true category |
| `long-but-simple` | long input, easy task (checks the length heuristic) |

## Avoiding leakage

No prompt may be identical to one of the 40 example prompts embedded in `app/semantic.py`
(`CATEGORY_EXAMPLES`). The benchmark script rejects exact copies and warns about prompts
with very high word overlap. Some prompts are on similar topics to those examples; that is
intentional, because the benchmark should test generalisation to new wording.

## Changing the dataset

The benchmark is only comparable between runs if the prompt set is stable.

* Do not edit prompts or labels silently. If a label is wrong, fix it, note why in your commit message,
  and treat earlier results as produced from the previous version.
* Do not adjust labels to make a router look better. Never tune the router's rules or examples on
  these same prompts and then quote the score as performance; collect a separate set for that.
* Each report prints a fingerprint (SHA-256, line-ending independent) of the file it used.
  Quote it with any result.

## Known limits

* 60 prompts is small; see the 95% confidence intervals printed in the report.
* The prompts are synthetic and written by one author, so they do not represent real user traffic.
* About a sixth of the labels carry the `ambiguous` tag. Review those first.
