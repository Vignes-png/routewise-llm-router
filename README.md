

\# RouteWise — Intelligent LLM Router



\## What is RouteWise?



RouteWise is an LLMOps project that intelligently routes user prompts to suitable large language models (LLMs).



Different prompts have different requirements. A simple question may not need an expensive model, whereas a difficult coding or reasoning task may benefit from a stronger model.



RouteWise aims to balance response quality, estimated cost, and latency.



\## How does it work?



1\. The user submits a prompt.

2\. The semantic router identifies the type of task.

3\. The rule-based policy uses the routing result and system constraints to select a model.

4\. The dispatcher sends the prompt to the selected model.

5\. The model generates a response, which is returned to the user.

6\. The system records relevant routing decisions and performance metrics.



\## Planned Features



\- Rule-based routing baseline

\- Semantic routing using local embeddings

\- Actual LLM response generation

\- Multiple model/provider integrations

\- Fallback handling for failed requests

\- SQLite-based request logging

\- Latency and estimated cost tracking

\- Routing strategy benchmarks

\- Streamlit demo interface



\## Technology Stack



\- Python 3.11

\- FastAPI

\- Pydantic

\- LiteLLM

\- Sentence Transformers

\- SQLite

\- Streamlit

\- Git



\## Current Status



Phase 0: Development environment and Git repository setup.



The application and routing logic have not been implemented yet.



\## Security



API keys and secrets must be stored in a local `.env` file.



Never commit API keys, passwords, or other secrets to Git.



## Observability

RouteWise records metadata for requests reaching `POST /chat` in a local
SQLite database named `routewise.db`.

Recorded metadata includes routing decisions, the selected model, latency,
provider-reported token usage, success or failure status, and estimated cost
when pricing and usage data are available.

Prompts, generated answers, API keys, and raw provider error messages are not
stored in the observability database.

To inspect the latest requests, activate the `routewise` environment and run:

```powershell
python -m app.observability
```

The cost figures are estimates based on the configured model-pricing table,
not actual billing records. Unknown costs are stored as unknown.

The local database and its journal files must not be committed to Git.

## Routing benchmark

A small, fixed, labelled dataset for comparing the **rule-based router** (`POST /route`) with the
**semantic-plus-rules router** (`POST /route/semantic`). It makes **no LLM calls**, needs no API key and
costs nothing. Only the local embedding model is used.

> A 60-prompt synthetic benchmark is useful for finding bugs and obvious weaknesses. It is **not** proof
> of production performance. The labels are hand-designed expectations that still need review.

### Run it

From the project root, with the Conda environment active:

```powershell
conda activate routewise
python -m pytest -q
python -m scripts.benchmark_routing
```

The first semantic run downloads the embedding model (about 90 MB, once).

Options (all optional):

```powershell
python -m scripts.benchmark_routing --dataset data\routing_benchmark.jsonl --output reports\routing_benchmark_results.csv --low-similarity-threshold 0.40 --max-listed 40
```

It prints a summary and writes one row per prompt to `reports\routing_benchmark_results.csv`
(ground-truth labels, both predictions, similarity score, correctness flags and each router's reason).
Exit codes: `0` ok, `2` invalid dataset, `3` router or embedding-model failure, `4` CSV could not be written.

The dataset, its labelling rubric and its change policy are described in `data\README.md`.

### What the numbers mean

| Metric | Plain-language meaning |
|---|---|
| **Category accuracy** | Share of prompts where the router's category equals `expected_category`. |
| **Precision** (per category) | When the router says "coding", how often is it really coding? Low precision = the router over-uses that category. |
| **Recall** (per category) | Of the prompts that really are coding, how many did the router find? Low recall = it misses that category. |
| **F1** (per category) | One number combining precision and recall (their harmonic mean). It is low if either is low. |
| **Macro-F1** | The plain average of the per-category F1 scores, so every category counts equally. |
| **Confusion matrix** | Rows = expected category, columns = predicted category. The diagonal is correct; off-diagonal cells show *which* mistakes happen. |
| **Tier accuracy** | Share of prompts where `cheap`/`strong` equals `expected_tier`. Scored separately for each router. |
| **Over-routed** | Expected `cheap` but sent to `strong`: wastes money. |
| **Under-routed** | Expected `strong` but sent to `cheap`: risks answer quality. |
| **Always-cheap baseline** | Tier accuracy of a "router" that always says `cheap`. A real router should beat it. |
| **95% CI** | Wilson confidence interval. With 60 prompts it is wide, so small differences are noise. |
| **Category disagreements** | Prompts where the two routers pick different categories, and who matched the label. |
| **Semantic similarity** | The semantic router's score for its best category (cosine-based, **not** a probability). Low scores are listed; a useful score should be lower when the router is wrong. |

Both routers choose the model tier with the **same** complexity rules, so their tier results should be
identical. The report checks this. Only the **category** differs between the two approaches.

### Limits of this benchmark

* Measures routing decisions against labels. It does **not** measure answer quality, latency or cost.
* 60 synthetic prompts from one author are not real traffic. Confidence intervals are wide.
* Labels are judgements and may be wrong; review the "inspect manually" lists and the `ambiguous` prompts.
* Do not change the router's rules or examples to fix this benchmark and then quote the score. That is
  overfitting; use a separate set for any tuning.
