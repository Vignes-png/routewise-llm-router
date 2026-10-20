# RouteWise — Cost- and Quality-Aware LLM Router

**Project context, technical decisions, build checklist, and AI handoff notes**  
**Status:** Planning / not yet implemented (update this field as work begins)  
**Last updated:** 19 October 2026  
**Working title:** RouteWise (rename if a better name is chosen)

> This is the source-of-truth document for this project. Any AI assistant helping with it should read this first, preserve the agreed scope, and update this document when meaningful decisions change. Do not assume code has already been written just because a phase is described here.

---

# 1. Project explanation — read this first

## What are we building?

We are building an **LLM router**: a small service that sits between an application and multiple large language models (LLMs). Instead of sending every user prompt to the same model, the service chooses an appropriate model based on the task, estimated difficulty, cost, quality needs, latency, and model availability.

A typical use case:

1. A user enters: “Summarize this short paragraph in three bullet points.”
2. The router analyzes the request.
3. It decides that a cheaper/faster model is likely enough.
4. The dispatcher sends the prompt to that model.
5. The model generates the summary.
6. The API returns the answer plus metadata explaining the routing decision.
7. The application records latency, token usage, estimated cost, success/failure, and whether a fallback was needed.

For a more demanding request such as “Debug this concurrency bug and explain the race condition,” the router may choose the stronger model if evaluation shows that it handles such tasks better.

**The goal is not to build a new LLM.** The goal is to build the engineering layer that chooses and operates existing models more intelligently.

## Why is this useful?

Different models trade off response quality, response time, context capacity, availability, and price. Using the strongest model for every request may waste money; using the cheapest model for every request may degrade quality. A router tries to find a useful balance.

The engineering question is:

> Can we route appropriate requests to lower-cost models while preserving an acceptable level of response quality, and can we demonstrate that trade-off with measurements?

That question is what makes this an LLMOps / ML engineering project rather than merely a chat interface.

## What the user should see in the finished demo

The finished app should allow someone to:

- Type a prompt into a simple chat/demo interface.
- Choose a routing strategy: **Rules**, **Semantic**, or **Auto/Selected strategy**.
- Optionally preview the route without generating an answer (route-only / dry-run mode).
- Submit the prompt and receive an actual response from the selected model.
- See why the router picked the model tier and which configured model was called.
- See request latency, token counts when supplied by the provider, estimated cost when it can be calculated, and whether a fallback was used.
- View aggregate metrics and a benchmark comparing routing strategies.

The project should also expose an API so the UI is only one client; the router itself must not live inside Streamlit.

---

# 2. Goals and non-goals

## Main goals

1. Build a functioning FastAPI service that can accept a prompt, route it, call an LLM, and return the response.
2. Keep routing logic separate from model-provider code.
3. Support at least two distinct model configurations for meaningful routing. They may be hosted API models, local models, or one of each. Actual choices depend on which free options are available and what the computer can run.
4. Implement a rule-based baseline and a local semantic-routing strategy.
5. Record useful observability data in SQLite.
6. Compare routing policies on a fixed prompt set, including quality, estimated cost, and latency.
7. Build a simple Streamlit UI and a readable GitHub README.
8. Keep the first version free-first. No paid subscription or paid API usage is required by design, but free provider quotas and eligibility can change.

## Non-goals for version 1

Do **not** begin with these:

- Training a large language model.
- Fine-tuning BERT or training a complex learned classifier.
- Implementing a production-grade multi-tenant gateway.
- Building authentication, user accounts, billing, or team budgets.
- Deploying to Kubernetes or building a distributed microservice system.
- Implementing advanced Prometheus/Grafana infrastructure before basic SQLite metrics work.
- Supporting every provider or every model.
- Claiming verified dollar savings without token usage and a clearly documented pricing calculation.

These may be future extensions, not MVP requirements.

---

# 3. Scope and definitions

## Two operating modes

### A. Route-only mode

Input: a prompt.  
Output: a routing decision, such as selected tier, predicted task category, explanation, strategy, and confidence/similarity where applicable.  
No model inference is called.

This mode is useful for debugging and evaluating routing behavior.

### B. Route-and-generate mode (main feature)

Input: a prompt.  
Flow: route the prompt → dispatch it to the configured model → return the actual answer → save metadata.  
Output: generated answer plus routing and execution metadata.

**The final portfolio demo must include route-and-generate mode.** Route-only mode is an additional debugging/evaluation feature, not the entire project.

## Key terminology

- **Gateway/API:** The entry point accepting HTTP requests.
- **Router engine:** Decides which model tier or route is appropriate.
- **Routing strategy:** The algorithm used to make that decision (rules or semantic matching in v1).
- **Model tier:** A logical label, initially `cheap` or `strong`. It is not itself a provider or a model name.
- **Model registry/configuration:** Maps a tier to a configured model identifier and provider settings.
- **Dispatcher:** Executes a model call through LiteLLM and normalizes the result.
- **Fallback:** A second configured model is used when the primary model fails for a recoverable reason.
- **Observability:** Recording what happened: latency, usage, status, estimated cost, routing reason, fallbacks, etc.
- **Evaluation:** Comparing strategies on the same test prompts to see whether routing choices produce an acceptable quality/cost trade-off.
- **Estimated cost:** A calculation based on observed token counts and a configured price table. A free-tier API request may have zero current billable cost, but the project can still estimate what it would cost at published paid rates. Label this as estimated, not actual savings.

---

# 4. Agreed high-level architecture

```text
                    ┌──────────────────────────┐
                    │   Streamlit demo UI      │
                    │ prompt + metrics + route │
                    └────────────┬─────────────┘
                                 │ HTTP
                                 ▼
                    ┌──────────────────────────┐
                    │       FastAPI API        │
                    │ validation + endpoints   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │      Router Service      │
                    │ coordinates route/call   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       Router Engine      │
                    │ rules / semantic policy  │
                    └────────────┬─────────────┘
                                 │ tier + reason
                                 ▼
                    ┌──────────────────────────┐
                    │ Model Registry/Dispatcher│
                    │ LiteLLM + timeout/retry  │
                    └───────┬───────────┬──────┘
                            │           │
                            ▼           ▼
                    ┌────────────┐ ┌────────────┐
                    │ Hosted API │ │ Local LLM  │
                    │ if free tier│ │ if feasible│
                    └────────────┘ └────────────┘

     Router/API ───────────────► SQLite event/request log

     Fixed evaluation dataset ─► benchmark runner ─► CSV/JSON results
```

## Layer responsibilities

### Layer 1 — Demo client

Streamlit provides the human interface. It calls the FastAPI API; it does not implement routing or call provider SDKs directly.

### Layer 2 — FastAPI gateway

Validates incoming data, calls the routing/service layer, returns a structured response, and exposes health and metrics endpoints. It should never contain the whole routing algorithm in endpoint functions.

### Layer 3 — Router engine

Receives a prompt and a requested strategy. Returns a `RouteDecision` object. It must be testable without API keys or live model calls.

### Layer 4 — Registry and dispatcher

Resolves `cheap` and `strong` tiers to configured model IDs. Uses LiteLLM to call provider models through a largely unified interface. Handles timeouts, safe retries/fallbacks, and normalized usage fields.

### Supporting component — Observability

SQLite records one event for each completed or failed request. Store metadata such as timestamp, routing strategy, category, tier, actual model identifier, reason, latency, input/output/total token counts if available, estimated cost, status, and fallback use. Avoid recording API keys. Avoid persisting raw user prompts by default; any optional prompt storage must be clearly configured and documented.

### Supporting component — Evaluation

A local benchmark script runs fixed prompts against different routing policies or analyzes saved results. Evaluation should not be mixed into normal API request handling.

---

# 5. Recommended technology stack

Use the simplest reliable tools. Do not add a framework just because it is popular.

| Need | Choice | Why |
|---|---|---|
| Language | Python 3.11 or 3.12 | Familiar, widely supported version range for common project dependencies |
| API | FastAPI + Uvicorn | Clear request/response schemas and a simple development server |
| Validation/settings | Pydantic / Pydantic Settings or `python-dotenv` | Validate requests and read environment configuration |
| Provider abstraction | LiteLLM Python SDK | Keeps provider-specific call code smaller and normalizes responses |
| Rule baseline | Plain Python | Transparent, easy to test, no dependency needed |
| Semantic routing | `sentence-transformers` with `all-MiniLM-L6-v2` | Embeddings can be computed locally; compare cosine similarity to route examples/descriptions |
| Storage | SQLite | No separate database server required |
| Data and benchmark | pandas + JSONL/CSV | Easy to inspect and analyze benchmark outputs |
| Tests | pytest + FastAPI TestClient/httpx | Test the API and routing logic locally |
| Demo UI | Streamlit | Simple user-facing chat and metric panels |
| Version control | Git + GitHub | Track changes, show progress, and publish the project |
| Secrets | `.env` (ignored by Git) | Keep API keys out of source control |

**Install semantic-routing dependencies only when that phase begins.** Sentence Transformers may download model weights and additional ML dependencies; installing it in the first minute adds setup risk before the API baseline works.

## Official documentation to consult

- FastAPI first steps: https://fastapi.tiangolo.com/tutorial/first-steps/
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/
- LiteLLM Python SDK: https://docs.litellm.ai/docs/python_sdk
- LiteLLM quickstart: https://docs.litellm.ai/docs/learn/python_sdk_quickstart
- Sentence Transformers similarity: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- Gemini API pricing/free-tier details, if that provider is chosen: https://ai.google.dev/gemini-api/docs/pricing
- Groq rate limits, if that provider is chosen: https://console.groq.com/docs/rate-limits
- Ollama API introduction, if local inference is chosen: https://github.com/ollama/ollama/blob/main/docs/api/introduction.mdx

Free models, quotas, model IDs, and account eligibility can change. Check the provider's current official docs before selecting actual model identifiers. Do not hard-code a claim that a model is free forever.

---

# 6. Free-first model strategy

The software stack can be free. Actual inference is more nuanced: API providers may offer free quotas, and those quotas can change or be unavailable to a particular account. Local inference avoids per-request hosted inference charges but uses local hardware and may be slow.

## Required design principle

Do not write model IDs throughout the code. Put actual model names in configuration so they can be changed without editing routing logic.

Conceptual configuration:

```env
# Example placeholders only. Replace with valid model identifiers from current docs.
CHEAP_MODEL=provider/model-id-for-cheap-tier
STRONG_MODEL=provider/model-id-for-strong-tier
FALLBACK_MODEL=provider/model-id-for-fallback

# Provider-specific API keys go here, when needed.
# Keep real values only in .env, never commit them.
```

The actual provider prefix and variable names depend on the chosen provider/LiteLLM integration. Confirm the current LiteLLM provider documentation when implementing the call.

## Practical sequence

1. Implement all route logic and most tests without an API key.
2. Add a mock/fake model dispatcher for repeatable tests.
3. Choose one provider offering accessible free API quota, if available, and prove one real call works.
4. Configure a second distinct model route (another eligible model/provider or a local Ollama model) before claiming the system has meaningful live cost/quality routing.
5. If hosted quota fails, tests and route-only mode must still work. Use a local model if the device can handle it, or postpone live generation until a valid model endpoint is configured.

Do not send secrets or private prompts to a service unless you are comfortable with its data-use policy. Free tiers can have different data-use terms from paid tiers.

---

# 7. Repository layout

Create a project folder named `routewise-llm-router` (or similar). Build this structure gradually; do not create every empty file at the start just to match a diagram.

```text
routewise-llm-router/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app and endpoint wiring
│   ├── config.py               # environment-backed settings
│   ├── schemas.py              # request and response schemas
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── rules.py             # rule-based baseline
│   │   ├── semantic.py          # embedding-based classifier (later phase)
│   │   └── policy.py            # strategy selection and tier policy
│   ├── services/
│   │   ├── __init__.py
│   │   └── router_service.py    # orchestrate routing + dispatch + logging
│   ├── providers/
│   │   ├── __init__.py
│   │   └── dispatcher.py        # LiteLLM calls, timeout/fallback handling
│   ├── storage/
│   │   ├── __init__.py
│   │   └── sqlite_store.py      # database setup and event persistence
│   └── metrics/
│       ├── __init__.py
│       └── cost.py              # estimated cost calculations
├── evaluation/
│   ├── prompts.jsonl           # fixed, versioned evaluation prompts
│   ├── run_benchmark.py         # benchmark runner (later phase)
│   └── results/                 # generated results; document whether committed
├── tests/
│   ├── test_rules.py
│   ├── test_api.py
│   ├── test_dispatcher.py
│   └── test_storage.py
├── ui/
│   └── streamlit_app.py
├── .env.example                 # blank placeholders only
├── .gitignore
├── requirements.txt
├── README.md
└── PROJECT_CONTEXT.md           # this document (copy it into the repository)
```

The structure can be simplified or adjusted when justified. Do not split every three-line function into a new file. The separation matters most between routing, provider calls, API endpoints, storage, and evaluation.

---

# 8. Full step-by-step build plan

Estimated effort with AI-assisted development: **about 25–35 focused hours**, spread over approximately 3–4 weeks at 1–2 hours per day. This is an estimate, not a guarantee. Environment problems, provider quota issues, and unfamiliar library behavior can add time.

## Phase 0 — Repository and requirements (1–2 hours)

### Tasks

1. Create the local folder and initialize Git.
2. Create a virtual environment.
3. Add `.gitignore` before creating secrets.
4. Add `README.md` and copy this context document into the repository as `PROJECT_CONTEXT.md`.
5. Write the MVP checklist in the README.
6. Confirm the current Python version and that the virtual environment is active before installing packages.

### Windows PowerShell commands

```powershell
mkdir routewise-llm-router
cd routewise-llm-router
git init
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `py -3.12` is not available, check `python --version` and use an installed supported version. Do not install random Python versions repeatedly to fix a package error; first read the actual error.

### Linux / Pop!_OS alternative

```bash
mkdir routewise-llm-router
cd routewise-llm-router
git init
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Initial `.gitignore`

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
*.db
*.sqlite3
.pytest_cache/
.streamlit/secrets.toml
.DS_Store
```

### Acceptance criteria

- The repository exists and Git is initialized.
- The virtual environment activates successfully.
- `python --version` returns the expected version.
- `.env` and the SQLite database are excluded from Git.

Commit: `chore: initialize project and environment`

---

## Phase 1 — Minimal FastAPI application (2–3 hours)

### Tasks

1. Install core packages only.
2. Create `app/__init__.py` and `app/main.py`.
3. Create a FastAPI instance.
4. Add `GET /health`, returning a small JSON status such as `{"status":"ok"}`.
5. Start the server and open its automatically generated API docs.
6. Add a basic test for `/health`.

### Core dependency installation

```powershell
python -m pip install fastapi "uvicorn[standard]" pydantic-settings python-dotenv litellm pytest httpx streamlit pandas
```

Record versions in `requirements.txt` after verifying the environment works:

```powershell
python -m pip freeze > requirements.txt
```

A fully pinned freeze can be noisy across machines; for a student project it is acceptable initially. If dependency conflicts appear, inspect them rather than repeatedly reinstalling everything.

### Run server (from repository root)

```powershell
uvicorn app.main:app --reload
```

Open these in the browser:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

Stop with `Ctrl+C`.

### Acceptance criteria

- Server starts without an import error.
- `/health` returns HTTP 200 and the expected JSON.
- `/docs` opens.
- A pytest test passes without any LLM API key.

Commit: `feat: add FastAPI skeleton and health endpoint`

---

## Phase 2 — Define the API contract and route-only mode (3–4 hours)

### Tasks

1. Add Pydantic request and response schemas in `app/schemas.py`.
2. Add a `POST /route` endpoint that does **not** call a model.
3. Create a `RouteDecision` representation with fields similar to:
   - `strategy`: `rules` or `semantic`
   - `category`: predicted task category
   - `tier`: `cheap` or `strong`
   - `reason`: concise human-readable explanation
   - `confidence`: optional; do not claim calibrated probability if it is only a similarity score
   - `similarity_score`: optional for semantic strategy
4. Validate empty or extremely long prompts and return understandable validation errors.
5. Make sure rule-based routing is a pure function that does not require FastAPI or provider credentials.
6. Write unit tests for expected rule decisions and edge cases.

### Initial task categories

Use a small, clearly defined set:

- `simple_qa`
- `summarization`
- `creative_writing`
- `coding`
- `math_reasoning`
- `complex_reasoning`
- `unknown`

Do not assume category equals difficulty. For example, a simple code syntax question and a difficult concurrency bug are both coding tasks but may deserve different tiers.

### Baseline routing policy

Start with transparent, deterministic rules for an initial baseline. You may consider task hints, simple metadata, prompt length, and obvious complexity cues. Avoid a naive policy that routes every prompt containing `code` or `python` to a strong model. Add test examples that expose such mistakes.

The rule baseline is intentionally simple; later results will show where it fails.

### Acceptance criteria

- `POST /route` returns a structured decision.
- It works without `.env`, API keys, or a network connection after dependencies are installed.
- Same prompt + same strategy produces the same decision.
- Tests cover empty prompts, normal prompts, code questions, summarization, complex reasoning, and ambiguous inputs.

Commit: `feat: add route-only endpoint and rule baseline`

---

## Phase 3 — Real model dispatch and `/chat` (4–6 hours)

### Tasks

1. Implement configuration for the selected cheap/strong model IDs and provider API keys.
2. Create `.env.example` with placeholder values only; create `.env` locally with your real values.
3. Implement `app/providers/dispatcher.py` to call a model through LiteLLM.
4. Make provider code separate from routing policy.
5. Add `POST /chat`: validate request → obtain route decision → map tier to model config → call model → return response and execution metadata.
6. Add a mock/fake dispatcher so tests can exercise `/chat` without a real API call.
7. Add predictable error responses for missing configuration, timeouts, rate limits, provider errors, and unexpected failures. Never return stack traces or secrets to the client.
8. Check model API terms and current free quota before sending a real request.

### Suggested response contract

The exact schema can evolve, but the response should expose fields comparable to:

```json
{
  "request_id": "generated-unique-id",
  "answer": "Actual model response text",
  "routing": {
    "strategy": "rules",
    "category": "summarization",
    "tier": "cheap",
    "model": "configured/model-id",
    "reason": "Short reason for the route"
  },
  "execution": {
    "latency_ms": 1234,
    "input_tokens": null,
    "output_tokens": null,
    "total_tokens": null,
    "estimated_cost_usd": null,
    "fallback_used": false,
    "status": "success"
  }
}
```

Numbers above are only illustrative; do not hardcode them. Use `null` or an omitted field when a provider does not return a measurement. Never invent token counts.

### Security requirements

- `.env` must be ignored by Git.
- `.env.example` contains no real credentials.
- Do not print API keys to logs.
- Do not paste secrets into ChatGPT/Claude conversations. Share redacted errors instead.
- Do not add a paid provider requirement to the project simply to satisfy the architecture.

### Acceptance criteria

- A configured valid model can generate a real response through `/chat`.
- Route metadata and answer are both returned.
- Automated tests use a mock dispatcher and do not spend API quota.
- A missing key/configuration causes a clear controlled error.
- A real response is tested manually only when a working API or local model is available.

Commit: `feat: dispatch routed prompts to a model`

---

## Phase 4 — Two distinct model routes, configuration, and fallback (3–4 hours)

### Tasks

1. Configure distinct target models for `cheap` and `strong` tiers. Do not claim meaningful live routing if both tiers point to the same model; same-model configuration can still be useful temporarily for testing plumbing.
2. Keep the mapping in one model registry/configuration module or settings object.
3. Add one fallback model/deployment if a genuinely available second target exists.
4. Implement a small, explicit fallback policy for recoverable errors (for example, selected transient rate-limit/server/time-out errors). Do not fallback on authentication/configuration errors as if retrying would solve them.
5. Limit attempts; do not create infinite retry loops.
6. Record the original failure and whether a fallback succeeded.
7. Test all paths using mocks: primary success, fallback success, both providers fail, invalid configuration.

### Important trade-off

A fallback can improve reliability, but falling back from a cheap model to a more expensive one changes cost. Record that event; do not hide it from cost/quality analysis.

### Acceptance criteria

- Different tiers can resolve to different configured models.
- A model can fail without crashing the whole API process.
- Fallback behavior is finite, deterministic, observable, and covered by tests.
- The API reports the model that actually produced the response, not merely the originally selected model.

Commit: `feat: add model registry and bounded fallback`

---

## Phase 5 — Semantic routing (4–6 hours)

### Purpose

Move beyond literal keyword rules. Use text embeddings to compare an incoming prompt with descriptions or examples of task categories. The embedding model runs locally; it does not need to call a hosted LLM provider for every routing decision.

### Tasks

1. Install `sentence-transformers` only at this phase.
2. Load `sentence-transformers/all-MiniLM-L6-v2` locally.
3. Create short, representative route descriptions/examples for the task categories.
4. Pre-compute embeddings for the descriptions/examples at startup or cache them; avoid re-encoding the entire route library on every request.
5. Encode the incoming prompt and compute cosine similarities.
6. Select the most similar category, store the similarity score, and apply a separate policy to map that category/other signals to `cheap` or `strong`.
7. Define a low-confidence behavior: e.g. use a conservative default, or return `unknown` and use an explicit fallback policy. Do not blindly trust the top similarity when all scores are low.
8. Add unit tests around the policy and test the embedding component separately where practical.
9. Compare semantic routing against the rule baseline on the same prompt set.

### Critical caveat

Semantic similarity is not the same thing as task difficulty or model quality. A prompt may be semantically about coding but still be simple. Do not claim that an embedding router is inherently “intelligent” just because it uses vectors. The evaluation must demonstrate where it helps and where it fails.

### Acceptance criteria

- The semantic strategy can run with locally loaded embeddings and no per-request embedding API bill.
- The route decision explains the predicted category and includes the similarity score where useful.
- Low-similarity/ambiguous inputs have a defined behavior.
- The same evaluation set is used to compare rule and semantic routing.

Commit: `feat: add local semantic routing`

---

## Phase 6 — SQLite observability and cost estimation (3–4 hours)

### Tasks

1. Define a SQLite table for request events.
2. Store one record per completed/failed inference, including:
   - timestamp and request ID
   - chosen routing strategy, category, tier, and reason
   - configured model selected and model that actually returned the answer
   - latency in milliseconds
   - input/output/total token counts when returned by the provider
   - estimated cost (nullable when usage/pricing is unavailable)
   - success/error status
   - whether fallback was used
3. Do not store API keys or other secrets.
4. Avoid storing raw user prompts by default. If a local demo needs to show request text, add an explicit configurable opt-in and document it.
5. Add a function for calculating estimated cost based on usage and a clearly maintained model price table. Model prices can change; keep them in configuration/data instead of burying them in routing code.
6. Expose a `GET /metrics/summary` endpoint with aggregate counts and averages.
7. Check that failed requests are recorded too.

### Metrics worth reporting

- Total requests and successful requests
- Error rate
- Requests by route strategy and tier
- Average/median latency by tier or model
- Fallback count/rate
- Token usage when available
- Estimated cost per request and in aggregate when calculable
- Baseline estimated cost if all prompts had been routed to the stronger model
- Estimated cost under the router
- Estimated cost difference and percentage difference, only when the comparison data is complete enough

### Cost calculation discipline

If the provider provides token counts, use them. If not, mark the usage and/or cost unavailable unless you implement and disclose an approximation. Keep separate concepts clear:

- **Actual billed cost:** what the provider charged; may be zero under a free quota and may not be directly observable from the API response.
- **Estimated production cost:** calculated from tokens and a stated price schedule.
- **Estimated saving:** comparison between baseline and router under that same price schedule. It is not automatically a cash saving that has already happened.

### Acceptance criteria

- Successful and failed calls generate database records.
- Metrics can be obtained without opening the database manually.
- Missing token usage or missing price data does not crash metric calculation.
- README describes how cost estimates are calculated and what they do not prove.

Commit: `feat: record requests and expose metrics`

---

## Phase 7 — Benchmark and evaluation (4–6 hours)

### Goal

Show whether the router actually produces a useful quality/cost trade-off.

### Dataset

Start with 30–50 fixed prompts to validate the script, then expand to roughly 100–200 prompts if time and free quota allow. Use varied categories such as coding, summarization, simple factual questions, creative writing, math, and multi-step reasoning. Keep the prompt set versioned and do not change it silently between comparisons.

Each JSONL row can include:

```json
{
  "id": "coding_001",
  "category": "coding",
  "prompt": "Explain what a race condition is in a multithreaded program.",
  "expected_tier": "strong",
  "evaluation_notes": "Should accurately explain concurrent access and nondeterministic behavior."
}
```

This is an example schema, not a label that has already been validated. Expected tiers should be considered an initial hypothesis, not absolute truth.

### Required comparisons

1. **Always cheap:** send every evaluation prompt to the cheaper model.
2. **Always strong:** send every evaluation prompt to the stronger model.
3. **Rule router:** let the rules strategy choose.
4. **Semantic router:** let the semantic strategy choose.

Run all policies on the same prompts, model configuration, and evaluation conditions where possible. Record failed prompts as failures; do not silently drop them.

### What to measure

- Routing distribution (percentage routed to each tier)
- Failure and fallback rates
- Median and average latency; report the sample size
- Estimated cost for the same set of prompts
- Response quality
- How often a cheap route appears to miss the required quality level

### Quality evaluation without paying another model

Start with a manual score rubric for a sample of answers, such as 1–5 for correctness, relevance, completeness, and instruction following. Use the same rubric across strategies. Where tasks have objective results, add deterministic checks. An LLM-as-judge can be a later extension, but it can add quota/cost and introduce evaluator bias.

Do not rely only on route classification accuracy. The final question is whether the selected model answers well at an acceptable cost and latency.

### Cost comparison methodology

Use the same configured pricing table for each model and apply it to observed input/output tokens. If price data or token usage is missing, disclose omissions and avoid definitive saving claims. If the only current API usage is free, show estimated production pricing separately from current cash spend.

### Acceptance criteria

- The benchmark runs reproducibly from a command.
- It writes a CSV/JSON results file.
- Baselines and router strategies run over the same prompt IDs.
- Results state the model IDs, dataset version, time/date, price assumptions, sample sizes, and failed calls.
- The README can honestly summarize a real result, even if the router does not beat all baselines.

Commit: `feat: add routing benchmark and quality evaluation`

---

## Phase 8 — Streamlit dashboard/demo (3–4 hours)

### Tasks

1. Create `ui/streamlit_app.py`.
2. Add a text box for the prompt.
3. Add a strategy selector: `rules`, `semantic`, and possibly `auto` once that mode exists.
4. Add “Preview route” button that calls `POST /route`.
5. Add “Generate answer” button that calls `POST /chat`.
6. Show answer, chosen tier/model, routing explanation, latency, usage/cost when available, and fallback status.
7. Add a metrics area based on `GET /metrics/summary`.
8. Make API-not-running/provider-quota failure messages understandable.
9. Do not put API provider keys in Streamlit source code or public UI.

### Run UI

Keep FastAPI running in one terminal, then in a second terminal from the repository root:

```powershell
streamlit run ui/streamlit_app.py
```

### Acceptance criteria

- A new user can type a prompt and receive an answer when a model is configured.
- Route preview works independently of live inference.
- Execution metadata and aggregate metrics are understandable.
- UI failures are readable and do not expose secrets or stack traces.

Commit: `feat: add Streamlit demo and metrics view`

---

## Phase 9 — README, reproducibility, and final review (2–3 hours)

### README must contain

1. Project title and one-paragraph description.
2. Problem statement and why routing matters.
3. Architecture diagram.
4. Features with accurate scope.
5. Stack and prerequisites.
6. Setup instructions for Windows PowerShell (and optionally Linux).
7. How to configure `.env` without revealing real secrets.
8. How to run backend, tests, benchmark, and UI.
9. API endpoint examples and sample response (fake/sample data clearly labelled).
10. Evaluation methodology and genuine measured findings.
11. Cost-estimate assumptions and free-tier caveats.
12. Known limitations and future improvements.
13. Screenshot(s) or a short demo GIF/video if available.

### Final checks

- Run all tests from a clean activated environment.
- Check `git status` and confirm `.env`, API keys, databases containing user data, and private outputs are not accidentally committed.
- Ensure README setup steps work from the repository root.
- Run one end-to-end request manually if a working model endpoint is configured.
- Run a small benchmark and keep its output.
- Check that the router's explanation matches the actual selected model.
- Make sure measured results support every claim in the README.

Commit: `docs: document setup architecture and benchmark results`

---

# 9. API design summary

The precise schema is allowed to evolve, but these endpoints are recommended.

| Endpoint | Purpose | Requires live model? |
|---|---|---:|
| `GET /health` | Check API availability | No |
| `POST /route` | Return a route decision only | No |
| `POST /chat` | Route, call model, return answer, record telemetry | Yes, except tests using a mock dispatcher |
| `GET /metrics/summary` | Return aggregated SQLite metrics | No additional inference |

### Example request to `POST /route`

```json
{
  "prompt": "Summarize this paragraph in three bullet points.",
  "strategy": "rules"
}
```

### Example request to `POST /chat`

```json
{
  "prompt": "Explain the difference between a process and a thread.",
  "strategy": "semantic"
}
```

Do not use these examples as a reason to hard-code the answer or route. The API should compute it.

---

# 10. Testing checklist

Tests should not require paid API calls or consume a free quota. Use a fake/mock dispatcher for routine tests.

## Routing tests

- Empty or whitespace-only prompt is rejected.
- Short general question can route according to policy.
- Summarization prompt is recognized where the policy supports it.
- Simple coding question is not automatically considered complex just because it mentions code.
- A multi-step reasoning prompt can use the strong tier according to the policy.
- Ambiguous prompt has a defined default.
- Semantic route returns category and score.
- Low semantic similarity does not create false certainty.
- Same input and same strategy give a stable result unless an intentionally stochastic component is later introduced.

## API tests

- `/health` returns 200.
- `/route` works with no API configuration.
- Invalid payload returns appropriate 4xx validation response.
- `/chat` with a fake dispatcher returns a structured response.
- Provider exception becomes a controlled API error.
- No raw secret or traceback leaks to clients.

## Dispatcher/fallback tests

- Primary success returns its output.
- Recoverable primary failure triggers only the configured bounded fallback.
- Authentication/configuration failure does not loop through retries endlessly.
- Both primary and fallback failure are recorded clearly.
- Response reports the model that actually generated it.

## Storage/metrics tests

- Successful call creates a record.
- Failed call creates a record if the request reached execution.
- Null token usage does not break cost math.
- Missing price config yields unknown/null cost instead of a fabricated value.
- Summary counts match inserted mock records.

---

# 11. Evaluation principles and honesty standards

This project becomes valuable through evidence. Follow these standards:

1. Do not claim “X% cheaper” unless the comparison uses a documented pricing table, observed or explicitly estimated token usage, and comparable prompts.
2. Do not claim “same quality” based on one or two examples. Score a fixed sample using a consistent rubric.
3. Do not call latency measurements universal. State hardware/provider, sample size, and that free-tier congestion can affect timings.
4. Do not invent token counts when a provider omits usage.
5. Do not describe a similarity score as a calibrated probability.
6. Do not equate predicted task category with required reasoning level.
7. Do not remove failed requests from result tables without reporting them.
8. Include negative results if the router performs worse than a baseline; a well-analyzed failure is still engineering evidence.
9. Label local demo results, mock-provider tests, actual API results, and estimated production cost distinctly.
10. The router must explain the decision using information it actually computed, not a post-hoc fabricated reason.

---

# 12. Stretch features (only after the MVP works)

Priority order:

1. **Manual feedback/evaluation UI:** allow a user to rate the answer; save score against request ID.
2. **Budget/latency policy:** let a request prefer cheapest, balanced, or strongest within a simple cap. Do not implement complex user-level budgets before basic policy evaluation.
3. **Confidence-aware routing:** route to strong on uncertainty or when semantic similarity is low.
4. **Learned routing:** investigate RouteLLM or a small classifier as a comparison only after a reliable benchmark exists.
5. **Streaming responses:** provide partial tokens to the UI.
6. **Langfuse/OpenTelemetry integration:** only after the SQLite observability layer is useful.
7. **Deployment:** choose a free hosting option only after checking current limits and whether the selected inference provider is accessible from it.

Do not add stretch features at the expense of a tested core.

---

# 13. Git workflow and collaboration rules

- Commit a working baseline before each major feature.
- Use descriptive commit messages, e.g. `feat: add semantic routing`.
- Avoid letting two AI assistants independently rewrite the same files at the same time.
- Keep one local repository as the source of truth; after applying AI-generated code, run tests before asking the next assistant to modify it.
- When an AI suggests changing the architecture, ask what requirement the change serves and whether it adds complexity to the MVP.
- Never paste API secrets into prompts, logs, screenshots, or GitHub.
- When asking for debugging, provide the exact command run, the complete relevant error (with secrets redacted), operating system, Python version, and current phase.
- If code generation produces a large multi-file patch, inspect each file and test incrementally rather than accepting everything blindly.

---

# 14. Definition of done

The MVP is portfolio-ready when all of the following are true:

- [ ] A clean setup procedure is written and reproducible.
- [ ] FastAPI starts and `/health` succeeds.
- [ ] `/route` works without live API credentials.
- [ ] Rule-based and semantic strategies exist and are comparable.
- [ ] `/chat` returns an actual LLM response using a configured model endpoint.
- [ ] At least two distinct model configurations exist for meaningful routing when the available free/local setup permits it; limitations are disclosed if not.
- [ ] Dispatcher behavior, errors, and bounded fallbacks are tested using mocks.
- [ ] SQLite logs success and failure metadata without storing secrets.
- [ ] Metrics report latency, failures/fallbacks, token usage when supplied, and estimated cost when calculable.
- [ ] Benchmark compares always-cheap, always-strong, rule, and semantic policies on the same fixed prompts.
- [ ] Quality is assessed using a documented consistent method.
- [ ] Streamlit demonstrates route preview, generated answers, and metrics.
- [ ] README includes setup, architecture, limitations, methodology, and actual findings.
- [ ] Git history is clean; `.env` and secrets are not committed.

If no second working model can be configured with free/local resources, the software should still be testable using a fake dispatcher, but the README must explicitly state that live multi-model cost/quality comparison remains incomplete.

---

# 15. Current state and project decision log

Update this section as decisions are made. Do not let a future assistant assume a decision has happened if the item is still marked undecided.

- [x] Working title: RouteWise — Cost- and Quality-Aware LLM Router (provisional)
- [x] Main product decision: build route-only and route-and-generate; route-and-generate is the primary demo
- [x] API framework: FastAPI
- [x] Provider abstraction: LiteLLM Python SDK
- [x] Routing strategies planned: rules first, then local semantic matching
- [x] Observability planned: SQLite first
- [x] UI planned: Streamlit, after API/metrics work
- [x] Evaluation planned: fixed prompts, baselines, human/deterministic quality checks, estimated cost and latency
- [ ] Local project folder created
- [ ] Python version confirmed
- [ ] Virtual environment created
- [ ] Git repository initialized
- [ ] Provider chosen and free quota confirmed (if a hosted API is used)
- [ ] Actual model IDs selected
- [ ] First end-to-end response generated
- [ ] Semantic routing implemented
- [ ] Benchmark results collected
- [ ] UI completed

**Current phase:** Phase 0 — not started (change this when work actually begins).  
**Current blocker:** None known yet; local environment and accessible model endpoint have not been confirmed in this document.  
**Latest verified test result:** None yet. Do not claim any tests pass until they have actually been run.

---

# 16. Instructions for any AI assistant helping with this project

You are acting as a technical mentor and pair programmer for a student building RouteWise, a cost- and quality-aware LLM router. Read this entire context document before proposing work.

## Rules for your assistance

1. Preserve the project's agreed purpose and layered architecture unless there is a concrete technical reason to change it.
2. Help the student understand what each layer does, not just produce code.
3. Work **one small phase or testable milestone at a time**. Do not generate the entire project in one giant answer.
4. Before writing code, state the files that will be created/changed and the intended behavior.
5. Give complete copy-paste-ready code for each file you ask the student to create or replace. Clearly identify the exact repository-relative path.
6. Use Windows PowerShell commands by default; also provide Linux equivalents if they differ materially. The student may use Windows or Pop!_OS, so confirm the active environment from the current conversation rather than assuming.
7. Do not ask the student to repeat information already present here. Ask only for an essential missing detail, such as the exact error or provider model ID.
8. Do not assume a file exists, a command succeeded, a dependency installed, an API key is available, a test passed, or a provider model is free. Ask for or inspect the actual result.
9. Never invent current free-tier quota information or a valid model ID. Check the current official documentation when needed, and keep model IDs configurable.
10. Use mocks for automated tests. Do not make routine tests spend API quota.
11. When debugging, identify the likely cause, explain it plainly, provide a minimal fix, and give the command to verify the fix.
12. If more than one solution is possible, recommend the simplest option meeting the current requirement and explain any trade-off. Avoid premature abstractions and unnecessary dependencies.
13. Keep secrets private. Never request that real API keys be pasted into the chat. If `.env` is needed, tell the student how to edit it locally with placeholder examples.
14. Never describe estimated costs as actual bills or estimated savings as proven cash savings.
15. After each milestone, provide a short checkpoint: what changed, exact command to test it, expected output, common failure signs, and the next milestone. Update the state/checklist when the student confirms it worked.
16. Do not silently change the architecture or add scope. If proposing a stretch feature, explain why the MVP is already ready for it and ask whether it is worth the added work.

## Exact initial task for the next AI

Start with **Phase 0 only**. First summarize the project in a few lines to confirm understanding. Then help the student inspect their current environment and create the repository, virtual environment, `.gitignore`, README, and project context file. Do not jump directly into semantic routing or the dashboard. Confirm the setup works before proceeding to Phase 1.

---

# 17. Copy/paste prompt to bring Claude (or another AI) up to speed

Copy the prompt below and attach/paste this entire document after it:

> I am building RouteWise, a cost- and quality-aware LLM router, and the attached project-context document is the source of truth. Read it completely before helping.
>
> The idea is to build a FastAPI backend that routes each prompt to a configured LLM tier, uses LiteLLM to call the selected model, returns the actual model answer plus routing/execution metadata, records useful metrics in SQLite, and compares routing policies using a fixed benchmark. We will implement a rule-based baseline first, then local semantic routing using Sentence Transformers, then logging/evaluation, then a Streamlit UI. The main demo must generate a real answer; route-only mode is a debugging feature too.
>
> Keep the implementation free-first. Do not assume I have paid API credits, working API keys, a specific model ID, or suitable hardware for local inference. Keep model names and keys configurable. Use mock dispatchers for tests. Never request that I paste secrets into chat.
>
> Guide me phase by phase, starting with Phase 0 only. Use complete copy-paste-ready files and exact commands. Explain what each step is doing and how I verify it. Do not generate the whole project at once, skip validation, or claim a command/test succeeded without seeing its output. Keep scope aligned with the attached plan and update it when we confirm progress.
>
> My current status is whatever the “Current state and project decision log” section says. Do not assume unmarked setup or implementation tasks are complete.

---

# 18. Progress log template

After each session, append a short entry here.

```text
Date:
Phase/milestone:
Files changed:
Commands run:
Tests run and exact results:
What is now working:
Known issue/blocker:
Next task:
```
