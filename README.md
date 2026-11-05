# RouteWise

**A multi-mode LLM gateway for model selection, inference, and request observability.**

RouteWise provides a FastAPI gateway and Streamlit playground for sending prompts to Gemini model variants through LiteLLM.

> **Status:** Working student prototype. Fast, Quality, and Auto modes are implemented. Auto uses a semantic-plus-rules heuristic; it has not been proven to outperform fixed-model selection on a sufficiently large held-out evaluation.

## Features

- **Fast:** Routes directly to the configured cheap tier (Gemini Flash-Lite).
- **Quality:** Routes directly to the configured strong tier (Gemini Flash).
- **Auto:** Uses semantic task classification and rule-based tier selection.
- **FastAPI + LiteLLM:** Validates requests, dispatches model calls, and returns structured responses.
- **Streamlit playground:** Lets users select a mode and view answers, model details, routing explanations, latency, and token usage.
- **SQLite observability:** Records request metadata, status, selected model, latency, token usage, estimated cost, and error category. It does not store prompts, answers, API keys, or raw provider errors.
- **Tests and evaluation scripts:** Cover API behaviour and support paired comparisons of the configured models.

## Tech stack

Python 3.11 · FastAPI · Uvicorn · Streamlit · LiteLLM · Google Gemini API · Sentence Transformers (`all-MiniLM-L6-v2`) · SQLite · Pydantic · Pytest

## Project structure

```text
routewise-llm-router/
├── app/              # API, schemas, routing, dispatch, observability
├── frontend/         # Streamlit playground
├── .streamlit/       # Streamlit theme
├── data/             # Evaluation dataset and metadata
├── reports/          # Benchmark output CSVs
├── scripts/          # Dataset and benchmark scripts
├── tests/            # Automated tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting started

### Prerequisites

- Git
- Conda
- Python 3.11
- A Gemini API key

### 1. Clone the repository

```powershell
git clone https://github.com/Vignes-png/routewise-llm-router.git
cd routewise-llm-router
```

### 2. Create and activate the environment

```powershell
conda create -n routewise python=3.11 -y
conda activate routewise
```

If the `routewise` environment already exists, just activate it:

```powershell
conda activate routewise
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Keep `.env` private and never commit your API key.

### 5. Start the FastAPI backend

In the first terminal, from the project root:

```powershell
conda activate routewise
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The backend is available at:

- Health check: http://127.0.0.1:8000/health
- API documentation: http://127.0.0.1:8000/docs

### 6. Start the Streamlit frontend

Open a second terminal in the project root:

```powershell
conda activate routewise
python -m streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. The frontend calls the local FastAPI backend. To use another backend URL, set `ROUTEWISE_API_URL` before starting Streamlit.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Basic service information |
| GET | `/health` | Health check |
| POST | `/route` | Rule-based routing decision |
| POST | `/route/semantic` | Semantic classification and tier decision |
| POST | `/chat` | Select a mode, dispatch to a model, and return its answer |

Example `POST /chat` request:

```json
{
  "prompt": "Explain binary search with a simple example.",
  "mode": "fast"
}
```

`mode` accepts `auto`, `fast`, or `quality`; it defaults to `auto` when omitted. A successful response includes the answer, selected model and tier, routing method and reason, latency, optional token usage, and an `X-Request-ID` response header. `task_category` and `similarity_score` are null for manual modes.

## Tests

Run from the project root:

```powershell
python -m pytest -q
```

Tests mock model calls where appropriate, so the test suite should not require live Gemini requests or consume model quota.

## Evaluation

The exploratory paired-model dataset contains 40 prompts:

- **20 GSM8K prompts** from the test split, adapted with a final-answer formatting instruction and evaluated by numerical answer matching.
- **20 IFEval prompts** from the train split, retaining the original prompts and instruction metadata for separate evaluation.

The samples use pinned dataset revisions and a fixed seed. Duplicate IFEval rows at the selected revision were removed by dataset key before sampling.

In the exploratory run, 80 calls were attempted; 78 succeeded after retries, producing 38 complete paired comparisons. In the 19 complete GSM8K pairs, both models answered 18 questions correctly and Flash alone answered one additional question correctly. The IFEval sample is exploratory: local checks are not official IFEval scores; run the official evaluator before reporting official results.

This small sample does **not** establish that Auto generalises or outperforms always selecting one model. Auto remains a semantic-plus-rules heuristic baseline, not a trained predictive router.

## Observability and limitations

SQLite records request ID, timestamp, status, routing method, task category, selected tier/model, latency, token usage, estimated cost, and error category. Cost values are estimates, not billing statements.

RouteWise is a local development prototype. It does not currently provide user authentication, multi-tenant isolation, production rate limiting, or cross-provider routing. Do not expose the local API publicly without adding appropriate security controls.

## Future work

- Run the official IFEval evaluator and publish reproducible results.
- Expand evaluation with a held-out set before making generalisation claims.
- Test whether a learned router adds measurable value over fixed-model baselines.
- Add authentication, quotas, and deployment safeguards if moving beyond a local prototype.

## Responsible use

Keep credentials in environment variables or an untracked `.env` file. Review dataset licences and provider terms before redistributing evaluation data or using RouteWise with sensitive prompts.
