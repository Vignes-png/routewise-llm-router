

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

