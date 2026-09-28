# AI Analytics Engine

MSBA 6131 Project Demo: an end-to-end, AI-powered analytics engine that runs on your own machine.

Upload a dataset, ask a question in plain language, and an AI data scientist (Claude) writes and runs Python against your data. It explores, visualizes, tests hypotheses and builds predictive models, then explains its findings. Every step is visible: the code, its output, the charts, and any files it produces.

## Quick start

```bash
uv sync                          # install dependencies into .venv
cp .env.example .env             # then put your Anthropic API key in .env
uv run analytics-engine          # start the server
```

Open http://127.0.0.1:8000, upload a file (try `samples/customer_churn.csv`), and ask something like *"What drives churn? Build a model to predict it."*

## What it does

| Capability | Examples |
|---|---|
| Exploratory analysis | data quality checks, distributions, segment comparisons, correlations |
| Visualization | trends, comparisons, relationships, heatmaps (matplotlib/seaborn, consistent theme) |
| Statistics | hypothesis tests with effect sizes (scipy, statsmodels) |
| Predictive modeling | classification/regression with scikit-learn pipelines, held-out evaluation, baselines, feature importance |
| Outputs | charts inline; generated files (predictions, cleaned data, models) offered as downloads |

Supported uploads: CSV, TSV, TXT, Excel (`.xlsx/.xls/.xlsm`), JSON/JSONL and Parquet. Each upload is profiled right away (rows, columns, missing values, duplicates, column summaries) and loaded into the session as a pandas DataFrame.

## How it works

```
Browser (chat UI)  ──POST /chat, /upload──▶  FastAPI server  ──▶  Claude (Anthropic API)
       ▲                                         │   ▲                  │ run_python(code)
       └────────── SSE event stream ─────────────┘   │                  ▼
                                                     └── results ── Jupyter kernel (per session)
```

- **Agent loop** ([analytics_engine/agent.py](analytics_engine/agent.py)): Claude gets one tool, `run_python`. It plans, runs code, reads the output (including the rendered charts, as images), fixes errors and iterates, then writes a Markdown answer. Thinking, text and code stream to the UI live.
- **Execution** ([analytics_engine/kernel.py](analytics_engine/kernel.py)): each chat session has its own persistent IPython kernel in the project's uv environment. State carries over between questions like a notebook. Cells time out after 5 minutes, and **Stop** interrupts them.
- **Sessions** ([analytics_engine/session.py](analytics_engine/session.py)): each session keeps the conversation history, its workspace folder (`workspace/<session-id>/` with `data/`, `figures/` and any outputs) and an event log. Reloading the page replays the session.
- **Server** ([analytics_engine/server.py](analytics_engine/server.py)): REST endpoints plus a Server-Sent Events stream per session.
- **Front-end** ([analytics_engine/static/](analytics_engine/static/)): a dependency-free single page (vanilla JS, no build step). Markdown is rendered with marked + DOMPurify and code is highlighted with highlight.js. It supports drag-and-drop upload, light/dark mode and mobile layouts.

## Configuration

Set these in `.env` (see [.env.example](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | required |
| `ANALYST_MODEL` | `claude-opus-5` | Claude model |
| `ANALYST_EFFORT` | `high` | `low`–`max`; lower is faster and cheaper |
| `ANALYST_FALLBACKS` | `true` | server-side fallback model if a request is declined |
| `ANALYST_EXEC_TIMEOUT` | `300` | seconds per code cell |
| `ANALYST_MAX_STEPS` | `30` | max code cells per question |
| `ANALYST_MAX_UPLOAD_MB` | `200` | upload size limit |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind address |

## Security note

The engine executes AI-generated Python code on your machine with your user's permissions. That is the point of the tool, but it means:
- keep it bound to `127.0.0.1` (the default); don't expose it on a network;
- only upload data you're comfortable sending to the Anthropic API (column summaries, sample rows and analysis outputs are sent to the model).
