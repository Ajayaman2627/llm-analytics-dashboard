# LLM Analytics Dashboard

A full-stack analytics platform for benchmarking Large Language Models (LLMs) on cost, latency, and response quality — built with Python, Streamlit, and the GitHub Models API.

## Live Demo
🔗 [Deploy link will go here after Streamlit Cloud setup]

## Features

- **Multi-Model Comparison** — Send any prompt to GPT-4o, GPT-4o-mini, Llama 3.2, and Mistral simultaneously and compare responses side by side
- **NLP Quality Scoring** — Every response is scored on 5 dimensions: Relevance, Completeness, Conciseness, Readability, and Confidence using keyword overlap and linguistic heuristics
- **Live Analytics Dashboard** — Plotly charts for latency trends, token usage, cost breakdown, and token efficiency per model
- **Persistent Call Logging** — Every API call is saved to SQLite via SQLAlchemy with full metadata (tokens, cost, latency, timestamp)
- **Exportable History** — Filter and export call logs as CSV

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + Plotly |
| Backend | Python 3.9 |
| LLM API | GitHub Models (GPT-4o, GPT-4o-mini, Llama 3.2, Mistral) |
| Database | SQLite + SQLAlchemy ORM |
| NLP Scoring | Custom evaluator (no external ML library needed) |

## Architecture

```
┌─────────────────────────────────┐
│         Streamlit UI (app.py)   │
└────────────┬────────────────────┘
             │
     ┌───────▼───────┐     ┌────────────────┐
     │  llm_client   │────▶│  GitHub Models │
     │  (API layer)  │     │  API           │
     └───────┬───────┘     └────────────────┘
             │ logs every call
     ┌───────▼───────┐
     │  SQLite DB    │
     │  (SQLAlchemy) │
     └───────┬───────┘
             │
   ┌─────────▼──────────┐   ┌──────────────────┐
   │  analytics.py      │   │  evaluator.py     │
   │  (metrics queries) │   │  (NLP scoring)    │
   └────────────────────┘   └──────────────────┘
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-analytics-dashboard.git
cd llm-analytics-dashboard

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your GitHub token
echo "GITHUB_TOKEN=your_github_token_here" > .env
# Get a free token at github.com/settings/tokens (no scopes needed)

# 5. Run the app
streamlit run app.py
```

## NLP Scoring Methodology

Each response is evaluated on 5 dimensions using pure Python NLP (no external ML models):

| Dimension | Method | Weight |
|---|---|---|
| **Relevance** | Jaccard similarity between prompt and response keywords | 30% |
| **Completeness** | Response length relative to prompt complexity | 25% |
| **Conciseness** | Penalizes padding; rewards focused responses | 20% |
| **Readability** | Avg sentence length + avg word length heuristic | 15% |
| **Confidence** | Detects hedge phrases ("I think", "maybe", "unclear") | 10% |

Final score = weighted average → 0–100

## Project Structure

```
llm-analytics-dashboard/
├── app.py           # Streamlit frontend (3 tabs)
├── llm_client.py    # API wrapper + call logger
├── database.py      # SQLAlchemy schema
├── analytics.py     # Metric queries
├── evaluator.py     # NLP quality scorer
└── requirements.txt
```

## Author

**Ajayaman Kantumuchu** — MS Computer Science, Cal State San Bernardino  
[LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) · [GitHub](https://github.com/YOUR_USERNAME)
