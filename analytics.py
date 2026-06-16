import pandas as pd
from sqlalchemy import text
from database import get_engine

_engine = get_engine()

def _query(sql: str) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ── Summary cards (the 4 numbers shown at the top of the dashboard) ──────────

def get_summary_stats() -> dict:
    df = _query("SELECT * FROM llm_calls")
    if df.empty:
        return {"total_calls": 0, "total_tokens": 0,
                "total_cost_usd": 0.0, "avg_latency_ms": 0.0}
    return {
        "total_calls":    len(df),
        "total_tokens":   int(df["total_tokens"].sum()),
        "total_cost_usd": round(df["cost_usd"].sum(), 6),
        "avg_latency_ms": round(df["latency_ms"].mean(), 1),
    }


# ── Per-model breakdowns ──────────────────────────────────────────────────────

def get_model_comparison() -> pd.DataFrame:
    """Average latency, tokens, and cost grouped by model."""
    sql = """
        SELECT
            model,
            COUNT(*)                        AS total_calls,
            ROUND(AVG(latency_ms), 1)       AS avg_latency_ms,
            ROUND(AVG(total_tokens), 1)     AS avg_tokens,
            ROUND(AVG(response_tokens), 1)  AS avg_response_tokens,
            ROUND(SUM(cost_usd), 6)         AS total_cost_usd
        FROM llm_calls
        GROUP BY model
        ORDER BY avg_latency_ms ASC
    """
    return _query(sql)


# ── Time-series data (for line/bar charts) ────────────────────────────────────

def get_calls_over_time() -> pd.DataFrame:
    """Number of calls per hour bucket — shows usage patterns."""
    sql = """
        SELECT
            strftime('%Y-%m-%d %H:00', timestamp) AS hour,
            COUNT(*)                               AS call_count,
            SUM(total_tokens)                      AS tokens_used
        FROM llm_calls
        GROUP BY hour
        ORDER BY hour ASC
    """
    return _query(sql)


def get_latency_over_time() -> pd.DataFrame:
    """Latency trend per call — useful for spotting model slowdowns."""
    sql = """
        SELECT
            id,
            timestamp,
            model,
            latency_ms
        FROM llm_calls
        ORDER BY timestamp ASC
    """
    return _query(sql)


# ── Token efficiency ──────────────────────────────────────────────────────────

def get_token_efficiency() -> pd.DataFrame:
    """
    Response tokens / prompt tokens ratio per model.
    Higher = model gives longer answers for the same input.
    Useful for understanding verbosity vs cost tradeoff.
    """
    sql = """
        SELECT
            model,
            ROUND(AVG(CAST(response_tokens AS FLOAT) /
                  NULLIF(prompt_tokens, 0)), 2)  AS response_ratio,
            ROUND(AVG(prompt_tokens), 1)          AS avg_prompt_tokens,
            ROUND(AVG(response_tokens), 1)        AS avg_response_tokens
        FROM llm_calls
        GROUP BY model
    """
    return _query(sql)


# ── Call history (for the searchable log table) ───────────────────────────────

def get_call_history(limit: int = 50, model_filter: str = None) -> pd.DataFrame:
    where = f"WHERE model = '{model_filter}'" if model_filter else ""
    sql = f"""
        SELECT
            id,
            timestamp,
            model,
            SUBSTR(prompt, 1, 80)   AS prompt_preview,
            SUBSTR(response, 1, 120) AS response_preview,
            total_tokens,
            latency_ms,
            cost_usd,
            session_label
        FROM llm_calls
        {where}
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    return _query(sql)


# ── Cost breakdown ────────────────────────────────────────────────────────────

def get_cost_by_model() -> pd.DataFrame:
    sql = """
        SELECT
            model,
            SUM(cost_usd)  AS total_cost_usd,
            COUNT(*)       AS call_count,
            ROUND(SUM(cost_usd) / NULLIF(COUNT(*), 0), 8) AS cost_per_call
        FROM llm_calls
        GROUP BY model
        ORDER BY total_cost_usd DESC
    """
    return _query(sql)
