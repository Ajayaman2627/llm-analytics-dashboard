import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from database import get_engine, get_session, LLMCall, MODEL_COSTS

load_dotenv()

def _get_token() -> str:
    """Read token from .env locally or Streamlit secrets in production."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        try:
            import streamlit as st
            token = st.secrets["GITHUB_TOKEN"]
        except Exception:
            pass
    if not token:
        raise ValueError("GITHUB_TOKEN not found in .env or Streamlit secrets")
    return token

# Single shared engine so all calls use the same DB file
_engine = get_engine()

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "Llama-3.2-11B-Vision-Instruct",
    "mistral-small-2503",
]

def _get_client():
    return OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=_get_token(),
    )

def call_llm(prompt: str, model: str = "gpt-4o-mini",
             system: str = "You are a helpful assistant.",
             session_label: str = "default",
             max_tokens: int = 500) -> dict:
    """
    Send a prompt to a model, log the call to the DB, and return results.

    Returns a dict with: model, prompt, response, tokens, latency_ms, cost_usd
    """
    client = _get_client()

    start = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    latency_ms = (time.time() - start) * 1000

    response_text  = completion.choices[0].message.content or ""
    prompt_tokens  = completion.usage.prompt_tokens
    response_tokens = completion.usage.completion_tokens
    total_tokens   = completion.usage.total_tokens

    cost_per_1k    = MODEL_COSTS.get(model, 0.0)
    cost_usd       = (total_tokens / 1000) * cost_per_1k

    # Persist to SQLite
    session = get_session(_engine)
    record = LLMCall(
        model=model,
        prompt=prompt,
        response=response_text,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 2),
        cost_usd=round(cost_usd, 6),
        session_label=session_label,
    )
    session.add(record)
    session.commit()
    session.close()

    return {
        "model":           model,
        "prompt":          prompt,
        "response":        response_text,
        "prompt_tokens":   prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens":    total_tokens,
        "latency_ms":      round(latency_ms, 2),
        "cost_usd":        round(cost_usd, 6),
    }


def compare_models(prompt: str, models: list = None,
                   session_label: str = "comparison") -> list[dict]:
    """
    Send the same prompt to multiple models and return all results.
    Used for the side-by-side comparison feature.
    """
    if models is None:
        models = AVAILABLE_MODELS

    results = []
    for model in models:
        try:
            result = call_llm(prompt, model=model, session_label=session_label)
            results.append(result)
        except Exception as e:
            results.append({
                "model": model,
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "prompt_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0,
                "cost_usd": 0,
            })
    return results
