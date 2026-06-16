"""
evaluator.py — NLP-based response quality scorer

Scores each LLM response on 5 dimensions:
  1. Relevance     — keyword overlap between prompt and response
  2. Completeness  — response length relative to prompt complexity
  3. Conciseness   — penalizes unnecessarily long/padded responses
  4. Readability   — sentence structure and avg word length
  5. Confidence    — absence of hedge phrases ("I think", "maybe", "I'm not sure")

Final score: weighted average → 0–100
"""

import re
import math


# ── Stopwords (common words that carry no meaning for relevance) ──────────────

STOPWORDS = {
    "a", "an", "the", "is", "in", "it", "of", "to", "and", "or",
    "for", "on", "with", "as", "by", "at", "be", "was", "are",
    "this", "that", "what", "how", "can", "do", "does", "from",
    "will", "its", "i", "you", "we", "they", "their", "our",
}

HEDGE_PHRASES = [
    "i think", "i believe", "i'm not sure", "i am not sure",
    "maybe", "perhaps", "possibly", "i cannot", "i can't",
    "i don't know", "not certain", "unclear", "it depends",
    "generally speaking", "in general",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words, remove stopwords."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


def _sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]


# ── 5 individual scorers (each returns 0–100) ─────────────────────────────────

def score_relevance(prompt: str, response: str) -> float:
    """
    Jaccard similarity between prompt keywords and response keywords.
    High overlap = model stayed on topic.
    """
    prompt_words    = set(_tokenize(prompt))
    response_words  = set(_tokenize(response))
    if not prompt_words:
        return 50.0
    intersection = prompt_words & response_words
    union        = prompt_words | response_words
    jaccard      = len(intersection) / len(union)
    return round(min(jaccard * 300, 100), 1)   # scale: Jaccard 0.33 → 100


def score_completeness(prompt: str, response: str) -> float:
    """
    Checks if the response is long enough relative to prompt complexity.
    Prompt complexity = number of meaningful keywords.
    """
    prompt_keywords   = len(_tokenize(prompt))
    response_keywords = len(_tokenize(response))
    if prompt_keywords == 0:
        return 50.0
    # Expect ~3 response words per prompt keyword as a baseline
    ratio = response_keywords / (prompt_keywords * 3)
    return round(min(ratio * 100, 100), 1)


def score_conciseness(prompt: str, response: str) -> float:
    """
    Penalizes responses that are far too long for a simple prompt.
    Sweet spot: response is 5–15x the prompt length in characters.
    """
    p_len = max(len(prompt), 1)
    r_len = len(response)
    ratio = r_len / p_len

    if 3 <= ratio <= 20:
        return 100.0
    elif ratio < 3:
        # Too short
        return round(max((ratio / 3) * 100, 10), 1)
    else:
        # Too long — diminishing score after 20x
        penalty = math.log(ratio / 20 + 1) * 40
        return round(max(100 - penalty, 10), 1)


def score_readability(response: str) -> float:
    """
    Simple readability proxy:
    - avg sentence length (shorter = more readable, up to a point)
    - avg word length (shorter words = clearer writing)
    """
    sentences = _sentences(response)
    words     = re.findall(r'\b\w+\b', response)
    if not sentences or not words:
        return 50.0

    avg_sentence_len = len(words) / len(sentences)
    avg_word_len     = sum(len(w) for w in words) / len(words)

    # Ideal: 15–20 words/sentence, 4–6 chars/word
    sentence_score = 100 - abs(avg_sentence_len - 17) * 3
    word_score     = 100 - abs(avg_word_len - 5) * 10

    return round(max((sentence_score + word_score) / 2, 0), 1)


def score_confidence(response: str) -> float:
    """
    Penalizes hedge phrases. Each hedge phrase found docks points.
    Confident, direct responses score higher.
    """
    text  = response.lower()
    hits  = sum(1 for phrase in HEDGE_PHRASES if phrase in text)
    score = max(100 - hits * 20, 0)
    return float(score)


# ── Master evaluator ──────────────────────────────────────────────────────────

WEIGHTS = {
    "relevance":    0.30,
    "completeness": 0.25,
    "conciseness":  0.20,
    "readability":  0.15,
    "confidence":   0.10,
}

def evaluate(prompt: str, response: str) -> dict:
    """
    Run all 5 scorers and return a full breakdown + weighted final score.
    """
    scores = {
        "relevance":    score_relevance(prompt, response),
        "completeness": score_completeness(prompt, response),
        "conciseness":  score_conciseness(prompt, response),
        "readability":  score_readability(response),
        "confidence":   score_confidence(response),
    }
    final = sum(scores[dim] * WEIGHTS[dim] for dim in scores)
    scores["overall"] = round(final, 1)
    return scores


def evaluate_comparison(results: list[dict]) -> list[dict]:
    """
    Takes the output of compare_models() and adds evaluation scores to each result.
    Returns the same list, each dict enriched with an 'evaluation' key.
    """
    for r in results:
        if not r["response"].startswith("Error"):
            r["evaluation"] = evaluate(r["prompt"], r["response"])
        else:
            r["evaluation"] = {k: 0 for k in
                                ["relevance","completeness","conciseness",
                                 "readability","confidence","overall"]}
    return results
