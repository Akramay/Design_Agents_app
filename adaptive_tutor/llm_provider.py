"""
llm_provider.py
───────────────
Unified free LLM interface used by all agents.

PRIMARY  → Groq Llama 3.3 70B       (30 req/min, 14,400 req/day — no credit card)
FALLBACK → Google Gemini 2.0 Flash  (15 req/min, 1,500 req/day — no credit card)

Groq is now PRIMARY because it has a much more generous rate limit (30 RPM vs 15 RPM)
and does not get exhausted by the multi-attempt question generation loop.

Get your free keys (no credit card for either):
  Groq   : https://console.groq.com/keys
  Gemini : https://aistudio.google.com/app/apikey

Set in your .env file:
  GROQ_API_KEY=gsk_...
  GEMINI_API_KEY=AIza...   (optional, used as fallback if Groq fails)

If neither key is set, the module raises a clear error on first call.
"""

import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# ── Model identifiers ──────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash"

# ── Read keys from environment ─────────────────────────────────
GROQ_KEY   = os.environ.get("GROQ_API_KEY",   "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")


def call_llm(prompt: str, max_tokens: int = 500) -> str:
    print(f"\n[DEBUG] --- LLM PROMPT SENT ---")
    print(prompt[:200] + "...")

    if not GROQ_KEY and not GEMINI_KEY:
        raise RuntimeError(
            "No API key found. Set GROQ_API_KEY or GEMINI_API_KEY in your .env file.\n"
            "  Groq (recommended) : https://console.groq.com/keys\n"
            "  Gemini             : https://aistudio.google.com/app/apikey"
        )

    try:
        # Groq is PRIMARY — 30 RPM, far more generous than Gemini's 15 RPM
        if GROQ_KEY:
            response = _call_groq(prompt, max_tokens)
        else:
            response = _call_gemini(prompt, max_tokens)

        print(f"[DEBUG] --- RAW LLM RESPONSE RECEIVED ---")
        print(response)
        print(f"[DEBUG] ----------------------------------\n")
        return response

    except Exception as e:
        # If primary provider fails, try the other one as emergency fallback
        print(f"  [LLM] Primary provider failed: {e}")
        if GROQ_KEY and GEMINI_KEY:
            print(f"  [LLM] Trying Gemini as emergency fallback...")
            try:
                response = _call_gemini(prompt, max_tokens)
                print(f"[DEBUG] --- RAW LLM RESPONSE (Gemini fallback) ---")
                print(response)
                print(f"[DEBUG] ------------------------------------------\n")
                return response
            except Exception as e2:
                print(f"!!! CRITICAL: Both providers failed. Groq: {e} | Gemini: {e2}")
                raise RuntimeError(f"Both LLM providers failed. Groq: {e} | Gemini: {e2}")
        print(f"!!! CRITICAL API ERROR: {e}")
        raise e


def parse_json(raw: str) -> dict | list:
    try:
        raw_clean = re.sub(r"```json\s*", "", raw)
        raw_clean = re.sub(r"```\s*", "", raw_clean)
        brace   = raw_clean.find("{")
        bracket = raw_clean.find("[")
        start   = min([p for p in [brace, bracket] if p != -1] or [0])
        return json.loads(raw_clean[start:])
    except Exception as e:
        print(f"!!! JSON PARSE ERROR: Could not turn response into a dictionary. Error: {e}")
        print(f"!!! RAW CONTENT THAT FAILED: {raw}")
        raise e


# ── Private helpers ────────────────────────────────────────────

def _call_groq(prompt: str, max_tokens: int) -> str:
    """
    Groq with Llama 3.3 70B via OpenAI-compatible REST API.
    Free tier: 30 RPM, 14,400 req/day — no credit card.
    Docs: https://console.groq.com/docs/openai
    """
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    url     = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": 0.7,
    }

    resp = requests.post(url, json=body, headers=headers, timeout=30)

    if resp.status_code == 429:
        print("  [LLM] Groq rate limited, waiting 10s...")
        time.sleep(10)
        resp = requests.post(url, json=body, headers=headers, timeout=30)

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(prompt: str, max_tokens: int) -> str:
    """
    Google Gemini 2.0 Flash via REST API.
    Free tier: 15 RPM, 1,500 req/day — no credit card.
    Docs: https://ai.google.dev/gemini-api/docs

    NOTE: Gemini is now the FALLBACK. Its 15 RPM limit is too low for the
    multi-attempt question generation loop (6 attempts x 2 calls = 12 calls/concept).
    """
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.7,
        },
    }

    resp = requests.post(url, json=body, timeout=30)

    if resp.status_code == 429:
        print("  [LLM] Gemini rate limited, waiting 15s...")
        time.sleep(15)
        resp = requests.post(url, json=body, timeout=30)

    resp.raise_for_status()
    data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response structure: {data}") from e