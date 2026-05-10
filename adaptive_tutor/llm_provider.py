"""
llm_provider.py
───────────────
Unified free LLM interface used by all agents.

PRIMARY  → Google Gemini 2.0 Flash  (1,500 req/day, no credit card)
FALLBACK → Groq Llama 3.3 70B       (30 req/min,  no credit card)

Get your free keys (no credit card for either):
  Gemini : https://aistudio.google.com/app/apikey
  Groq   : https://console.groq.com/keys

Set environment variables:
  export GEMINI_API_KEY="AIza..."
  export GROQ_API_KEY="gsk_..."

If only one key is set that provider is used exclusively.
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
GEMINI_MODEL  = "gemini-2.0-flash"
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ── Read keys from environment ─────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_KEY   = os.environ.get("GROQ_API_KEY",   "")

def call_llm(prompt: str, max_tokens: int = 500) -> str:
    print(f"\n[DEBUG] --- LLM PROMPT SENT ---")
    print(prompt[:200] + "...") # See the start of what you're asking
    
    try:
        if GEMINI_KEY:
            response = _call_gemini(prompt, max_tokens)
        else:
            response = _call_groq(prompt, max_tokens)
            
        print(f"[DEBUG] --- RAW LLM RESPONSE RECEIVED ---")
        print(response) # THIS IS WHAT YOU NEED TO SEE
        print(f"[DEBUG] ----------------------------------\n")
        return response
    except Exception as e:
        print(f"!!! CRITICAL API ERROR: {e}")
        raise e

def parse_json(raw: str) -> dict | list:
    try:
        # Existing logic to strip markdown
        raw_clean = re.sub(r"```json\s*", "", raw)
        raw_clean = re.sub(r"```\s*", "", raw_clean)
        brace = raw_clean.find("{")
        bracket = raw_clean.find("[")
        start = min([p for p in [brace, bracket] if p != -1] or [0])
        return json.loads(raw_clean[start:])
    except Exception as e:
        print(f"!!! JSON PARSE ERROR: Could not turn response into a dictionary. Error: {e}")
        print(f"!!! RAW CONTENT THAT FAILED: {raw}")
        raise e

# ── Private helpers ────────────────────────────────────────────

def _call_gemini(prompt: str, max_tokens: int) -> str:
    """
    Google Gemini 2.0 Flash via REST API.
    Free tier: 15 RPM, 1,500 req/day, 1M token context — no credit card.
    Docs: https://ai.google.dev/gemini-api/docs
    """
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
        # Rate limited — wait 5 seconds and retry once
        print("  [LLM] Gemini rate limited, waiting 5s...")
        time.sleep(5)
        resp = requests.post(url, json=body, timeout=30)

    resp.raise_for_status()
    data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response structure: {data}") from e


def _call_groq(prompt: str, max_tokens: int) -> str:
    """
    Groq with Llama 3.3 70B via OpenAI-compatible REST API.
    Free tier: 30 RPM, 14,400 req/day — no credit card.
    Docs: https://console.groq.com/docs/openai
    """
    url  = "https://api.groq.com/openai/v1/chat/completions"
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
        print("  [LLM] Groq rate limited, waiting 5s...")
        time.sleep(5)
        resp = requests.post(url, json=body, headers=headers, timeout=30)

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()