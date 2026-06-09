"""
llm_provider.py
───────────────
Unified LLM interface used by all agents.

PRIMARY  → Cerebras Llama 3.3 70B (FASTEST - OpenAI compatible)
FALLBACK 1 → Groq Llama 3.3 70B  (30 req/min)
FALLBACK 2 → Google Gemini 2.0 Flash (15 req/min)

Cerebras is PRIMARY because it's the fastest inference engine and uses
OpenAI-compatible API (same as Groq, so drop-in replacement).

Get your API keys:
  Cerebras: https://cloud.cerebras.ai/
  Groq     : https://console.groq.com/keys
  Gemini   : https://aistudio.google.com/app/apikey

Set in your .env file:
  CEREBRAS_API_KEY=csk_...
  GROQ_API_KEY=gsk_...      (fallback)
  GEMINI_API_KEY=AIza...    (fallback)

If no Cerebras key, falls back to Groq, then Gemini.
"""

import os
import json
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# ── Model identifiers ──────────────────────────────────────────
CEREBRAS_MODEL = "llama-3.3-70b"
GROQ_MODEL     = "llama-3.3-70b-versatile"
GEMINI_MODEL   = "gemini-2.0-flash"

# ── Read keys from environment ─────────────────────────────────
CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")
GROQ_KEY     = os.environ.get("GROQ_API_KEY",     "")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY",   "")


class LLMProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, original: Exception | None = None):
        super().__init__(message)
        self.provider = provider
        self.original = original


def call_llm(prompt: str, max_tokens: int = 500) -> str:
    if not CEREBRAS_KEY and not GROQ_KEY and not GEMINI_KEY:
        raise RuntimeError(
            "No API key found. Set CEREBRAS_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY in your .env file.\n"
            "  Cerebras (fastest): https://cloud.cerebras.ai/\n"
            "  Groq              : https://console.groq.com/keys\n"
            "  Gemini            : https://aistudio.google.com/app/apikey"
        )

    try:
        if CEREBRAS_KEY:
            print(f"\n  {'═'*70}")
            print(f"  [LLM-PROVIDER] ⚪ Using CEREBRAS (Llama 3.3 70B - FASTEST!)")
            print(f"  Model: llama-3.3-70b")
            print(f"  Speed: Fastest inference engine available")
            print(f"  Status: Primary provider - OpenAI compatible API")
            print(f"  {'═'*70}")
            response = _call_cerebras(prompt, max_tokens)
        elif GROQ_KEY:
            print(f"\n  {'═'*70}")
            print(f"  [LLM-PROVIDER] 🔵 Using GROQ (Llama 3.3 70B)")
            print(f"  Model: llama-3.3-70b-versatile")
            print(f"  Rate Limit: 30 requests/min, 14,400 requests/day")
            print(f"  Status: Fallback #1 (Cerebras key not found)")
            print(f"  {'═'*70}")
            response = _call_groq(prompt, max_tokens)
        else:
            print(f"\n  {'═'*70}")
            print(f"  [LLM-PROVIDER] 🟢 Using GEMINI (2.0 Flash)")
            print(f"  Model: gemini-2.0-flash")
            print(f"  Rate Limit: 15 requests/min, 1,500 requests/day")
            print(f"  Status: Fallback #2 (Cerebras & Groq keys not found)")
            print(f"  {'═'*70}")
            response = _call_gemini(prompt, max_tokens)

        print(f"  [LLM-PROVIDER] ✓ Request completed successfully")
        return response

    except Exception as e:
        print(f"  [LLM-PROVIDER] ✗ Primary provider failed: {e}")
        
        # Fallback chain
        if CEREBRAS_KEY and GROQ_KEY:
            print(f"  [LLM-PROVIDER] ⚠️  Cerebras failed, switching to Groq (Fallback #1)...")
            try:
                print(f"\n  {'═'*70}")
                print(f"  [LLM-PROVIDER] 🔵 Using GROQ (Fallback #1)")
                print(f"  {'═'*70}")
                response = _call_groq(prompt, max_tokens)
                print(f"  [LLM-PROVIDER] ✓ Groq fallback completed successfully")
                return response
            except Exception as e2:
                print(f"  [LLM-PROVIDER] ✗ Groq also failed: {e2}")
                if GEMINI_KEY:
                    print(f"  [LLM-PROVIDER] ⚠️  Trying Gemini (Fallback #2)...")
                    try:
                        print(f"\n  {'═'*70}")
                        print(f"  [LLM-PROVIDER] 🟢 Using GEMINI (Fallback #2)")
                        print(f"  {'═'*70}")
                        response = _call_gemini(prompt, max_tokens)
                        print(f"  [LLM-PROVIDER] ✓ Gemini fallback completed successfully")
                        return response
                    except Exception as e3:
                        print(f"  [LLM-PROVIDER] ✗ All providers failed. Cerebras: {e} | Groq: {e2} | Gemini: {e3}")
                        raise RuntimeError(f"All LLM providers failed. Cerebras: {e} | Groq: {e2} | Gemini: {e3}")
                raise RuntimeError(f"Cerebras and Groq failed. Cerebras: {e} | Groq: {e2}")
        
        elif CEREBRAS_KEY and GEMINI_KEY:
            print(f"  [LLM-PROVIDER] ⚠️  Cerebras failed, switching to Gemini (Fallback)...")
            try:
                print(f"\n  {'═'*70}")
                print(f"  [LLM-PROVIDER] 🟢 Using GEMINI (Fallback)")
                print(f"  {'═'*70}")
                response = _call_gemini(prompt, max_tokens)
                print(f"  [LLM-PROVIDER] ✓ Gemini fallback completed successfully")
                return response
            except Exception as e2:
                print(f"  [LLM-PROVIDER] ✗ Both failed. Cerebras: {e} | Gemini: {e2}")
                raise RuntimeError(f"Cerebras and Gemini failed. Cerebras: {e} | Gemini: {e2}")
        
        elif GROQ_KEY and GEMINI_KEY:
            print(f"  [LLM-PROVIDER] ⚠️  Groq failed, switching to Gemini (Fallback)...")
            try:
                print(f"\n  {'═'*70}")
                print(f"  [LLM-PROVIDER] 🟢 Using GEMINI (Fallback)")
                print(f"  {'═'*70}")
                response = _call_gemini(prompt, max_tokens)
                print(f"  [LLM-PROVIDER] ✓ Gemini fallback completed successfully")
                return response
            except Exception as e2:
                print(f"  [LLM-PROVIDER] ✗ Both failed. Groq: {e} | Gemini: {e2}")
                raise RuntimeError(f"Both providers failed. Groq: {e} | Gemini: {e2}")
        
        raise


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

def _call_cerebras(prompt: str, max_tokens: int) -> str:
    """
    Cerebras Llama 3.3 70B via OpenAI-compatible REST API.
    Fastest inference engine - optimized for speed.
    Docs: https://cloud.cerebras.ai/
    """
    if not CEREBRAS_KEY:
        raise RuntimeError("CEREBRAS_API_KEY not set.")

    url = "https://api.cerebras.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CEREBRAS_KEY}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":       CEREBRAS_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Cerebras request failed: {e}") from e

    if resp.status_code == 429:
        print("[LLM] Cerebras rate limited, waiting 10s...")
        time.sleep(10)
        resp = requests.post(url, json=body, headers=headers, timeout=30)

    try:
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Cerebras API error: {resp.text}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Unexpected Cerebras response format: {resp.text}") from e


def _call_groq(prompt: str, max_tokens: int) -> str:
    """
    Groq with Llama 3.3 70B via OpenAI-compatible REST API.
    Free tier: 30 RPM, 14,400 req/day — no credit card.
    Docs: https://console.groq.com/docs/openai
    """
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")

    url = "https://api.groq.com/openai/v1/chat/completions"
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

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Groq request failed: {e}") from e

    if resp.status_code == 429:
        print("[LLM] Groq rate limited, waiting 10s...")
        time.sleep(10)
        resp = requests.post(url, json=body, headers=headers, timeout=30)

    try:
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        raise RuntimeError(f"Groq API error: {resp.text}") from e
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Unexpected Groq response format: {resp.text}") from e


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

    try:
        resp = requests.post(url, json=body, timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e

    if resp.status_code == 429:
        print("[LLM] Gemini rate limited, waiting 15s...")
        time.sleep(15)
        resp = requests.post(url, json=body, timeout=30)

    try:
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Gemini API error: {resp.text}") from e
    except ValueError as e:
        raise RuntimeError(f"Gemini returned invalid JSON: {resp.text}") from e

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected Gemini response structure: {data}") from e