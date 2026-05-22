"""
ai_failover.py
--------------
Multi-provider news/search helper with automatic failover.

Order: Claude -> Gemini -> OpenAI.
Each provider uses its OWN native web search tool, and each gets a few
retries with exponential backoff (for transient 429/5xx/529 errors)
before we give up and move to the next provider.

Usage:
    from ai_failover import generate_with_search
    text = generate_with_search("Give me today's top 5 Australian news stories...")

Required environment variables (see notes at bottom):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GEMINI_API_KEY     (a.k.a. Google AI Studio key)

Install deps:
    pip install anthropic openai google-genai
"""

import os
import time

# --------------------------------------------------------------------------
# CONFIG: model names change often. Update these in ONE place when needed.
# Confirm current model strings at each provider's docs before deploying.
# --------------------------------------------------------------------------
CLAUDE_MODEL = "claude-opus-4-7"
OPENAI_MODEL = "gpt-5.5"
GEMINI_MODEL = "gemini-2.5-flash"

MAX_TOKENS = 2000          # for providers that need an explicit cap
MAX_RETRIES = 4            # retries PER provider before failover
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


# --------------------------------------------------------------------------
# Generic retry wrapper
# --------------------------------------------------------------------------
def _status_of(err):
    """Pull an HTTP status code out of whatever exception a given SDK throws."""
    # anthropic + openai SDKs expose .status_code; google-genai uses .code
    return getattr(err, "status_code", None) or getattr(err, "code", None)


def _with_retries(provider_name, fn):
    """Call fn(), retrying transient errors with exponential backoff.

    Raises the last exception once retries are exhausted OR immediately
    if the error is clearly non-transient (bad key, bad request, etc.),
    so the caller can fail over to the next provider.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001 (intentional)
            status = _status_of(e)
            transient = status in RETRYABLE_STATUS or status is None and "timeout" in str(e).lower()
            if not transient or attempt == MAX_RETRIES - 1:
                # non-transient, or out of retries -> let caller fail over
                raise
            wait = 2 ** attempt + attempt * 0.5     # 1s, 2.5s, 5s, 9.5s ...
            print(f"[{provider_name}] transient error {status}, "
                  f"retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)


# --------------------------------------------------------------------------
# Provider 1: Claude (Anthropic) with native web search
# --------------------------------------------------------------------------
def call_claude(prompt):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def _do():
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
        )
        # Concatenate the text blocks; ignore tool-use/citation blocks.
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    return _with_retries("Claude", _do)


# --------------------------------------------------------------------------
# Provider 2: OpenAI with native web search (Responses API)
# --------------------------------------------------------------------------
def call_openai(prompt):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    def _do():
        resp = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],   # "web_search_preview" on older models
            input=prompt,
        )
        return resp.output_text

    return _with_retries("OpenAI", _do)


# --------------------------------------------------------------------------
# Provider 3: Gemini with grounding (Google Search)
# --------------------------------------------------------------------------
def call_gemini(prompt):
    from google import genai
    from google.genai import types
    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env

    def _do():
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return resp.text

    return _with_retries("Gemini", _do)


# --------------------------------------------------------------------------
# The failover loop
# --------------------------------------------------------------------------
PROVIDERS = [call_claude, call_gemini, call_openai]


def generate_with_search(prompt):
    """Try each provider in order; return the first success."""
    last_error = None
    for provider in PROVIDERS:
        name = provider.__name__
        try:
            print(f"--> trying {name}")
            result = provider(prompt)
            if result and result.strip():
                print(f"--> {name} succeeded")
                return result
            print(f"[{name}] returned empty output, failing over")
        except Exception as e:                      # noqa: BLE001
            last_error = e
            print(f"[{name}] failed ({_status_of(e)}): {e} -- failing over")
    raise RuntimeError(f"All providers failed. Last error: {last_error}")


if __name__ == "__main__":
    # Quick smoke test
    out = generate_with_search(
        "In 3 bullet points, what are today's top news stories in Australia?"
    )
    print("\n=== RESULT ===\n", out)
