"""Stand-in for Newsdesk's extras.py: the two helpers series.py and inbox.py use, copied unchanged."""
import json
import urllib.parse

from . import services as S

DEFAULT_GEMINI = "gemini-3.5-flash-lite"


def extract_json(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON in the answer")
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError("Unterminated JSON")


def ask_ai(vault, cfg, prompt, max_tokens=1500, free_only=False):
    """Gemini first (free), Claude as backup — same as fetch_news.py. free_only: never use Claude."""
    key = vault.get("GEMINI_API_KEY")
    if key:
        model = cfg.get("ai_model_gemini", DEFAULT_GEMINI)
        st, _, b = S.http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                                  f"?key={urllib.parse.quote(key)}", {"Content-Type": "application/json"},
                          {"contents": [{"parts": [{"text": prompt}]}],
                           "generationConfig": {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}},
                          timeout=90)
        d = S.jbody(b) or {}
        if st == 200:
            try:
                return d["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                pass
        if free_only:
            raise Exception(f"Gemini didn't answer (HTTP {st}: {(d.get('error') or {}).get('message', '')[:160]}). "
                            "Try again in a minute, or untick 'Free only' to let Claude do it.")
    elif free_only:
        raise Exception("GEMINI_API_KEY is missing in secrets.env")
    return S.claude(vault, "Reply with JSON only.", prompt, cfg.get("ai_model", S.DEFAULT_AI_MODEL), max_tokens)
