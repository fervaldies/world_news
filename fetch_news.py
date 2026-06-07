"""
fetch_news.py — world_news
--------------------------
Fetches top world news headlines using GNews API (free),
uses GitHub Models to pick the 5 most diverse/important stories,
then translates them to Spanish using GitHub Models (free).

Required environment variables:
  GNEWS_API_KEY   — free API key from gnews.io
  GITHUB_TOKEN    — automatically available in GitHub Actions
"""

import sys
import json
import re
import os
import urllib.request
import urllib.error
from datetime import datetime

GNEWS_API_KEY      = os.environ.get("GNEWS_API_KEY", "")
GITHUB_TOKEN       = os.environ.get("GITHUB_TOKEN", "")
GITHUB_MODELS_URL  = "https://models.inference.ai.azure.com/chat/completions"
GITHUB_MODEL       = "gpt-4o-mini"


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_json(text):
    """Extract the first complete JSON object from text, ignoring any preamble."""
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {repr(text)}")
    return text[start:end + 1]


def clean_title(title):
    """Remove source attribution like ' - Reuters' from the end of a headline."""
    if " - " in title:
        title = title.rsplit(" - ", 1)[0]
    return title.strip()


def github_models_call(messages, max_tokens=600):
    """Make a call to GitHub Models API and return the response text."""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set")

    payload = json.dumps({
        "model": GITHUB_MODEL,
        "messages": messages,
        "max_tokens": max_tokens
    }).encode("utf-8")

    req = urllib.request.Request(
        GITHUB_MODELS_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GITHUB_TOKEN}"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"]


# ── news fetching ─────────────────────────────────────────────────────────────

def fetch_world_news():
    """Fetch top world headlines from GNews API (10 articles for diversity selection)."""
    if not GNEWS_API_KEY:
        raise ValueError("GNEWS_API_KEY is not set")

    url = (
        f"https://gnews.io/api/v4/top-headlines"
        f"?lang=en&max=10&apikey={GNEWS_API_KEY}"
    )
    print("📰 Fetching from GNews API...")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    articles = data.get("articles", [])
    print(f"  GNews returned {len(articles)} articles")

    if len(articles) < 5:
        raise ValueError(f"GNews returned only {len(articles)} articles — need at least 5")

    headlines = []
    for article in articles:
        title = clean_title(article.get("title", ""))
        if title:
            headlines.append(title)

    return headlines


def pick_best_5(headlines):
    """Use GitHub Models to pick the 5 most diverse and globally relevant stories."""
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))

    print("🤖 Picking best 5 via GitHub Models...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "You are a news editor. From the list below, pick the 5 most important "
            "and globally relevant stories. Prioritise variety — avoid picking 2 stories "
            "about the same country or topic. "
            "Return ONLY raw JSON, no markdown, no backticks:\n"
            '{"selected_indexes": [0, 1, 2, 3, 4]}\n\n'
            "Indexes are 0-based. Stories:\n\n"
            f"{numbered}"
        )
    }], max_tokens=100)

    text    = extract_json(text)
    indexes = json.loads(text)["selected_indexes"]
    selected = [headlines[i] for i in indexes if i < len(headlines)]

    # Pad if needed
    if len(selected) < 5:
        for h in headlines:
            if h not in selected:
                selected.append(h)
            if len(selected) == 5:
                break

    return [{"title": t} for t in selected[:5]]


# ── translation ───────────────────────────────────────────────────────────────

def translate_to_spanish(headlines):
    """Translate headlines to Spanish (Spain) using GitHub Models."""
    headlines_text = "\n".join(f"- {n['title']}" for n in headlines)

    print("🌐 Translating via GitHub Models...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "Translate these headlines to Spanish from Spain. "
            "Return ONLY raw JSON, no markdown, no backticks, no explanation:\n"
            '{"news": [{"title": "translated"}, {"title": "translated"}, '
            '{"title": "translated"}, {"title": "translated"}, {"title": "translated"}]}\n\n'
            f"Headlines:\n{headlines_text}"
        )
    }])

    return extract_json(text)


# ── main ──────────────────────────────────────────────────────────────────────

def get_news(day_name):
    # Step 1 — fetch world news headlines
    all_headlines = fetch_world_news()

    # Step 2 — pick 5 diverse, globally relevant stories
    en_news = pick_best_5(all_headlines)
    en_data = {"news": en_news}
    print(f"✅ Selected {len(en_data['news'])} headlines:")
    for n in en_data["news"]:
        print(f"  - {n['title']}")

    # Step 3 — translate to Spanish
    es_text = translate_to_spanish(en_data["news"])
    es_data = json.loads(es_text)
    print("✅ Translation complete")

    # Step 4 — write YML files
    date_str = datetime.now().strftime("%Y-%m-%d")

    def build_yml(data):
        lines = [f"date: {date_str}", f"day: {day_name}", "news:"]
        for item in data["news"]:
            title = item["title"].replace('"', "'")
            lines.append(f'  - title: "{title}"')
        return "\n".join(lines) + "\n"

    with open(f"{day_name}NewsEN.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(en_data))
    with open(f"{day_name}NewsES.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(es_data))

    print(f"✅ Created {day_name}NewsEN.yml and {day_name}NewsES.yml")


if __name__ == "__main__":
    day_name = sys.argv[1]
    get_news(day_name)
