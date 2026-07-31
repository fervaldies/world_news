"""
fetch_news.py — world_news
--------------------------
Fetches world news by randomly picking 5 topics from a curated list of
continent/region-flavoured topics, searching GNews for each (5 articles each),
pooling ~25 candidates, de-duplicating, then using GitHub Models to pick the
best 5 (naming specific people/places/events, with topic variety).
Then rewrites them punchy and translates to Spanish — all free.

Required environment variables:
  GNEWS_API_KEY   — free API key from gnews.io
  GITHUB_TOKEN    — automatically available in GitHub Actions
"""

import sys
import json
import re
import os
import time
import random
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

GNEWS_API_KEY      = os.environ.get("GNEWS_API_KEY", "")
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-3.5-flash-lite"
GEMINI_URL      = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Curated topic pool — continent/region flavoured, specific enough to force
# concrete, named stories rather than vague wire copy. Each run picks 5 at random.
TOPICS = [
    "Europe politics",
    "European Union",
    "United Kingdom news",
    "France news",
    "Germany news",
    "Oceania news",
    "Australia news",
    "United States politics",
    "United States economy",
    "Canada news",
    "Middle East conflict",
    "Israel news",
    "Asia technology",
    "China news",
    "Japan news",
    "India news",
    "Africa news",
    "South Africa news",
    "Nigeria news",
    "South America news",
    "Brazil news",
    "Mexico news",
    "Russia Ukraine war",
    "world economy",
    "global climate",
    "space exploration",
    "artificial intelligence",
    "world health",
]

TOPICS_PER_RUN    = 5
ARTICLES_PER_TOPIC = 5


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
    """Remove trailing source attribution like ' - Reuters' if it looks like a source."""
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        suffix = parts[1].strip()
        sentence_words = {"the", "a", "an", "and", "or", "but", "in",
                          "on", "at", "to", "of", "for", "is", "are",
                          "was", "were", "not", "new", "old", "it"}
        words = suffix.lower().split()
        looks_like_source = (
            len(suffix) < 30 and
            not any(w in sentence_words for w in words)
        )
        if looks_like_source:
            return parts[0].strip()
    return title.strip()


def remove_near_duplicates(headlines):
    """Drop headlines that share too many significant words with an earlier one."""
    stop = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
            "for", "as", "is", "are", "was", "were", "with", "from", "by", "its",
            "amid", "over", "after", "new", "say", "says"}
    kept, kept_wordsets = [], []
    for h in headlines:
        words = {w.lower().strip(".,'\"") for w in h.split()
                 if w.lower() not in stop and len(w) > 3}
        is_dup = False
        for ws in kept_wordsets:
            overlap = len(words & ws)
            if overlap >= 3 and overlap >= 0.5 * min(len(words), len(ws)):
                is_dup = True
                break
        if not is_dup:
            kept.append(h)
            kept_wordsets.append(words)
    return kept


def github_models_call(messages, max_tokens=600):
    """Make a call to Gemini API and return the response text.
    Keeps the same function name/interface so callers don't need to change."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    # Gemini has no separate "system" role — fold everything into one user turn
    prompt_text = "\n\n".join(m["content"] for m in messages)

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }).encode("utf-8")

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"  ⏳ Gemini rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ── news fetching ─────────────────────────────────────────────────────────────

def fetch_topic(topic):
    """Search GNews for one topic, return cleaned headlines."""
    params = urllib.parse.urlencode({
        "q":      topic,
        "lang":   "en",
        "max":    str(ARTICLES_PER_TOPIC),
        "apikey": GNEWS_API_KEY,
    })
    url = f"https://gnews.io/api/v4/search?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  ⚠️  '{topic}' failed: {e.code}")
        return []
    articles = data.get("articles", [])
    print(f"  '{topic}': {len(articles)} articles")
    out = []
    for article in articles:
        title = clean_title(article.get("title", ""))
        if title and len(title) > 10:
            out.append(title)
    return out


def fetch_world_news():
    """Pick 5 random topics, fetch each, pool + de-duplicate the results."""
    if not GNEWS_API_KEY:
        raise ValueError("GNEWS_API_KEY is not set")

    chosen = random.sample(TOPICS, TOPICS_PER_RUN)
    print(f"🎲 Topics this run: {', '.join(chosen)}")

    pool = []
    seen = set()
    for topic in chosen:
        for title in fetch_topic(topic):
            if title not in seen:
                seen.add(title)
                pool.append(title)
        time.sleep(1)  # be gentle on the free GNews rate limit

    print(f"📰 Pool of {len(pool)} unique headlines before de-dup")
    pool = remove_near_duplicates(pool)
    print(f"🧹 {len(pool)} headlines after near-duplicate removal")

    if len(pool) < 5:
        raise ValueError(f"Only {len(pool)} headlines after de-dup — need at least 5")

    return pool


def pick_best_5(headlines):
    """Use GitHub Models to pick the 5 best, varied, globally relevant stories."""
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    print("🤖 Picking best 5 via GitHub Models...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "You are a world news editor. Pick the 5 best stories from the list.\n\n"
            "STEP 1 — Remove duplicates FIRST. Group headlines that describe the same "
            "underlying event (same country + same topic = same story, even if the "
            "wording, angle, or numbers differ). From each group keep only the single "
            "clearest headline.\n\n"
            "STEP 2 — From the de-duplicated stories, pick the 5 most important and "
            "globally interesting. Prefer variety of regions and topics, but you MAY "
            "pick 2 from the same region if both are genuinely strong and clearly "
            "different events.\n\n"
            "AVOID vague headlines with no specific person, country, company, or place "
            "(e.g. 'Man robs a bank' with no location), and minor sport or celebrity items.\n\n"
            "PREFER headlines that NAME a specific country, leader, company, or concrete "
            "event, and that would make sense to an international viewer.\n\n"
            "Return ONLY raw JSON, no markdown, no backticks:\n"
            '{"selected_indexes": [0, 1, 2, 3, 4]}\n\n'
            f"Indexes are 0-based. Stories:\n\n{numbered}"
        )
    }], max_tokens=100)
    text    = extract_json(text)
    indexes = json.loads(text)["selected_indexes"]
    selected = [headlines[i] for i in indexes if i < len(headlines)]
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


def rewrite_headlines(headlines):
    """Rewrite raw API headlines to sound natural and punchy for social media."""
    numbered = "\n".join(f"{i+1}. {n['title']}" for i, n in enumerate(headlines))
    print("✍️ Rewriting headlines for natural language...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "Rewrite these 5 news headlines to sound natural and punchy for social media. "
            "Rules: remove prefixes like 'LIVE UPDATES:', 'Study:', 'Report:', 'Breaking:'. "
            "Remove dates in parentheses. Replace semicolons with a comma or 'and'. "
            "Simplify scientific jargon into plain language. Remove marketing-speak. "
            "Keep each headline under 20 words and factually accurate. "
            "Return ONLY raw JSON, no markdown, no backticks:\n"
            '{"news": [{"title": "rewritten"}, {"title": "rewritten"}, '
            '{"title": "rewritten"}, {"title": "rewritten"}, {"title": "rewritten"}]}\n\n'
            f"Headlines:\n{numbered}"
        )
    }])
    text = extract_json(text)
    rewritten = json.loads(text)["news"]
    return rewritten


# ── main ──────────────────────────────────────────────────────────────────────

def get_news(day_name):
    # Step 1 — fetch world news across 5 random topics
    all_headlines = fetch_world_news()

    # Step 2 — pick 5 diverse, globally relevant stories
    en_news = pick_best_5(all_headlines)
    en_news = rewrite_headlines(en_news)
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
