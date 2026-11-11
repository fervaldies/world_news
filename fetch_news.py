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
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL       = "gemini-3.5-flash-lite"
GEMINI_URL         = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL    = "claude-haiku-4-5-20251001"

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


def gemini_call(messages, max_tokens=600):
    """Make a call to Gemini API and return the response text."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

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
            error_body = e.read().decode("utf-8")
            print(f"  ❌ Gemini API error {e.code}: {error_body}")
            transient = e.code == 429 or 500 <= e.code < 600
            if transient and attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"  ⏳ Gemini transient error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def claude_call(messages, max_tokens=600):
    """Fallback: call Claude Haiku via the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    prompt_text = "\n\n".join(m["content"] for m in messages)

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt_text}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read().decode("utf-8"))
    return result["content"][0]["text"]


def github_models_call(messages, max_tokens=600):
    """Try Gemini first (free); fall back to Claude Haiku if Gemini is down.
    Keeps the same function name/interface so callers don't need to change."""
    try:
        return gemini_call(messages, max_tokens)
    except Exception as e:
        print(f"  ⚠️ Gemini unavailable after retries ({e}) — falling back to Claude Haiku...")
        return claude_call(messages, max_tokens)


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
        remember_article(title, article)
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
    remember_picked(en_news)
    en_news = rewrite_headlines(en_news)
    en_news  = friday_good_news(en_news)
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
    write_extra_languages(en_data["news"], day_name, date_str)
    enrich_day_files(day_name, en_data["news"])



# ── Extra languages for the PT / ID networks (non-fatal) ──────────────────────
EXTRA_LANGS = {
    "PT": ("Brazilian Portuguese (pt-BR), as written by Brazilian news media. Use Brazilian number "
           "formats (6,3 bilhões; 2,8%; US$ 5 milhões)."),
    "ID": ("Indonesian (Bahasa Indonesia), natural news style as written by Kompas or Tempo. Use "
           "Indonesian number formats (6,3 miliar; 2,8 persen)."),
}


def write_extra_languages(en_news, day_name, date_str):
    """Translate today's 5 English headlines into PT and ID and write {Day}NewsPT/ID.yml."""
    headlines_text = "\n".join(f"- {n['title']}" for n in en_news)
    for code, style in EXTRA_LANGS.items():
        try:
            print(f"🌐 Translating to {code}...")
            text = github_models_call([{
                "role": "user",
                "content": (
                    f"Translate these 5 news headlines into {style} "
                    "Keep every number, name, place and organisation exactly as in the original; "
                    "natural, concise headline style; same order; no extra words. "
                    "Return ONLY raw JSON, no markdown, no backticks, no explanation:\n"
                    '{"news": [{"title": "..."}, {"title": "..."}, {"title": "..."}, '
                    '{"title": "..."}, {"title": "..."}]}\n\n'
                    f"Headlines:\n{headlines_text}"
                )
            }])
            data = json.loads(extract_json(text))
            news = [n["title"].strip() for n in data.get("news", []) if n.get("title")]
            if len(news) != len(en_news):
                raise ValueError(f"got {len(news)} headlines instead of {len(en_news)}")
            lines = [f"date: {date_str}", f"day: {day_name}", "news:"]
            lines += [f'  - title: "{t.replace(chr(34), chr(39))}"' for t in news]
            with open(f"{day_name}News{code}.yml", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"✅ Created {day_name}News{code}.yml")
        except Exception as e:
            print(f"⚠️ {code} translation skipped (non-fatal): {e}")


# ═══ Story details for the new daily style (non-fatal) ═══════════════════════
# Keeps each story's source and link, writes "why it matters" for two stories, picks a
# visual (map / number / icon) and a category, and on Fridays swaps story 5 for good news.
import datetime as _dt
ARTICLE_INFO = {}            # headline → {"source": ..., "url": ...}
PICKED = []                  # details of the 5 picked stories, in order
GOOD_INDEX = None
CATEGORIES = ["POLITICS", "ECONOMY", "BUSINESS", "TECH", "CLIMATE", "ENERGY", "HEALTH", "JUSTICE", "TRANSPORT",
              "WEATHER", "SPORT", "SCIENCE", "SOCIETY", "CONFLICT", "WORLD"]


def remember_article(title, article):
    src = (article.get("source") or {}).get("name") or ""
    ARTICLE_INFO.setdefault(title, {"source": src.strip(), "url": article.get("url") or ""})


def remember_picked(news):
    global PICKED
    PICKED = [dict(ARTICLE_INFO.get(n["title"], {})) for n in news]


def friday_good_news(en_news):
    """On Fridays: replace story 5 with a genuinely positive story (if one is found)."""
    global GOOD_INDEX
    if _dt.datetime.utcnow().weekday() != 4 or len(en_news) < 5 or not GNEWS_API_KEY:
        return en_news
    try:
        q = '(breakthrough OR rescued OR recovery OR "good news" OR conservation)'
        params = urllib.parse.urlencode({"q": q, "lang": "en", "max": "10", "apikey": GNEWS_API_KEY,
                                         "from": (_dt.datetime.utcnow() - _dt.timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")})
        with urllib.request.urlopen(f"https://gnews.io/api/v4/search?{params}", timeout=15) as r:
            arts = json.loads(r.read().decode("utf-8")).get("articles", [])
        cands = [(clean_title(a.get("title", "")), a) for a in arts if a.get("title")]
        cands = [(t, a) for t, a in cands if t and t not in [n["title"] for n in en_news]]
        if not cands:
            return en_news
        numbered = "\n".join(f"{i}. {t}" for i, (t, a) in enumerate(cands))
        text = github_models_call([{"role": "user", "content": (
            "Pick the ONE headline that is genuinely good news for ordinary people (a breakthrough, rescue, "
            "record, recovery, community win, conservation success). Not sport results, not company profits, "
            "not anything involving deaths or disasters. If none qualifies, answer -1. Then rewrite it as a "
            "natural, punchy, factual headline under 18 words. Return ONLY raw JSON: "
            '{"index": 0, "title": "..."}\n\n' + numbered)}], max_tokens=200)
        j = json.loads(extract_json(text))
        i = int(j.get("index", -1))
        if i < 0 or i >= len(cands) or not str(j.get("title", "")).strip():
            print("ℹ️ No good-news story today — keeping the normal five")
            return en_news
        t, a = cands[i]
        en_news = en_news[:4] + [{"title": str(j["title"]).strip()}]
        while len(PICKED) < 5:
            PICKED.append({})
        PICKED[4] = {"source": ((a.get("source") or {}).get("name") or "").strip(), "url": a.get("url") or ""}
        GOOD_INDEX = 4
        print(f"🌱 Good news for Friday: {en_news[4]['title']}")
    except Exception as e:
        print(f"⚠️ Good-news search skipped (non-fatal): {e}")
    return en_news


def ai_json(prompt, max_tokens, check):
    """Gemini (twice), then Claude. Each answer must be valid JSON that passes `check`, else the next try."""
    errors = []
    for who, fn in (("Gemini", gemini_call), ("Gemini", gemini_call), ("Claude", claude_call)):
        try:
            data = json.loads(extract_json(fn([{"role": "user", "content": prompt}], max_tokens) or ""))
            check(data)
            return data
        except Exception as e:
            errors.append(f"{who}: {str(e)[:160]}")
            print(f"  ⚠️ {who} answer not usable ({str(e)[:160]})")
    raise ValueError(" | ".join(errors))


def enrich_day_files(day_name, en_news):
    """Adds source / url / good (always) and cat / visual / why (AI, best effort) to every language
    file of today. Never fatal; never puts English text into the ES / PT / ID files."""
    import yaml
    titles = [n["title"] for n in en_news]
    stories, tr = None, {}
    try:
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))

        def check_stories(d):
            st = d.get("stories")
            if not isinstance(st, list) or len(st) != len(titles) or not all(isinstance(s, dict) for s in st):
                raise ValueError(f"expected {len(titles)} stories, got {len(st) if isinstance(st, list) else st!r}")

        stories = ai_json(
            "For each of these 5 news headlines, return details for a short news video.\n"
            f"- cat: one of {', '.join(CATEGORIES)}.\n"
            "- why: ONLY for the 2 stories that most need context, one factual sentence (max 12 words) on why it "
            "matters to ordinary people; no opinion, nothing not implied by the headline. Empty string for the others.\n"
            "- visual: {\"type\": \"number\", \"number\": \"945\", \"label\": \"days in detention\"} when the headline has "
            "a striking number (label max 3 words); else {\"type\": \"map\", \"country\": \"<country name as in "
            "Natural Earth, e.g. Australia, Japan, United States of America>\", \"place\": \"<short place label>\", "
            "\"lat\": <number>, \"lon\": <number>, \"span\": <degrees to show around it: 3 for a city, 12 for a "
            "state, 25 for a country>} when it is about a specific place (add country2/place2/lat2/lon2 when two "
            "places are linked); else {\"type\": \"icon\"}.\n"
            "Return ONLY raw JSON, no markdown: {\"stories\": [{\"cat\": \"...\", \"why\": \"...\", \"visual\": {...}}, ...5]}\n\n"
            + numbered, 3000, check_stories)["stories"]
    except Exception as e:
        print(f"⚠️ Categories / pictures / why-lines skipped (non-fatal): {e}")
    bits = None
    if stories:
        bits = {"why": [str(s.get("why") or "") for s in stories],
                "label": [str((s.get("visual") or {}).get("label") or "") for s in stories],
                "place": [str((s.get("visual") or {}).get("place") or "") for s in stories],
                "place2": [str((s.get("visual") or {}).get("place2") or "") for s in stories]}
        if any(x for v in bits.values() for x in v):
            def check_tr(d):
                for lg in ("es", "pt", "id"):
                    if not isinstance(d.get(lg), dict):
                        raise ValueError(f"no '{lg}' block")
            try:
                tr = ai_json(
                    "Translate every string in these lists into Spanish from Spain (es), Brazilian Portuguese (pt) "
                    "and Indonesian (id). Keep empty strings empty, keep numbers and names exactly, keep it short, "
                    "same list lengths. Return ONLY raw JSON, no markdown: "
                    "{\"es\": {same keys}, \"pt\": {...}, \"id\": {...}}\n\n"
                    + json.dumps(bits, ensure_ascii=False), 3000, check_tr)
            except Exception as e:
                print(f"⚠️ Translation of the short texts skipped (non-fatal) — ES/PT/ID get pictures without them: {e}")
    done = []
    for code in ("EN", "ES", "PT", "ID"):
        path = f"{day_name}News{code}.yml"
        if not os.path.exists(path):
            continue
        try:
            data = yaml.safe_load(open(path, encoding="utf-8")) or {}
            items = data.get("news") or []
            loc = bits if code == "EN" else (tr.get(code.lower()) or {})
            for i, it in enumerate(items[:len(titles)]):
                info = PICKED[i] if i < len(PICKED) else {}
                good = (GOOD_INDEX == i) or bool(it.get("good"))
                it.update({"source": info.get("source") or it.get("source") or "",
                           "url": info.get("url") or it.get("url") or "", "good": good})
                if stories:
                    s = stories[i]
                    vis = dict(s.get("visual") or {"type": "icon"})
                    for key in ("label", "place", "place2"):
                        vals = loc.get(key) or []
                        if i < len(vals) and vals[i]:
                            vis[key] = vals[i]                     # translated (or English for EN)
                        elif key == "label" and code != "EN":
                            vis.pop(key, None)                     # no English label on a Spanish video
                    whys = loc.get("why") or []
                    it.update({"why": (whys[i] if i < len(whys) else "") or "",
                               "cat": "GOOD NEWS" if good else str(s.get("cat") or "WORLD").upper(),
                               "visual": vis})
                else:                                              # keep what's there, fill the minimum
                    it.setdefault("cat", "GOOD NEWS" if good else "WORLD")
                    it.setdefault("visual", {"type": "icon"})
                    it.setdefault("why", "")
            data["news"] = items
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=1000)
            done.append(code)
        except Exception as e:
            print(f"⚠️ Story details not written to {path} (non-fatal): {e}")
    print(f"✅ Story details in {', '.join(done) or 'no files'}: sources ✓ · categories/pictures "
          f"{'✓' if stories else '✗'} · ES/PT/ID short texts {'✓' if tr else '✗'}")


def details_only(day_name):
    """Repair today's files without re-picking the stories: python fetch_news.py <Day> --details-only"""
    import yaml
    data = yaml.safe_load(open(f"{day_name}NewsEN.yml", encoding="utf-8")) or {}
    news = [{"title": str(n.get("title", ""))} for n in data.get("news") or []]
    global GOOD_INDEX
    GOOD_INDEX = next((i for i, n in enumerate(data.get("news") or []) if n.get("good")), None)
    print(f"🔧 Adding story details to the existing {day_name} files ({len(news)} stories)")
    enrich_day_files(day_name, news)


if __name__ == "__main__":
    day_name = sys.argv[1]
    if "--details-only" in sys.argv[2:]:
        details_only(day_name)
    else:
        get_news(day_name)
