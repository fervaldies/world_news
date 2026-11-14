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
    numbered = trend_numbered(headlines)            # ★ on stories people are searching for
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


# ── Google Trends boost ───────────────────────────────────────────────────────
# Stories that match what people in the region are searching for right now get a small
# bonus when the 5 are picked (a ★ and a note to the AI: prefer it only when two stories
# are equally important). Free RSS feed, no key. If Trends fails, nothing changes.
TRENDS_GEOS = ["US", "GB", "IN", "CA"]       # Google Trends countries for this channel
TRENDING_TITLES = set()
_TREND_STOP = set("""a an and are as at be but by for from has have he her his in into is it its of on or
our over says she than that the their them they this to up was we were what when where which who why will
with you your after before about more most new news live today update latest vs""".split())
# one-word searches too general to mean "this story" (they still count inside longer searches)
_TREND_GENERIC = set("""weather rain storm heat news football soccer cricket tennis golf rugby nba nfl afl nrl lotto
powerball lottery game games movie movies music traffic google youtube facebook instagram netflix wordle horoscope
time today tomorrow results score scores election""".split())


def _trend_words(text, min_len=3):
    return {w for w in re.findall(r"[a-z0-9áéíóúñüàèìòùçãõâêôäöß']+", str(text).lower())
            if len(w) >= min_len and w not in _TREND_STOP}


def google_trends(geos):
    """[(term, [related news titles])] from Google Trends' 'Trending now' feed for these countries."""
    import xml.etree.ElementTree as ET
    out = []
    for geo in geos:
        for url in (f"https://trends.google.com/trending/rss?geo={geo}",
                    f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}"):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (news-reels)"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    root = ET.fromstring(r.read())
                n = 0
                for item in root.iter("item"):
                    term = (item.findtext("title") or "").strip()
                    news = [(el.text or "").strip() for el in item.iter() if el.tag.endswith("news_item_title")]
                    if term:
                        out.append((term, news))
                        n += 1
                if n:
                    break
            except Exception:
                continue
    return out


def trend_matches(headlines, trends):
    """Indexes of headlines that match a trending search: every meaningful word of the search term is in
    the headline, or the headline shares 3+ meaningful words with a news title linked to that search."""
    hits = {}
    for i, h in enumerate(headlines):
        hw = _trend_words(h)
        for term, news in trends:
            tw = _trend_words(term)
            if len(tw) == 1 and tw <= _TREND_GENERIC:
                tw = set()
            if tw and tw <= hw and any(len(w) >= 4 for w in tw):
                hits[i] = term
                break
            if any(len(hw & _trend_words(t, 4)) >= 3 for t in news):
                hits[i] = term
                break
    return hits


def trend_numbered(headlines):
    """The numbered list for the picker, with ★ on trending stories and a short rule at the end."""
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    try:
        trends = google_trends(TRENDS_GEOS)
        if not trends:
            print("📈 Google Trends: nothing received — picking as usual")
            return numbered
        hits = trend_matches(headlines, trends)
        TRENDING_TITLES.clear()
        TRENDING_TITLES.update(headlines[i] for i in hits)
        print(f"📈 Google Trends: {len(trends)} searches, {len(hits)} of {len(headlines)} stories match"
              + (": " + "; ".join(f"{headlines[i][:50]} ← {t}" for i, t in list(hits.items())[:5]) if hits else ""))
        if not hits:
            return numbered
        numbered = "\n".join(f"{i}. {'★ ' if i in hits else ''}{h}" for i, h in enumerate(headlines))
        return numbered + ("\n\n★ = people are searching for this right now. When two stories are equally "
                           "important, prefer the ★ one. Never pick a weaker or duplicate story just for ★, "
                           "and don't copy the ★ into anything.")
    except Exception as e:
        print(f"📈 Google Trends skipped ({e}) — picking as usual")
        return numbered


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
    ARTICLE_INFO.setdefault(title, {"source": src.strip(), "url": article.get("url") or "",
                                    "description": (article.get("description") or "")[:600],
                                    "content": (article.get("content") or "")[:1500]})    # for the chart figures


def remember_picked(news):
    global PICKED
    PICKED = [{**ARTICLE_INFO.get(n["title"], {}), "trending": n["title"] in TRENDING_TITLES} for n in news]
    if any(p["trending"] for p in PICKED):
        print(f"📈 {sum(p['trending'] for p in PICKED)} of the 5 picked stories are trending searches")


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


# ── Charts from the article's own figures (charts item 1) + number check (item 16) ─────────────
# One AI call (Gemini, Claude as fallback) reads each picked story's text (headline + GNews description
# + the start of the article) and returns the figures with the exact sentence each comes from. Then CODE
# checks every number against the article text; a chart that fails keeps the older number / map / icon.
CHART_KINDS = ("bars", "line", "donut", "slope", "bignum", "map", "waffle", "versus", "scale", "ladder", "range")
AU_PLACES = {"new south wales": "NSW", "nsw": "NSW", "victoria": "VIC", "vic": "VIC", "queensland": "QLD", "qld": "QLD",
             "western australia": "WA", "wa": "WA", "south australia": "SA", "sa": "SA", "tasmania": "TAS", "tas": "TAS",
             "northern territory": "NT", "nt": "NT", "australian capital territory": "ACT", "act": "ACT", "canberra": "ACT"}
EUROPE = {"AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
          "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "GBR", "NOR", "CHE", "ISL", "UKR",
          "SRB", "ALB", "BIH", "MKD", "MNE", "MDA", "BLR", "KOS", "XKX"}
COUNTRY_ISO = {  # names the articles use → ISO3 (Natural Earth codes)
    "austria": "AUT", "belgium": "BEL", "bulgaria": "BGR", "croatia": "HRV", "cyprus": "CYP", "czechia": "CZE",
    "czech republic": "CZE", "denmark": "DNK", "estonia": "EST", "finland": "FIN", "france": "FRA", "germany": "DEU",
    "greece": "GRC", "hungary": "HUN", "ireland": "IRL", "italy": "ITA", "latvia": "LVA", "lithuania": "LTU",
    "luxembourg": "LUX", "malta": "MLT", "netherlands": "NLD", "the netherlands": "NLD", "poland": "POL", "portugal": "PRT",
    "romania": "ROU", "slovakia": "SVK", "slovenia": "SVN", "spain": "ESP", "sweden": "SWE", "united kingdom": "GBR",
    "uk": "GBR", "britain": "GBR", "great britain": "GBR", "norway": "NOR", "switzerland": "CHE", "iceland": "ISL",
    "ukraine": "UKR", "serbia": "SRB", "albania": "ALB", "moldova": "MDA", "belarus": "BLR", "russia": "RUS",
    "turkey": "TUR", "türkiye": "TUR", "united states": "USA", "us": "USA", "usa": "USA", "canada": "CAN",
    "mexico": "MEX", "brazil": "BRA", "argentina": "ARG", "chile": "CHL", "colombia": "COL", "peru": "PER",
    "china": "CHN", "japan": "JPN", "south korea": "KOR", "korea": "KOR", "india": "IND", "indonesia": "IDN",
    "australia": "AUS", "new zealand": "NZL", "south africa": "ZAF", "nigeria": "NGA", "egypt": "EGY", "kenya": "KEN",
    "saudi arabia": "SAU", "iran": "IRN", "israel": "ISR", "pakistan": "PAK", "bangladesh": "BGD", "vietnam": "VNM",
    "thailand": "THA", "malaysia": "MYS", "philippines": "PHL", "singapore": "SGP", "taiwan": "TWN"}


def map_keys(labels):
    """(region, keys) for a list of place names, or (None, None) if any place isn't recognised."""
    low = [str(l).strip().lower() for l in labels]
    if low and all(l in AU_PLACES for l in low):
        return "au", [AU_PLACES[l] for l in low]
    keys = [COUNTRY_ISO.get(l) for l in low]
    if not low or any(k is None for k in keys):
        return None, None
    return ("europe" if all(k in EUROPE for k in keys) else "world"), keys
CHART_NOTES = []                         # what happened to every chart today (Telegram summary + logs)


def _num_tokens(text):
    """Every number written in a text, as floats (3.6 / 3,6 / 1,200 / 1.200 / -2)."""
    out = []
    for m in re.findall(r"(?<![\w])-?\d[\d.,]*", str(text)):
        m = m.rstrip(".,")
        if not m or m == "-":
            continue
        if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", m):             # thousands separators only
            out.append(float(re.sub(r"[.,]", "", m)))
        elif m.count(",") + m.count(".") == 1:
            out.append(float(m.replace(",", ".")))
        else:
            try:
                out.append(float(re.sub(r"[.,]", "", m)))
            except ValueError:
                pass
    return out


def _close(v, nums, dec):
    tol = 0.5 * 10 ** (-max(0, int(dec))) + 1e-9
    return any(abs(float(v) - n) <= tol for n in nums)


def _norm_text(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("’", "'")).strip()


def chart_values(c):
    """Every number the chart shows (all must be written in the article)."""
    k = c.get("chart")
    out = []

    def add(*vs):
        out.extend(v for v in vs if v is not None and not isinstance(v, bool))
    if k == "slope":
        for x in c.get("series") or []:
            add(x.get("from"), x.get("to"))
    elif k == "bignum":
        add(c.get("value"), c.get("previous"))
        if c.get("change") is not None and c.get("previous") is None:
            add(c["change"])
    elif k == "scale":
        for x in c.get("items") or []:
            add(x.get("value"))
    elif k == "versus":
        for x in c.get("items") or []:
            add(x.get("from"), x.get("to"))
    elif k == "range":
        for x in c.get("rows") or []:
            add(x.get("low"), x.get("high"), x.get("value"), x.get("margin"))
    else:
        add(*(c.get("values") or []))
    if k in ("line", "range") and c.get("band"):
        add(*c["band"])
    return out


ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
            "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth",
            "twentieth")


def check_chart(c, text, headline):
    """Problems that stop this chart (empty list = OK). Values, units, years, quotes, headline match."""
    problems = []
    T = _norm_text(text)
    nums = _num_tokens(text)
    dec = int(c.get("decimals", 1) or 0)
    k = c.get("chart")
    if k not in CHART_KINDS:
        return [f"unknown chart type '{k}'"]
    try:
        vals = [float(v) for v in chart_values(c)]
    except (TypeError, ValueError):
        return ["a value isn't a number"]
    n = len(c.get("values") or [])
    labs_n = len(c.get("labels") or [])
    if k == "bars" and not 2 <= n <= 8:
        problems.append(f"bars need 2–8 values (got {n})")
    if k == "line" and not 4 <= n <= 15:
        problems.append(f"a line needs 4+ points (got {n})")
    if k == "map" and not 3 <= n <= 60:
        problems.append(f"a map needs 3+ places (got {n})")
    if k == "donut" and not 2 <= n <= 6:
        problems.append(f"a donut needs 2–6 shares (got {n})")
    if k == "slope" and (len(c.get("periods") or []) != 2 or not c.get("series")):
        problems.append("a before/after chart needs two periods")
    if k == "bignum" and (c.get("value") is None or (c.get("previous") is None and c.get("change") is None)):
        problems.append("one number without a change — the count-up number is used instead")
    if k == "waffle":
        if not 1 <= n <= 3 or labs_n != n:
            problems.append("a waffle needs 1–3 shares, each with a label")
        elif any(not 1 <= float(v) <= 100 for v in c["values"]) or sum(float(v) for v in c["values"]) > 100.5:
            problems.append("waffle shares must be 1–100% and add up to 100% at most")
    if k == "scale" and not 2 <= len(c.get("items") or []) <= 4:
        problems.append("a size comparison needs 2–4 amounts")
    if k == "scale" and any(float(x.get("value") or 0) <= 0 for x in c.get("items") or []):
        problems.append("a size comparison needs amounts above zero")
    if k == "ladder" and c.get("from_rank") is not None:   # an earlier rank the article doesn't give: no move, chart kept
        try:
            fr = int(c["from_rank"])
            ok = _close(fr, nums, 0) or (1 <= fr <= 20 and re.search(rf"\b{ORDINALS[fr - 1]}\b", T)) or \
                re.search(rf"\b{fr}(st|nd|rd|th)\b", T)
        except (TypeError, ValueError):
            ok = False
        if not ok or not 1 <= fr <= len(c.get("values") or []):   # also: the old rank must be one of the rows shown
            c.pop("from_rank", None)
            c.pop("from_period", None)
    if k == "ladder":
        hl = c.get("highlight")
        if not 3 <= n <= 30 or labs_n != n:
            problems.append(f"a ranking needs 3+ places, each with a value (got {n})")
        elif not (isinstance(hl, int) and 0 <= hl < n):
            problems.append("a ranking needs the place to follow (highlight)")
    if k == "range":
        rows = c.get("rows") or []
        if not 2 <= len(rows) <= 8:
            problems.append("a range chart needs 2–8 rows")
        for r in rows:
            if (r.get("low") is None or r.get("high") is None) and (r.get("value") is None or r.get("margin") is None):
                problems.append("each range needs low + high (or value + margin)")
                break
    if k == "versus" and (len(c.get("periods") or []) != 2 or not 1 <= len(c.get("items") or []) <= 3):
        problems.append("then vs now needs two dates and 1–3 things")
    for v in vals:
        if not _close(v, nums, dec):
            problems.append(f"{v:g} isn't in the article")
    for q in c.get("quotes") or []:
        if q and _norm_text(q)[:70] not in T:
            problems.append("a quoted sentence isn't in the article")
            break
    unit = str(c.get("unit") or "")
    if (unit == "%" or k == "waffle") and not ("%" in text or "per cent" in T or "percent" in T or "pct" in T):
        problems.append("shown as % but the article has no percentage")
    for word, short in (("million", "m"), ("billion", "bn"), ("trillion", "tn")):       # 1.2 million / 1.2m
        if word in unit.lower() and word not in T and not re.search(rf"\d\s?{short}\b", T):
            problems.append(f"'{word}' isn't in the article")
    labs = [str(x) for x in (c.get("labels") or []) + (c.get("periods") or [])] + \
           [str(e.get("at", "")) for e in c.get("events") or []] + [str(c.get("from_period") or "")]
    for y in {m for l in labs for m in re.findall(r"\b(19\d\d|20\d\d)\b", l)}:
        if y not in text:
            problems.append(f"the year {y} isn't in the article")
    if k == "line":
        evs = c.get("events") or []
        if len(evs) > 3:
            problems.append("a line takes up to 3 events")
        for e in evs:
            if str(e.get("at")) not in [str(x) for x in c.get("labels") or []]:
                problems.append(f"the event date '{e.get('at')}' isn't on the line")
                break
    if k == "donut" and unit == "%" and vals and not 95 <= sum(vals) <= 105:
        problems.append(f"the shares add up to {sum(vals):g}%, not 100%")
    for m in re.finditer(r"(-?\d[\d.,]*)\s*(%|per ?cent)", str(headline)):        # headline vs chart
        hv = _num_tokens(m.group(1))
        if unit == "%" and hv and not _close(hv[0], vals + ([float(c["change"])] if c.get("change") is not None else []), dec):
            problems.append(f"the headline says {m.group(1)}% but the chart doesn't")
    return list(dict.fromkeys(problems))


RECORD_WORDS = ("record", "highest", "lowest", "biggest", "smallest", "first", "most", "least", "since", "ever", "fastest", "slowest")


def check_so_what(c, text):
    """The key-point line is kept only when the article backs it: real quote, same numbers/years, same record words."""
    sw = str(c.get("so_what") or "").strip()
    if not sw:
        return ""
    T = _norm_text(text)
    q = _norm_text(c.get("so_what_quote") or "")
    if not q or q[:70] not in T:
        return ""
    nums = _num_tokens(text)
    if any(not _close(n, nums, 2) for n in _num_tokens(sw)):
        return ""
    if any(w in sw.lower() and w not in T for w in RECORD_WORDS):
        return ""
    return sw[:60]


def extract_charts(titles):
    """Per story: (chart dict or None, note). One AI call for the 5 stories."""
    texts = []
    for i, t in enumerate(titles):
        info = PICKED[i] if i < len(PICKED) else {}
        texts.append(" ".join(x for x in (t, info.get("description", ""), info.get("content", "")) if x).strip())
    if not any(_num_tokens(x) for x in texts):
        return [(None, "no figures in the article") for _ in titles]
    blocks = "\n\n".join(f"STORY {i}\n{x}" for i, x in enumerate(texts))

    def check_shape(d):
        if not isinstance(d.get("stories"), list) or len(d["stories"]) != len(titles):
            raise ValueError("wrong number of stories")
    try:
        out = ai_json(
            "You prepare small charts for a news video. For each story use ONLY figures written in its text below — "
            "never estimate, calculate, convert or invent numbers. For every figure give the exact sentence it comes "
            "from (copy it from the text). Pick the chart by the data:\n"
            "- change over time with 4+ dates → line; add events (max 3: {\"at\": one of the line's labels, \"text\": "
            "max 3 words}) ONLY for dated events the text names, and band [low, high] + band_label ONLY when the text "
            "gives a target range (e.g. the 2–3% inflation target);\n"
            "- parts of one whole (shares that add up) → donut;\n"
            "- one or a few shares of a group in % (e.g. '25% of Australians rent', 'one in four') → waffle "
            "(values = the 1–3 shares in %, labels = what each share is, e.g. 'of Australians rent their home');\n"
            "- the price / cost / amount of 1–3 things then and now → versus (periods [then, now], items "
            "[{\"label\", \"from\", \"to\"}]);\n"
            "- other things at two moments → slope;\n"
            "- 2–4 amounts in the same unit where how big one is next to the others is the point (a cost vs a "
            "budget) → scale (items [{\"label\", \"value\"}], highlight = the amount the story is about);\n"
            "- where ONE place stands among 5+ others with a value for each → ladder (labels, values, highlight = "
            "that place's index, ascending true when lower is better (e.g. unemployment), from_rank + from_period ONLY "
            "when the text gives its earlier rank);\n"
            "- forecasts with a low–high range, or polls with a margin of error → range (rows [{\"label\", \"low\", "
            "\"high\", \"value\"}] or [{\"label\", \"value\", \"margin\"}], 2–8 rows; band + band_label only for a "
            "target range the text gives);\n"
            "- a ranking / comparison of 2–8 things → bars;\n"
            "- one value plus its previous value or its change → bignum;\n"
            "- values for 3+ Australian states/territories or 3+ countries → map (labels = the place names);\n"
            "- only ONE number, or nothing comparable → none (never force a weak chart).\n"
            "Set hedge true when the text says the figure is an estimate, forecast, preliminary, expected or "
            "unconfirmed. source = the organisation the text credits for the figures (e.g. 'ABS', 'Eurostat'), "
            "else the outlet name. title = max 6 words saying what is measured (no opinion).\n"
            "so_what = the key point in max 6 words ONLY if the text itself states it (e.g. 'Highest since 2009', "
            "'Lowest in the EU'), with so_what_quote = the exact sentence; otherwise empty strings. Never claim a "
            "record or a first unless the text says so.\n"
            "Return ONLY JSON: {\"stories\": [{\"chart\": \"bars|line|donut|slope|bignum|map|waffle|versus|scale|"
            "ladder|range|none\", \"title\": \"...\", "
            "\"unit\": \"% or a short unit or ''\", \"prefix\": \"'$' or ''\", \"decimals\": 1, "
            "\"labels\": [...], \"values\": [...], \"periods\": [\"2024\", \"2025\"], "
            "\"series\": [{\"label\": \"...\", \"from\": 1, \"to\": 2}], \"value\": 0, \"previous\": 0, "
            "\"change\": 0, \"change_unit\": \"% or pts\", \"label\": \"...\", \"period_prev\": \"...\", "
            "\"items\": [{\"label\": \"...\", \"value\": 0, \"from\": 0, \"to\": 0}], "
            "\"rows\": [{\"label\": \"...\", \"low\": 0, \"high\": 0, \"value\": 0, \"margin\": 0}], "
            "\"events\": [{\"at\": \"2020\", \"text\": \"...\"}], \"band\": [2, 3], \"band_label\": \"...\", "
            "\"ascending\": false, \"from_rank\": 0, \"from_period\": \"...\", "
            "\"highlight\": 0, \"source\": \"...\", \"hedge\": false, \"quotes\": [\"...\"], "
            "\"so_what\": \"\", \"so_what_quote\": \"\"}, ...one per story in order]}\n"
            "Leave out the fields a chart type doesn't use.\n\n" + blocks, 3500, check_shape)
    except Exception as e:
        return [(None, f"figures not read ({str(e)[:80]})") for _ in titles]
    res = []
    for i, c in enumerate(out["stories"]):
        if not isinstance(c, dict) or c.get("chart") in (None, "none", ""):
            res.append((None, "no chart-worthy figures"))
            continue
        if c.get("chart") == "map":                                              # place names → map outlines
            region, keys = map_keys(c.get("labels") or [])
            if region and len(keys) >= 3:
                c = {**c, "region": region, "keys": keys}
            elif 2 <= len(c.get("values") or []) <= 8:
                c = {**c, "chart": "bars"}                                       # unknown place: bars instead
            else:
                res.append((None, "places not recognised for a map"))
                continue
        if c.get("chart") == "line" and len(c.get("values") or []) == 2:         # two moments → before/after
            c = {**c, "chart": "slope", "periods": [str(x) for x in c.get("labels") or []],
                 "series": [{"label": c.get("title") or "", "from": c["values"][0], "to": c["values"][1]}]}
        elif c.get("chart") == "line" and len(c.get("values") or []) < 4:       # 3 dates: bars read better
            c["chart"] = "bars"
        problems = check_chart(c, texts[i], titles[i])
        if problems:
            res.append((None, "; ".join(problems[:3])))
            continue
        c["so_what"] = check_so_what(c, texts[i])
        res.append((c, "ok"))
    return res


def chart_spec(c, source_fallback):
    """The day-file `visual` for a checked chart (English texts; other languages are added later)."""
    k = c["chart"]
    s = {"type": "chart", "chart": k, "unit": str(c.get("unit") or ""), "prefix": str(c.get("prefix") or ""),
         "decimals": int(c.get("decimals", 1) or 0), "source": str(c.get("source") or source_fallback or ""),
         "hedge": bool(c.get("hedge")), "title_en": str(c.get("title") or "")}
    if c.get("so_what"):
        s["so_what_en"] = c["so_what"]
    if k == "slope":
        s.update(periods=[str(p) for p in c["periods"]],
                 series=[{"label_en": str(x.get("label", "")), "from": x["from"], "to": x["to"]} for x in c["series"]],
                 highlight=int(c.get("highlight") or 0))
    elif k == "bignum":
        s.update(value=c["value"], label_en=str(c.get("label") or ""), change_unit=str(c.get("change_unit") or ""))
        for key in ("previous", "change", "period_prev"):
            if c.get(key) is not None:
                s[key] = c[key]
    elif k == "map":
        s.update(region=c["region"], keys=c["keys"], labels_en=[str(x) for x in c.get("labels") or []], values=c["values"])
    elif k in ("scale", "versus"):
        s["items"] = [{"label_en": str(x.get("label") or ""), **({"value": x["value"]} if k == "scale" else
                       {"from": x["from"], "to": x["to"]})} for x in c["items"]]
        if k == "versus":
            s["periods"] = [str(p) for p in c["periods"]]
        if isinstance(c.get("highlight"), int):
            s["highlight"] = c["highlight"]
    elif k == "range":
        s["rows"] = [{"label_en": str(r.get("label") or ""), **{x: r[x] for x in ("low", "high", "value", "margin")
                                                               if r.get(x) is not None}} for r in c["rows"]]
        if isinstance(c.get("highlight"), int):
            s["highlight"] = c["highlight"]
    else:
        s.update(labels_en=[str(x) for x in (c.get("labels") or [])], values=c["values"])
        if isinstance(c.get("highlight"), int):
            s["highlight"] = c["highlight"]
        if k == "ladder":
            s["ascending"] = bool(c.get("ascending"))
            if c.get("from_rank"):
                s["from_rank"] = int(c["from_rank"])
                if c.get("from_period"):
                    s["from_period"] = str(c["from_period"])
        if k == "line" and c.get("events"):
            s["events"] = [{"at": str(e.get("at")), "text_en": str(e.get("text") or "")} for e in c["events"]][:3]
    if k in ("line", "range") and c.get("band") and len(c["band"]) == 2:
        s["band"] = list(c["band"])
        if c.get("band_label"):
            s["band_label_en"] = str(c["band_label"])
    return s


def translate_charts(specs):
    """{lang: [per-story dict of translated texts or None]} for es / pt / id (one AI call)."""
    items = []
    for s in specs:
        if not s:
            items.append(None)
            continue
        items.append({"title": s.get("title_en", ""), "labels": s.get("labels_en") or [], "label": s.get("label_en", ""),
                      "period_prev": str(s.get("period_prev") or ""), "so_what": s.get("so_what_en", ""),
                      "series": [x.get("label_en", "") for x in s.get("series") or []],
                      "rows": [x.get("label_en", "") for x in (s.get("items") or s.get("rows") or [])],
                      "events": [x.get("text_en", "") for x in s.get("events") or []],
                      "band_label": s.get("band_label_en", ""),
                      "unit": s["unit"] if s["unit"] and s["unit"] not in ("%", "pp", "pts", "k", "m", "bn") else ""})
    if not any(items):
        return {}

    def check(d):
        for lg in ("es", "pt", "id"):
            if not isinstance(d.get(lg), list) or len(d[lg]) != len(items):
                raise ValueError(f"no '{lg}' list")
    try:
        return ai_json("Translate the texts of these small news charts into Spanish from Spain (es), Brazilian "
                       "Portuguese (pt) and Indonesian (id). Translate place and country names the way that language "
                       "writes them; keep numbers, organisations and years exactly; keep it short; same structure "
                       "(null stays null). Return ONLY JSON: {\"es\": [...], \"pt\": [...], \"id\": [...]}\n\n"
                       + json.dumps(items, ensure_ascii=False), 3000, check)
    except Exception as e:
        print(f"⚠️ Chart texts not translated ({e}) — ES/PT/ID keep the older picture")
        return {}


def localize_chart(spec, tr, lang):
    """Add one language's texts to a chart; None if that language has no translation (never English on screen)."""
    if lang == "en":
        return spec
    t = tr if isinstance(tr, dict) else None
    if not t:
        return None
    s = dict(spec)
    s[f"title_{lang}"] = t.get("title") or ""
    if spec.get("labels_en"):
        if len(t.get("labels") or []) != len(spec["labels_en"]):
            return None
        s[f"labels_{lang}"] = t["labels"]
    if spec.get("label_en"):
        s[f"label_{lang}"] = t.get("label") or ""
    if spec.get("series"):
        names = t.get("series") or []
        if len(names) != len(spec["series"]):
            return None
        s["series"] = [{**x, f"label_{lang}": names[j]} for j, x in enumerate(spec["series"])]
    for key, sub in (("items", "label"), ("rows", "label"), ("events", "text")):     # texts inside the lists
        if spec.get(key):
            names = t.get("events" if key == "events" else "rows") or []
            if len(names) != len(spec[key]) or not all(names):
                return None
            s[key] = [{**x, f"{sub}_{lang}": names[j]} for j, x in enumerate(spec[key])]
    if spec.get("band_label_en"):
        if not t.get("band_label"):
            return None
        s[f"band_label_{lang}"] = t["band_label"]
    if t.get("unit"):
        s[f"unit_{lang}"] = t["unit"]
    if spec.get("period_prev") and t.get("period_prev"):
        s[f"period_prev_{lang}"] = t["period_prev"]
    if spec.get("so_what_en") and t.get("so_what"):
        s[f"so_what_{lang}"] = t["so_what"]
    return s


def chart_telegram(day_name):
    """One Telegram line per fetch when a chart was dropped by the number check."""
    bad = [n for n in CHART_NOTES if not n.startswith("✅")]
    bot, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not bad or not (bot and chat):
        return
    name = {"aus_news": "🇦🇺 Australia", "world_news": "🌍 World", "europe_news": "🇪🇺 Europe"}.get(
        os.path.basename(os.getcwd()), os.path.basename(os.getcwd()))
    text = f"📊 {name} charts ({day_name}): " + " · ".join(CHART_NOTES)
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text[:3500]}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{bot}/sendMessage", data=data, timeout=20)
    except Exception as e:
        print(f"⚠️ Telegram (charts): {e}")


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
    CHART_NOTES.clear()
    charts, chart_tr = [None] * len(titles), {}
    try:
        found = extract_charts(titles)
        for i, (c, note) in enumerate(found):
            if c:
                charts[i] = chart_spec(c, (PICKED[i] if i < len(PICKED) else {}).get("source"))
                CHART_NOTES.append(f"✅ story {i + 1}: {c['chart']}")
            elif note not in ("no chart-worthy figures", "no figures in the article"):
                CHART_NOTES.append(f"✖ story {i + 1}: {note}")
        if any(charts):
            chart_tr = translate_charts(charts)
        print("📊 Charts: " + (" · ".join(CHART_NOTES) if CHART_NOTES else "none today (no comparable figures)"))
    except Exception as e:
        print(f"⚠️ Chart step skipped (non-fatal): {e}")
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
                           "url": info.get("url") or it.get("url") or "", "good": good,
                           "trending": bool(info.get("trending") or it.get("trending"))})
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
                if i < len(charts) and charts[i]:              # checked chart, older picture kept as fallback
                    lang = code.lower()
                    tr_i = (chart_tr.get(lang) or [None] * len(titles))[i] if lang != "en" else None
                    spec = localize_chart(charts[i], tr_i, lang)
                    if spec:
                        it["visual"] = {**spec, "fallback": it.get("visual") or {"type": "icon"}}
                    it["chart_check"] = {"result": "ok" if spec else "no translation — older picture used",
                                         "template": charts[i]["chart"]}
                else:
                    note = next((n for n in CHART_NOTES if n.startswith(f"✖ story {i + 1}:")), None)
                    if note:
                        it["chart_check"] = {"result": "not used: " + note.split(":", 1)[1].strip()}
            data["news"] = items
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, width=1000)
            done.append(code)
        except Exception as e:
            print(f"⚠️ Story details not written to {path} (non-fatal): {e}")
    chart_telegram(day_name)
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
