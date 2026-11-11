"""
alerts.py — breaking-news alerts on Telegram, run entirely by GitHub (world_news, public repo).

    python alerts/alerts.py bot     light job: read Fer's taps, run the breaking check when due,
                                    draft special editions, ask Publish EN+ES / EN / ES / Cancel
    python alerts/alerts.py post    heavy job (reels repo checked out in ./reels): render, send the
                                    videos for approval, post, report

Uses its OWN bot (TELEGRAM_ALERTS_BOT_TOKEN) — Newsdesk never reads it, so only this job polls it.
Only messages from TELEGRAM_ALERTS_CHAT_ID are accepted; everything else is ignored.
State (offset, alerts, drafts) lives in the PRIVATE repo news-metrics, branch `alerts-state`,
committed with [skip ci]. Public-log safety: this script never prints headlines, drafts, captions,
chat IDs, tokens or API answers — only short status lines. Full logs of a failed render/upload go
to Fer on Telegram, not to the Actions log.
"""

import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = "fervaldies"
STATE_REPO, STATE_BRANCH, STATE_FILE = f"{OWNER}/news-metrics", "alerts-state", "alerts/state.json"
SYD = ZoneInfo("Australia/Sydney")

CHANNELS = {
    "australia": {"name": "Australia", "flag": "🇦🇺", "chk": "AUSTRALIA", "repo": "australia-news-reels",
                  "video": "DailyNewsVideo", "music": "daily_music",
                  "yt": {"en": "AUSTRALIA_YOUTUBE_REFRESH_TOKEN", "es": "AUSTRALIA_YOUTUBE_REFRESH_TOKEN",
                         "pt": "AUSTRALIA_YOUTUBE_PT_REFRESH_TOKEN", "id": "AUSTRALIA_YOUTUBE_ID_REFRESH_TOKEN"},
                  "radar": {"endpoint": "top-headlines", "country": "au"},
                  "names": ("Australia News", "Noticias de Australia")},
    "world": {"name": "World", "flag": "🌍", "chk": "WORLD", "repo": "world-news-reels",
              "video": "worldvideo", "music": "worldmusic",
              "yt": {"en": "WORLD_YOUTUBE_EN_REFRESH_TOKEN", "es": "WORLD_YOUTUBE_ES_REFRESH_TOKEN",
                     "pt": "WORLD_YOUTUBE_PT_REFRESH_TOKEN", "id": "WORLD_YOUTUBE_ID_REFRESH_TOKEN"},
              "radar": {"endpoint": "top-headlines", "category": "world"},
              "names": ("World News", "Noticias del Mundo")},
    "europe": {"name": "Europe", "flag": "🇪🇺", "chk": "EUROPE", "repo": "europe-news-reels",
               "video": "eurovideo", "music": "euromusic",
               "yt": {"en": "EUROPE_YOUTUBE_EN_REFRESH_TOKEN", "es": "EUROPE_YOUTUBE_ES_REFRESH_TOKEN",
                      "pt": "EUROPE_YOUTUBE_PT_REFRESH_TOKEN", "id": "EUROPE_YOUTUBE_ID_REFRESH_TOKEN"},
               "radar": {"endpoint": "search", "q": "Europe OR EU OR European"},
               "names": ("Europe News", "Noticias de Europa")},
}
LANGS = ["en", "es", "pt", "id"]
FLAGS = {"en": "🇬🇧", "es": "🇪🇸", "pt": "🇧🇷", "id": "🇮🇩"}
# Portuguese / Indonesian: translated from the English draft, exactly as Newsdesk does it
EXTRA = {
    "pt": {"style": "Brazilian Portuguese (pt-BR) as written by Brazilian news media; Brazilian number formats "
                    "(US$ 6,3 bilhões; 2,8%)",
           "names": {"australia": "Notícias da Austrália", "world": "Notícias do Mundo", "europe": "Notícias da Europa"},
           "label": "Edição Especial",
           "months": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
                      "outubro", "novembro", "dezembro"], "date": "{d} de {m} de {y}"},
    "id": {"style": "natural Bahasa Indonesia news style (as in Kompas or Tempo); Indonesian number formats "
                    "(6,3 miliar dolar AS; 2,8 persen)",
           "names": {"australia": "Berita Australia", "world": "Berita Dunia", "europe": "Berita Eropa"},
           "label": "Edisi Khusus",
           "months": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September",
                      "Oktober", "November", "Desember"], "date": "{d} {m} {y}"},
}
MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
             "octubre", "noviembre", "diciembre"]

# settings (repo variables in world_news; sensible defaults)
ALERT_HOURS = [int(h) for h in (os.environ.get("ALERT_HOURS") or "7,10,13,16,19,22").split(",") if h.strip()]
LISTEN_MIN = int(os.environ.get("ALERT_LISTEN_MIN") or 25)          # live listening after an alert / a draft
MAX_JOB_MIN = 50                                                    # bot job never runs longer than this
APPROVE_VIDEO = (os.environ.get("ALERT_APPROVE_VIDEO") or "true").lower() != "false"
APPROVE_WAIT_MIN = int(os.environ.get("ALERT_APPROVE_MIN") or 40)
GEMINI_MODEL = os.environ.get("ALERT_GEMINI_MODEL") or "gemini-3.5-flash-lite"
CLAUDE_MODEL = os.environ.get("ALERT_CLAUDE_MODEL") or "claude-haiku-4-5-20251001"
ALERT_TTL_H, DRAFT_TTL_H = 8, 12

BOT = os.environ.get("TELEGRAM_ALERTS_BOT_TOKEN", "")
CHAT = str(os.environ.get("TELEGRAM_ALERTS_CHAT_ID", "")).strip()
PAT = os.environ.get("GH_PAT", "")


def log(msg):
    """Public log: status only (callers never pass headlines, drafts, IDs or tokens)."""
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


def now():
    return datetime.now(timezone.utc)


def iso(d=None):
    return (d or now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return now() - timedelta(days=365)


# ── HTTP helpers ─────────────────────────────────────────────────────────────
def http_json(method, url, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    h = {"User-Agent": "news-alerts", **({"Content-Type": "application/json"} if data else {}), **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def tg(method, **params):
    """Telegram Bot API call. Never logs the answer (it contains the chat ID)."""
    try:
        return http_json("POST", f"https://api.telegram.org/bot{BOT}/{method}", params, timeout=70).get("result")
    except urllib.error.HTTPError as e:
        log(f"Telegram {method}: HTTP {e.code}")
    except Exception as e:
        log(f"Telegram {method}: {type(e).__name__}")
    return None


def tg_file(method, field, path, **params):
    """Multipart upload (sendVideo / sendDocument)."""
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in params.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    with open(path, "rb") as f:
        content = f.read()
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{os.path.basename(path)}"\r\n'
                  "Content-Type: application/octet-stream\r\n\r\n").encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"https://api.telegram.org/bot{BOT}/{method}", data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()).get("result")
    except Exception as e:
        log(f"Telegram {method} (file): {type(e).__name__}")
        return None


def say(text, buttons=None):
    """Message to Fer (plain text). buttons = [[(label, data), ...], ...]."""
    p = {"chat_id": CHAT, "text": text[:4000], "disable_web_page_preview": True}
    if buttons:
        p["reply_markup"] = {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in buttons]}
    return tg("sendMessage", **p)


def drop_buttons(msg):
    if msg and msg.get("message_id"):
        tg("editMessageReplyMarkup", chat_id=CHAT, message_id=msg["message_id"], reply_markup={"inline_keyboard": []})


# ── state (private: news-metrics, branch alerts-state) ───────────────────────
def gh(method, path, body=None):
    return http_json(method, f"https://api.github.com{path}", body,
                     {"Authorization": f"token {PAT}", "Accept": "application/vnd.github+json"})


def load_state():
    try:
        d = gh("GET", f"/repos/{STATE_REPO}/contents/{STATE_FILE}?ref={STATE_BRANCH}")
        import base64
        st = json.loads(base64.b64decode(d["content"]).decode("utf-8"))
        st["_sha"] = d["sha"]
        return st
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    try:                                                     # first run: create the branch from main
        gh("GET", f"/repos/{STATE_REPO}/branches/{STATE_BRANCH}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sha = gh("GET", f"/repos/{STATE_REPO}/git/ref/heads/main")["object"]["sha"]
            gh("POST", f"/repos/{STATE_REPO}/git/refs", {"ref": f"refs/heads/{STATE_BRANCH}", "sha": sha})
            log("State branch created")
    return {"offset": 0, "alerts": {}, "drafts": {}, "last_check": None, "seen": []}


def save_state(st):
    import base64
    cutoff = now() - timedelta(days=2)                      # keep it small
    st["alerts"] = {k: v for k, v in st["alerts"].items() if parse_iso(v["t"]) > cutoff}
    st["drafts"] = {k: v for k, v in st["drafts"].items() if parse_iso(v["t"]) > cutoff}
    st["seen"] = [s for s in st.get("seen", []) if parse_iso(s["t"]) > now() - timedelta(hours=30)][-200:]
    body = {k: v for k, v in st.items() if not k.startswith("_")}
    req = {"message": "[skip ci] alerts state", "branch": STATE_BRANCH,
           "content": base64.b64encode(json.dumps(body, ensure_ascii=False, indent=1).encode()).decode()}
    if st.get("_sha"):
        req["sha"] = st["_sha"]
    for attempt in range(3):
        try:
            st["_sha"] = gh("PUT", f"/repos/{STATE_REPO}/contents/{STATE_FILE}", req)["content"]["sha"]
            return
        except urllib.error.HTTPError as e:
            if e.code in (409, 422) and attempt < 2:                # someone else wrote: take their sha, keep ours
                try:
                    req["sha"] = gh("GET", f"/repos/{STATE_REPO}/contents/{STATE_FILE}?ref={STATE_BRANCH}")["sha"]
                except Exception:
                    req.pop("sha", None)
                continue
            raise


# ── AI ───────────────────────────────────────────────────────────────────────
def gemini(prompt, system=None, search=False, max_tokens=2500):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise Exception("no Gemini key")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={urllib.parse.quote(key)}"
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens}}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if search:
        try:
            d = http_json("POST", url, {**body, "tools": [{"google_search": {}}]}, timeout=120)
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        except Exception:
            pass                                           # no search quota: write from the GNews articles only
    d = http_json("POST", url, body, timeout=120)
    return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])


def claude(prompt, system=None, search=False, max_tokens=2500):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise Exception("no Anthropic key")
    body = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    if search:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    d = http_json("POST", "https://api.anthropic.com/v1/messages", body,
                  {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=180)
    return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")


def ask(prompt, system=None, search=False, prefer="gemini", max_tokens=2500):
    """Gemini first (free), Claude as fallback (or the other way round when asked)."""
    order = (gemini, claude) if prefer == "gemini" else (claude, gemini)
    last = None
    for fn in order:
        try:
            out = fn(prompt, system, search, max_tokens)
            if out.strip():
                return out, fn.__name__
        except Exception as e:
            last = e
            log(f"{fn.__name__} unavailable ({type(e).__name__})")
    raise Exception(f"no AI answered ({type(last).__name__})")


def json_from(text):
    s = text.find("{")
    depth = 0
    for i in range(s, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return json.loads(text[s:i + 1])
    raise ValueError("no JSON")


# ── breaking check ───────────────────────────────────────────────────────────
def gnews(endpoint, **params):
    key = os.environ.get("GNEWS_API_KEY")
    q = urllib.parse.urlencode({"lang": "en", "max": 10, "apikey": key, **params})
    return http_json("GET", f"https://gnews.io/api/v4/{endpoint}?{q}", timeout=30).get("articles", [])


def words(t):
    return {w for w in re.findall(r"[a-z0-9]+", str(t).lower()) if len(w) >= 4}


def breaking_check(st):
    """3 GNews requests (one per channel) + one AI call. Returns new big stories (not alerted in the last 30 h)."""
    since = (now() - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines, used = [], 0
    for ck, c in CHANNELS.items():
        p = dict(c["radar"])
        ep = p.pop("endpoint")
        if ep == "search":
            p.update({"sortby": "publishedAt", "from": since})
        try:
            arts = gnews(ep, **p)
            used += 1
        except Exception as e:
            log(f"GNews {ck}: {type(e).__name__}")
            continue
        lines.append(f"\n[{ck}] {c['name']}")
        lines += [f"- {a.get('title', '')} — {(a.get('source') or {}).get('name', '')}" for a in arts if a.get("title")]
        time.sleep(1)
    st.setdefault("gnews", {})
    day = datetime.now(SYD).strftime("%Y-%m-%d")
    st["gnews"] = {day: st["gnews"].get(day, 0) + used}
    if not lines:
        return []
    prompt = (
        "You are the breaking-news desk for short-video news channels. Below are the newest headlines per channel "
        "(the label in brackets is the channel key). Pick the stories that are genuinely big or breaking right now: "
        "major impact, or covered by several outlets. Group duplicates as one story. Return 0 to 5 stories, most "
        "important first, as JSON only:\n"
        '{"items":[{"channel":"<channel key>","topic":"<2-3 word topic for a special edition title>",'
        '"headline":"<what happened, max 15 words>","sources":<number of headlines about it>}]}\n'
        'If nothing is big enough, return {"items":[]}.\n' + "\n".join(lines))
    out, _ = ask(prompt, max_tokens=800)
    items = []
    for it in json_from(out).get("items", [])[:5]:
        if it.get("channel") in CHANNELS and it.get("topic"):
            items.append({"channel": it["channel"], "topic": str(it["topic"])[:40],
                          "headline": str(it.get("headline", ""))[:160]})
    seen = st.get("seen", [])
    fresh = []
    for it in items:                                           # skip stories already alerted in the last 30 h
        w = words(it["topic"] + " " + it["headline"])
        if any(s["channel"] == it["channel"] and len(w & set(s["w"])) >= 3 for s in seen):
            continue
        fresh.append(it)
        seen.append({"t": iso(), "channel": it["channel"], "w": sorted(w)})
    st["seen"] = seen
    return fresh


def due(st, force=False):
    if force:
        return True
    h = datetime.now(SYD).hour
    last = parse_iso(st["last_check"]) if st.get("last_check") else None
    return h in ALERT_HOURS and (last is None or now() - last > timedelta(minutes=50))


def send_alert(st, items):
    aid = uuid.uuid4().hex[:6]
    st["alerts"][aid] = {"t": iso(), "items": items}
    text = "🚨 Breaking news check — big stories right now:\n\n" + "\n".join(
        f"{i + 1}. {CHANNELS[it['channel']]['flag']} {it['topic']} — {it['headline']}" for i, it in enumerate(items)) + \
        "\n\nTap one to draft a special edition."
    rows = [[(f"{i + 1}. {CHANNELS[it['channel']]['flag']} {it['topic']}"[:60], f"s:{aid}:{i}")] for i, it in enumerate(items)]
    rows.append([("✖️ Dismiss", f"d:{aid}")])
    say(text, rows)
    log(f"Alert sent ({len(items)} stories)")


# ── special edition writer ───────────────────────────────────────────────────
def special_system(ck):
    p = os.path.join(HERE, "prompts", f"{ck}.txt")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    en, es = CHANNELS[ck]["names"]
    return (f"You write special editions for the short-video news channel '{en}' / '{es}'. Find the 5 most important "
            "and recent developments on the topic, each from a different angle, as factual headlines under 18 words "
            "(no source names, prefixes, emojis, citations or tags), and translate each into Spain Spanish. "
            "If something is suspected or preliminary, say so. Topic in the titles: 3 words at most. Reply with YAML "
            f'only:\ntitle_en: "{en}\\nSpecial Edition\\n<Topic>\\n<Date>"\ntitle_es: "{es}\\nEdición Especial\\n<Tema>\\n<Fecha>"\n'
            "news_en: [5 headlines]\nnews_es: [5 headlines]\n")


def write_special(ck, topic, headline, prefer="gemini"):
    d = datetime.now(SYD)
    date_en, date_es = f"{d:%B} {d.day}, {d.year}", f"{d.day} de {MONTHS_ES[d.month - 1]} de {d.year}"
    region = {"australia": " Australia", "europe": " Europe"}.get(ck, "")
    arts = []
    try:
        for a in gnews("search", q=f"{topic}{region}", sortby="publishedAt"):
            arts.append(f"- {a.get('publishedAt', '')[:10]} · {(a.get('source') or {}).get('name', '')}: "
                        f"{a.get('title', '')} — {a.get('description', '')}")
    except Exception as e:
        log(f"GNews (topic): {type(e).__name__}")
    user = (f"Topic: {topic}\nWhat happened: {headline}\nToday's date: {date_en} (Spanish: {date_es}).")
    if arts:
        user += "\n\nLatest articles on this topic (use them, and search for more if you can):\n" + "\n".join(arts)
    out, who = ask(user, special_system(ck), search=True, prefer=prefer)
    out = re.sub(r"</?cite\b[^>]*>", "", out)
    block = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", out, flags=re.S)
    if block:
        out = block.group(1)
    i = out.find("title_en:")
    out = out[i:] if i >= 0 else out
    y = yaml.safe_load(out) or {}
    for k in ("title_en", "title_es"):
        if not isinstance(y.get(k), str):
            raise Exception(f"draft is missing {k}")
    for k in ("news_en", "news_es"):
        if not isinstance(y.get(k), list) or len([x for x in y[k] if str(x).strip()]) < 5:
            raise Exception(f"draft needs 5 {k}")
        y[k] = [str(x).strip() for x in y[k] if str(x).strip()][:5]
    return {"title_en": y["title_en"], "title_es": y["title_es"], "news_en": y["news_en"], "news_es": y["news_es"]}, who


def add_extra(ck, y, prefer="gemini"):
    """Adds title_pt/news_pt and title_id/news_id (translated from English). Returns the languages that failed."""
    lines = [l.strip() for l in str(y["title_en"]).split("\n") if l.strip()]
    topic = lines[2] if len(lines) >= 4 else (lines[-2] if len(lines) >= 2 else "")
    news = y["news_en"]
    d = datetime.now(SYD)
    failed = []
    for l, x in EXTRA.items():
        try:
            out, _ = ask(f"Translate this news topic and these {len(news)} headlines into {x['style']}. Natural headline "
                         "style, not literal; keep every number, name, place and organisation exactly; same order; "
                         "each headline under 18 words; the topic in 3 words max. Return ONLY raw JSON, no markdown:\n"
                         '{"topic": "...", "news": ["...", "..."]}\n\n'
                         f"Topic: {topic}\nHeadlines:\n" + "\n".join(f"- {h}" for h in news), prefer=prefer, max_tokens=1500)
            j = json_from(out)
            tr = [str(h).strip() for h in j.get("news", []) if str(h).strip()]
            if len(tr) != len(news):
                raise ValueError("wrong number of headlines")
            date = x["date"].format(d=d.day, m=x["months"][d.month - 1], y=d.year)
            y[f"title_{l}"] = f"{x['names'][ck]}\n{x['label']}\n{j.get('topic') or topic}\n{date}"
            y[f"news_{l}"] = tr
        except Exception as e:
            log(f"{l.upper()} translation failed ({type(e).__name__})")
            failed.append(l)
    return failed


def preview(draft):
    y = draft["yaml"]
    c = CHANNELS[draft["channel"]]
    t = lambda s: " · ".join(x for x in str(s).split("\n") if x.strip())
    blocks = []
    for l in LANGS:
        if y.get(f"news_{l}"):
            blocks.append(f"{FLAGS[l]} {t(y[f'title_{l}'])}\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(y[f"news_{l}"])))
        else:
            blocks.append(f"{FLAGS[l]} {l.upper()}: translation failed — can't be posted this time")
    return (f"🔴 {c['flag']} Special edition draft ({draft['writer']})\n\n" + "\n\n".join(blocks) +
            "\n\nCheck the facts. Tap a language to switch it on/off, then Publish:")


def draft_buttons(did, d):
    have = [l for l in LANGS if d["yaml"].get(f"news_{l}")]
    sel = d.get("sel", have)
    toggles = [((("✅ " if l in sel else "⬜ ") + l.upper()), f"t:{did}:{l}") for l in have]
    n = len([l for l in LANGS if l in sel])
    return [toggles, [(f"🚀 Publish {n} language{'s' if n != 1 else ''}" if n else "Pick a language first", f"p:{did}:sel")],
            [("✍️ Rewrite with Claude", f"r:{did}"), ("✖️ Cancel", f"p:{did}:x")]]


# ── reading Fer's taps ───────────────────────────────────────────────────────
def mine(upd):
    """Only Fer's chat counts. Returns (kind, payload, callback) or None."""
    if "callback_query" in upd:
        cq = upd["callback_query"]
        chat = str(((cq.get("message") or {}).get("chat") or {}).get("id", ""))
        if chat != CHAT or str((cq.get("from") or {}).get("id", "")) != CHAT:
            return None
        return "tap", cq.get("data", ""), cq
    m = upd.get("message") or {}
    if str((m.get("chat") or {}).get("id", "")) != CHAT:
        return None
    return "text", (m.get("text") or "").strip(), m


def updates(st, wait):
    res = tg("getUpdates", offset=st.get("offset", 0), timeout=wait, allowed_updates=["message", "callback_query"]) or []
    for u in res:
        st["offset"] = max(st.get("offset", 0), u["update_id"] + 1)
    return res


def handle(st, upd, allow_publish=True):
    """Act on one update. Returns ('post', draft_id) when Fer pressed Publish; None otherwise."""
    got = mine(upd)
    if not got:
        return None                                                  # someone else: ignore silently
    kind, data, obj = got
    if kind == "tap":
        tg("answerCallbackQuery", callback_query_id=obj["id"])
    if kind == "text":
        cmd = data.lower()
        if cmd in ("check", "/check", "breaking"):
            st["_force_check"] = True
        elif cmd in ("status", "/status", "/start", "help", "/help"):
            last = st.get("last_check")
            last_txt = f"{parse_iso(last).astimezone(SYD):%a %H:%M}" if last else "none yet"
            say(f"Last check: {last_txt} (Sydney).\n" + "🤖 Breaking-news alerts are on.\n"
                f"Checks at {', '.join(f'{h}:00' for h in ALERT_HOURS)} (Sydney time); taps are read every ~10 min, "
                f"or within seconds for {LISTEN_MIN} min after each alert or draft.\n"
                "Send CHECK to run a check now.")
        return None
    parts = data.split(":")
    msg = obj.get("message")
    if parts[0] == "d":                                              # dismiss an alert
        drop_buttons(msg)
        return None
    if parts[0] == "s" and len(parts) == 3:                          # story tapped → draft
        a = st["alerts"].get(parts[1])
        if not a or now() - parse_iso(a["t"]) > timedelta(hours=ALERT_TTL_H):
            say("⌛ That alert has expired — send CHECK for a fresh one.")
            return None
        it = a["items"][int(parts[2])]
        drop_buttons(msg)
        say(f"✍️ Drafting the {CHANNELS[it['channel']]['flag']} special edition: {it['topic']}…")
        return draft(st, it["channel"], it["topic"], it["headline"], "gemini")
    if parts[0] == "r" and len(parts) == 2:                          # rewrite with Claude
        d = st["drafts"].get(parts[1])
        if not d or d["status"] != "drafted":
            return None
        drop_buttons(msg)
        say("✍️ Rewriting with Claude…")
        return draft(st, d["channel"], d["topic"], d["headline"], "claude", replace=parts[1])
    if parts[0] == "t" and len(parts) == 3:                          # language on/off
        d = st["drafts"].get(parts[1])
        if not d or d["status"] != "drafted" or not d["yaml"].get(f"news_{parts[2]}"):
            return None
        sel = d.setdefault("sel", [l for l in LANGS if d["yaml"].get(f"news_{l}")])
        d["sel"] = [l for l in LANGS if (l in sel) != (l == parts[2])]
        if msg and msg.get("message_id"):
            tg("editMessageReplyMarkup", chat_id=CHAT, message_id=msg["message_id"],
               reply_markup={"inline_keyboard": [[{"text": t, "callback_data": x} for t, x in row]
                                                 for row in draft_buttons(parts[1], d)]})
        return None
    if parts[0] == "p" and len(parts) == 3:                          # publish choice
        d = st["drafts"].get(parts[1])
        if not d or d["status"] != "drafted" or now() - parse_iso(d["t"]) > timedelta(hours=DRAFT_TTL_H):
            say("⌛ That draft has expired or was already handled.")
            return None
        if parts[2] == "x":
            drop_buttons(msg)
            d["status"] = "cancelled"
            say("✖️ Cancelled — nothing posted.")
            return None
        sel = [l for l in LANGS if l in d.get("sel", LANGS) and d["yaml"].get(f"news_{l}")]
        if not sel:
            tg("sendMessage", chat_id=CHAT, text="Switch on at least one language first.")
            return None
        drop_buttons(msg)
        if not allow_publish:
            say("⏳ Busy finishing another special — tap Publish again in a few minutes.", draft_buttons(parts[1], d))
            return None
        d["langs"] = sel
        d["status"] = "queued"
        mins = 3 + 2 * len(d["langs"])
        say(f"🎬 Making the {' + '.join(l.upper() for l in d['langs'])} video{'s' if len(d['langs']) > 1 else ''}"
            + (f" — you'll get {'them' if len(d['langs']) > 1 else 'it'} here to approve before posting (about {mins}–{mins + 4} min)."
               if APPROVE_VIDEO else f" and posting (about {mins + 2}–{mins + 8} min)."))
        return ("post", parts[1])
    return None


def draft(st, ck, topic, headline, prefer, replace=None):
    try:
        y, who = write_special(ck, topic, headline, prefer)
    except Exception as e:
        log(f"Draft failed ({type(e).__name__})")
        say("⚠️ Couldn't write the draft (AI unavailable or the answer was unusable). Tap the story again, or try later.")
        return None
    add_extra(ck, y, "claude" if who == "claude" else "gemini")      # PT + ID from the English draft
    did = replace or uuid.uuid4().hex[:6]
    st["drafts"][did] = {"t": iso(), "channel": ck, "topic": topic, "headline": headline, "yaml": y,
                         "writer": "Claude" if who == "claude" else "Gemini", "status": "drafted",
                         "sel": [l for l in LANGS if y.get(f"news_{l}")]}
    say(preview(st["drafts"][did]), draft_buttons(did, st["drafts"][did]))
    log("Draft sent")
    return None


# ── job 1: bot ───────────────────────────────────────────────────────────────
def run_bot():
    if not (BOT and CHAT and PAT):
        log("Alerts bot secrets are missing — nothing to do")
        return
    st = load_state()
    start = time.time()
    listen_until = 0.0
    force = os.environ.get("CHECK_NOW") == "true"
    # still-listening window if a draft is waiting for an answer
    if any(d["status"] == "drafted" and now() - parse_iso(d["t"]) < timedelta(minutes=LISTEN_MIN) for d in st["drafts"].values()):
        listen_until = time.time() + 60 * LISTEN_MIN
    result = None
    while True:
        if due(st, force or st.pop("_force_check", False)):
            force = False
            st["last_check"] = iso()
            try:
                items = breaking_check(st)
                log(f"Breaking check: {len(items)} new big stor{'y' if len(items) == 1 else 'ies'}")
                if items:
                    send_alert(st, items)
                    listen_until = time.time() + 60 * LISTEN_MIN
            except Exception as e:
                log(f"Breaking check failed ({type(e).__name__})")
            save_state(st)
        waiting = time.time() < listen_until and time.time() - start < 60 * MAX_JOB_MIN
        ups = updates(st, 25 if waiting else 0)
        for u in ups:
            r = handle(st, u)
            if mine(u):
                listen_until = max(listen_until, time.time() + 60 * LISTEN_MIN)
            if r and r[0] == "post":
                result = r[1]
                break
        if ups:
            save_state(st)
        if result:
            break
        if st.get("_force_check"):
            continue
        if not (time.time() < listen_until and time.time() - start < 60 * MAX_JOB_MIN):
            break
    save_state(st)
    out = os.environ.get("GITHUB_OUTPUT")
    if result:
        d = st["drafts"][result]
        c = CHANNELS[d["channel"]]
        log(f"Publishing requested ({d['channel']}, {'+'.join(d['langs'])})")
        if out:
            with open(out, "a") as f:
                f.write(f"post=true\ndraft={result}\nrepo={c['repo']}\nchk={c['chk']}\n"
                        + "".join(f"yt_{l}={c['yt'][l]}\n" for l in LANGS))
    elif out:
        with open(out, "a") as f:
            f.write("post=false\n")


# ── job 2: render, approve, post ─────────────────────────────────────────────
def already_posted(ck, lang, y):
    """Duplicate guard: a special for this channel + language in the last 24 h with the same lead headline or topic."""
    try:
        tree = gh("GET", f"/repos/{STATE_REPO}/git/trees/main?recursive=1").get("tree", [])
    except Exception:
        return False
    days = {(now() - timedelta(days=k)).strftime("%Y-%m-%d") for k in (0, 1)}
    names = [t["path"] for t in tree if t["path"].startswith("posts/") and "_special_" in t["path"]
             and any(f"/{d}_{ck}_{lang}_" in t["path"] for d in days)]
    lead = str(y.get(f"news_{lang}", [""])[0]).strip().lower()
    topic = str(y.get(f"title_{lang}", "")).split("\n")[2:3]
    import base64
    for p in names[:20]:
        try:
            r = json.loads(base64.b64decode(gh("GET", f"/repos/{STATE_REPO}/contents/{p}")["content"]))
        except Exception:
            continue
        heads = [str(h).strip().lower() for h in (r.get("headlines") or [])]
        if (lead and lead in heads) or (topic and topic[0].strip().lower() in str(r.get("title", "")).lower()):
            return True
    return False


def sh(cmd, logf, env=None, cwd="reels", timeout=1800):
    with open(logf, "a") as f:
        f.write(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env={**os.environ, **(env or {})},
                           stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode == 0


def pick(prefix, ext):
    files = [f for f in glob.glob(f"reels/{prefix}*.{ext}") if re.fullmatch(rf"{prefix}\d*\.{ext}", os.path.basename(f))]
    return os.path.basename(random.choice(files)) if files else None


def wait_approval(st, did, video_msgs):
    """Long-poll for Post / Cancel on the videos. Other taps are answered with a 'busy' note."""
    end = time.time() + 60 * APPROVE_WAIT_MIN
    while time.time() < end:
        for u in updates(st, 25):
            got = mine(u)
            if not got:
                continue
            kind, data, obj = got
            if kind == "tap" and data in (f"a:{did}:ok", f"a:{did}:no"):
                tg("answerCallbackQuery", callback_query_id=obj["id"])
                for m in video_msgs:
                    drop_buttons(m)
                return data.endswith(":ok")
            handle(st, u, allow_publish=False)       # a story tap still gets drafted; Publish waits
        save_state(st)
    return False


def run_post():
    os.environ["GITHUB_TOKEN"] = PAT          # the upload scripts write the news-metrics records with it
    st = load_state()
    did = os.environ["DRAFT_ID"]
    d = st["drafts"].get(did)
    if not d or d["status"] != "queued":
        log("Nothing queued")
        return
    ck, c, y = d["channel"], CHANNELS[d["channel"]], d["yaml"]
    d["status"] = "rendering"
    save_state(st)
    os.makedirs("logs", exist_ok=True)
    with open("reels/alert_special.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(y, f, allow_unicode=True, sort_keys=False, width=1000)
    done, failed, skipped, made = [], [], [], {}
    for lang in d["langs"]:
        L = lang.upper()
        if already_posted(ck, lang, y):
            skipped.append(L)
            log(f"{L}: already posted — skipped")
            continue
        logf = os.path.abspath(f"logs/{lang}.log")
        v, m = pick(c["video"], "mp4"), pick(c["music"], "mp3")
        ok = bool(v and m) and sh(["python", "make_reels_es_en.py", "--yml", "alert_special.yml", "--video", v, "--music", m,
                                   "--lang", lang, "--output-dir", f"alert_out_{lang}", "--no-week-label", "--title-color", "red"], logf)
        vids = glob.glob(f"reels/alert_out_{lang}/reel_{lang}_*.mp4")
        caps = glob.glob(f"reels/alert_out_{lang}/caption_{lang}_*.txt")
        if ok and vids and caps:
            made[lang] = (vids[0], caps[0], logf)
            log(f"{L}: video made")
        else:
            failed.append(L)
            log(f"{L}: video failed")
            tg_file("sendDocument", "document", logf, chat_id=CHAT, caption=f"⚠️ {c['flag']} {L} video failed — log")
    if not made:
        d["status"] = "failed" if failed else "skipped"
        save_state(st)
        say(f"{'⚠️ Nothing posted.' if failed else '⏭️ Already posted — nothing to do.'}"
            + (f" Skipped (already posted): {', '.join(skipped)}." if skipped else ""))
        return
    approved = True
    if APPROVE_VIDEO:
        msgs = []
        for lang, (vid, cap, _) in made.items():
            msgs.append(tg_file("sendVideo", "video", vid, chat_id=CHAT, supports_streaming="true",
                                caption=f"🔴 {c['flag']} Special edition {lang.upper()} — ready"))
        ask_msg = say(f"Post {' + '.join(l.upper() for l in made)} now? (YouTube, Instagram, Facebook)",
                      [[("✅ Post", f"a:{did}:ok"), ("✖️ Cancel", f"a:{did}:no")]])
        save_state(st)
        approved = wait_approval(st, did, [ask_msg])
        if not approved:
            d["status"] = "cancelled"
            save_state(st)
            say("✖️ Not posted.")
            log("Not approved")
            return
    d["status"] = "posting"
    save_state(st)
    for lang, (vid, cap, logf) in made.items():
        L = lang.upper()
        env = {"YOUTUBE_REFRESH_TOKEN": os.environ.get(f"YT_{L}_REFRESH", ""), "GITHUB_TOKEN": PAT}
        rel_v, rel_c = os.path.relpath(vid, "reels"), os.path.relpath(cap, "reels")
        yt = sh(["python", "upload_youtube.py", rel_v, rel_c, lang], logf, env)
        ig_file = f"alert_out_{lang}/reel_instagram.mp4"
        enc = sh(["ffmpeg", "-y", "-loglevel", "error", "-i", rel_v, "-vf",
                  "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30", "-c:v", "libx264",
                  "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p", "-b:v", "3500k", "-maxrate", "3500k",
                  "-bufsize", "7000k", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart",
                  "-t", "90", ig_file], logf)
        ig = enc and sh(["python", "upload_instagram.py", ig_file, rel_c, lang], logf)
        (done if (yt and ig) else failed).append(L + ("" if (yt and ig) else f" ({'YouTube ' if not yt else ''}{'Instagram/Facebook' if not ig else ''})".replace("  ", " ")))
        log(f"{L}: YouTube {'ok' if yt else 'failed'} · Instagram/Facebook {'ok' if ig else 'failed'}")
        if not (yt and ig):
            tg_file("sendDocument", "document", logf, chat_id=CHAT, caption=f"⚠️ {c['flag']} {L} posting problem — log")
    d["status"] = "posted" if done else "failed"
    save_state(st)
    say(f"{'✅' if done and not failed else '⚠️'} {c['flag']} Special edition: "
        + (f"posted {', '.join(done)}" if done else "nothing posted")
        + (f"; failed {', '.join(failed)}" if failed else "") + (f"; skipped {', '.join(skipped)} (already posted)" if skipped else "")
        + ".")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bot"
    try:
        run_post() if mode == "post" else run_bot()
    except Exception as e:
        log(f"Stopped: {type(e).__name__}")              # no details in the public log
        try:
            say(f"⚠️ Alerts bot stopped ({type(e).__name__}) — check the Actions run.")
        except Exception:
            pass
        sys.exit(1)
