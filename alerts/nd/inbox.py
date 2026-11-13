"""
inbox.py — comments across Instagram, Facebook and YouTube.

  collect()      new comments you haven't replied to (last N days), saved as a "batch"
  copy_text()    the batch as plain text to paste into your Claude replies project
  parse_replies  reads Claude's answer (C1: "…", C2: SKIP …) back into replies
  post_reply()   posts one reply on the right platform/account
  style_export   your past replies (comment → your answer) for the project's knowledge
"""

import os
import re
import time
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import yaml

from . import services as S
from .runner import DATA_DIR

GRAPH = S.GRAPH
INBOX_PATH = os.path.join(DATA_DIR, "inbox.json")


def _g(path, token, **params):
    params["access_token"] = token
    st, _, b = S.http("GET", f"{GRAPH}/{path}?{urllib.parse.urlencode(params)}", timeout=40)
    d = S.jbody(b) or {}
    if st != 200:
        raise Exception((d.get("error") or {}).get("message") or f"HTTP {st}")
    return d


def accounts(vault, cfg):
    """Every account Newsdesk can read comments from (no secret values leave this function)."""
    out = []
    for key, c in cfg["channels"].items():
        for lang in ("en", "es", *c.get("extra_langs", [])):
            ig = vault.get(f"{key.upper()}_INSTAGRAM_{lang.upper()}_USER_ID")
            fb = vault.get(f"{key.upper()}_FACEBOOK_{lang.upper()}_PAGE_ID")
            base = {"channel": key, "name": c["name"], "flag": c["flag"], "color": c["color"], "lang": lang}
            if ig:
                out.append({**base, "platform": "instagram", "account_id": ig})
            if fb:
                out.append({**base, "platform": "facebook", "account_id": fb})
        secs = {}
        for lang, sec in c["youtube_secret"].items():
            secs.setdefault(sec, []).append(lang)
        for sec, langs in secs.items():
            if vault.get(sec):
                out.append({"channel": key, "name": c["name"], "flag": c["flag"], "color": c["color"],
                            "lang": "+".join(langs), "platform": "youtube", "secret": sec})
    f = cfg.get("facts")
    if f:
        base = {"channel": "facts", "name": f["name"], "flag": f["flag"], "color": f["color"], "lang": "en+es"}
        if vault.get(f.get("instagram_user_secret", "")):
            out.append({**base, "platform": "instagram", "account_id": vault.get(f["instagram_user_secret"])})
        if f.get("facebook_page_secret") and vault.get(f["facebook_page_secret"]):
            out.append({**base, "platform": "facebook", "account_id": vault.get(f["facebook_page_secret"])})
        if vault.get(f.get("youtube_secret", "")):
            out.append({**base, "platform": "youtube", "secret": f["youtube_secret"]})
    return out


# ── per-platform readers ─────────────────────────────────────────────────────
def _ig(vault, acct, since, want_mine=False, max_posts=15):
    tok = vault.get("INSTAGRAM_TOKEN")
    me = _g(acct["account_id"], tok, fields="username")["username"]
    media = _g(f"{acct['account_id']}/media", tok, fields="id,caption,permalink,timestamp,comments_count",
               limit=max_posts).get("data", [])
    items, pairs = [], []
    for m in media:
        t = S.parse_iso(m.get("timestamp"))
        if not m.get("comments_count") or (t and t < since):
            continue
        cs = _g(f"{m['id']}/comments", tok, fields="id,text,username,timestamp,replies{id,text,username,timestamp}",
                limit=50).get("data", [])
        video = (m.get("caption") or "").split("\n")[0][:140]
        for c in cs:
            if c.get("username") == me:
                continue
            mine = [r for r in (c.get("replies") or {}).get("data", []) if r.get("username") == me]
            if mine:
                pairs += [{"comment": c.get("text", ""), "reply": r.get("text", ""), "video": video} for r in mine]
                continue
            items.append({"platform": "instagram", "comment_id": c["id"], "author": c.get("username"),
                          "text": c.get("text", ""), "at": c.get("timestamp"), "video": video,
                          "link": m.get("permalink"), "account": f"@{me}"})
    return items, pairs


def _fb(vault, acct, since, want_mine=False, max_posts=15):
    page = acct["account_id"]
    page_tok = _g(page, vault.get("INSTAGRAM_TOKEN"), fields="access_token,name")
    tok, pname = page_tok["access_token"], page_tok.get("name", page)
    vids = _g(f"{page}/video_reels", tok, fields="id,description,created_time,permalink_url,comments.limit(0).summary(true)",
              limit=max_posts).get("data", [])
    items, pairs = [], []
    for v in vids:
        t = S.parse_iso(v.get("created_time"))
        n = ((v.get("comments") or {}).get("summary") or {}).get("total_count", 1)
        if not n or (t and t < since):
            continue
        cs = _g(f"{v['id']}/comments", tok, filter="toplevel", limit=50,
                fields="id,message,created_time,from,comments{from,message,created_time}").get("data", [])
        video = (v.get("description") or "").split("\n")[0][:140]
        for c in cs:
            if (c.get("from") or {}).get("id") == page:
                continue
            mine = [r for r in (c.get("comments") or {}).get("data", []) if (r.get("from") or {}).get("id") == page]
            if mine:
                pairs += [{"comment": c.get("message", ""), "reply": r.get("message", ""), "video": video} for r in mine]
                continue
            items.append({"platform": "facebook", "comment_id": c["id"], "author": (c.get("from") or {}).get("name", "someone"),
                          "text": c.get("message", ""), "at": c.get("created_time"), "video": video,
                          "link": "https://www.facebook.com" + v["permalink_url"] if str(v.get("permalink_url", "")).startswith("/") else v.get("permalink_url"),
                          "account": pname, "page": page})
    return items, pairs


def yt_token(vault, secret):
    data = urllib.parse.urlencode({"client_id": vault.get("YOUTUBE_CLIENT_ID"), "client_secret": vault.get("YOUTUBE_CLIENT_SECRET"),
                                   "refresh_token": vault.get(secret), "grant_type": "refresh_token"}).encode()
    st, _, b = S.http("POST", "https://oauth2.googleapis.com/token", {"Content-Type": "application/x-www-form-urlencoded"}, data)
    if st != 200:
        raise Exception(f"YouTube login failed ({(S.jbody(b) or {}).get('error', st)})")
    return S.jbody(b)["access_token"]


def _yt_get(tok, path, **params):
    st, _, b = S.http("GET", f"https://www.googleapis.com/youtube/v3/{path}?{urllib.parse.urlencode(params)}",
                      {"Authorization": f"Bearer {tok}"}, timeout=40)
    d = S.jbody(b) or {}
    if st != 200:
        raise Exception((d.get("error") or {}).get("message") or f"HTTP {st}")
    return d


def _yt(vault, acct, since, want_mine=False, max_posts=15):
    tok = yt_token(vault, acct["secret"])
    ch = _yt_get(tok, "channels", part="id,snippet,contentDetails", mine="true")["items"][0]
    cid, cname = ch["id"], ch["snippet"]["title"]
    uploads = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    vids = [i["contentDetails"]["videoId"] for i in
            _yt_get(tok, "playlistItems", part="contentDetails", playlistId=uploads, maxResults=max_posts).get("items", [])]
    if not vids:
        return [], []
    info = {v["id"]: v for v in _yt_get(tok, "videos", part="snippet,statistics", id=",".join(vids)).get("items", [])}
    items, pairs = [], []
    for vid in vids:
        v = info.get(vid) or {}
        t = S.parse_iso((v.get("snippet") or {}).get("publishedAt"))
        if int((v.get("statistics") or {}).get("commentCount", 0)) == 0 or (t and t < since):
            continue
        try:
            threads = _yt_get(tok, "commentThreads", part="snippet,replies", videoId=vid, maxResults=50, order="time").get("items", [])
        except Exception:
            continue                                   # comments turned off on this video
        title = (v.get("snippet") or {}).get("title", "")[:140]
        for th in threads:
            top = th["snippet"]["topLevelComment"]["snippet"]
            if (top.get("authorChannelId") or {}).get("value") == cid:
                continue
            mine = [r["snippet"] for r in (th.get("replies") or {}).get("comments", [])
                    if (r["snippet"].get("authorChannelId") or {}).get("value") == cid]
            if mine:
                pairs += [{"comment": top.get("textOriginal", ""), "reply": r.get("textOriginal", ""), "video": title} for r in mine]
                continue
            if not th["snippet"].get("canReply", True):
                continue
            items.append({"platform": "youtube", "comment_id": th["id"], "author": top.get("authorDisplayName", "someone"),
                          "text": top.get("textOriginal", ""), "at": top.get("publishedAt"), "video": title,
                          "link": f"https://www.youtube.com/watch?v={vid}&lc={th['id']}", "account": cname,
                          "secret": acct["secret"]})
    return items, pairs


READERS = {"instagram": _ig, "facebook": _fb, "youtube": _yt}


def collect(vault, cfg, days=14):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    errors, items = [], []

    def one(a):
        try:
            got, _ = READERS[a["platform"]](vault, a, since)
            for it in got:
                it.update(channel=a["channel"], name=a["name"], flag=a["flag"], color=a["color"], lang=a["lang"])
            return got, None
        except Exception as e:
            hint = ""
            if a["platform"] == "facebook" and "permission" in str(e).lower():
                hint = " (add pages_read_user_content and pages_manage_engagement to the token at the next renewal)"
            return [], f"{a['flag']} {a['name']} {a['lang'].upper()} {a['platform'].title()}: {e}{hint}"

    with ThreadPoolExecutor(6) as ex:
        for got, err in ex.map(one, accounts(vault, cfg)):
            items += got
            if err:
                errors.append(err)
    items.sort(key=lambda i: i.get("at") or "", reverse=True)
    batch = datetime.now().strftime("%Y%m%d-%H%M")
    for n, it in enumerate(items, 1):
        it["ref"] = f"C{n}"
        it["status"] = "new"
    data = {"batch": batch, "collected_at": datetime.now(timezone.utc).isoformat(), "items": items, "errors": errors}
    S.write_json(INBOX_PATH, data)
    return data


def load():
    return S.read_json(INBOX_PATH, {"batch": None, "items": [], "errors": []})


def public(data):
    """What the browser sees (no secret names)."""
    return {**data, "items": [{k: v for k, v in i.items() if k not in ("secret", "page")} for i in data.get("items", [])]}


def ago(iso):
    t = S.parse_iso(iso)
    if not t:
        return ""
    h = (datetime.now(timezone.utc) - t).total_seconds() / 3600
    return f"{h * 60:.0f} min ago" if h < 1 else f"{h:.0f} h ago" if h < 48 else f"{h / 24:.0f} days ago"


def copy_text(data, refs=None):
    items = [i for i in data["items"] if i["status"] == "new" and (not refs or i["ref"] in refs)]
    lines = [f"NEWSDESK COMMENTS · batch {data['batch']} · {len(items)} comment{'s' if len(items) != 1 else ''}", ""]
    for i in items:
        lang = {"en": "English", "es": "Spanish", "pt": "Portuguese (Brazil)", "id": "Indonesian",
                "en+es": "English or Spanish"}.get(i["lang"], i["lang"].upper())
        lines += [f"[{i['ref']}] {i['platform'].title()} · {i['account']} · {i['name']} ({lang} account)",
                  f"Video: {i['video'] or '(no caption)'}",
                  f"From: {i['author']} · {ago(i['at'])}",
                  f"Comment: {i['text']}", ""]
    lines.append("Reply with a yaml block: batch, then replies with one line per comment (C1: \"…\" or C1: SKIP).")
    return "\n".join(lines)


def parse_replies(data, text):
    text = re.sub(r"^```(?:ya?ml)?\s*|\s*```\s*$", "", text.strip(), flags=re.M)
    d = yaml.safe_load(text) or {}
    if not isinstance(d, dict):
        raise ValueError("Paste the yaml block from your replies project.")
    reps = d.get("replies", d)
    warn = []
    if d.get("batch") and str(d["batch"]) != str(data.get("batch")):
        warn.append(f"These replies are for batch {d['batch']}, but the latest list is {data.get('batch')}. "
                    "Check they still match before posting.")
    by = {i["ref"]: i for i in data["items"]}
    out = []
    for ref, reply in reps.items():
        ref = str(ref).strip().upper()
        if ref == "BATCH":
            continue
        it = by.get(ref)
        if not it:
            warn.append(f"{ref}: not in the list (ignored)")
            continue
        skip = reply is None or str(reply).strip().upper() in ("SKIP", "-", "")
        out.append({"ref": ref, "reply": "" if skip else str(reply).strip(), "skip": skip})
    return out, warn


def post_reply(vault, item, text):
    p = item["platform"]
    if p == "instagram":
        st, _, b = S.http("POST", f"{GRAPH}/{item['comment_id']}/replies?" + urllib.parse.urlencode(
            {"message": text, "access_token": vault.get("INSTAGRAM_TOKEN")}), timeout=40)
    elif p == "facebook":
        tok = _g(item["page"], vault.get("INSTAGRAM_TOKEN"), fields="access_token")["access_token"]
        st, _, b = S.http("POST", f"{GRAPH}/{item['comment_id']}/comments?" + urllib.parse.urlencode(
            {"message": text, "access_token": tok}), timeout=40)
    else:
        tok = yt_token(vault, item["secret"])
        st, _, b = S.http("POST", "https://www.googleapis.com/youtube/v3/comments?part=snippet",
                          {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                          {"snippet": {"parentId": item["comment_id"], "textOriginal": text}}, timeout=40)
    if st not in (200, 201):
        d = S.jbody(b) or {}
        raise Exception((d.get("error") or {}).get("message") or f"HTTP {st}")


def style_export(vault, cfg, days=365):
    """Your past replies as a markdown file for the Claude project's knowledge."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    blocks, errors = {}, []

    def one(a):
        try:
            _, pairs = READERS[a["platform"]](vault, a, since, want_mine=True, max_posts=50)
            return a, pairs, None
        except Exception as e:
            return a, [], f"{a['name']} {a['lang'].upper()} {a['platform']}: {e}"

    with ThreadPoolExecutor(6) as ex:
        for a, pairs, err in ex.map(one, accounts(vault, cfg)):
            if err:
                errors.append(err)
            if pairs:
                blocks.setdefault(f"{a['name']} · {a['lang'].upper()} · {a['platform'].title()}", []).extend(pairs)
    total = sum(len(v) for v in blocks.values())
    out = [f"# My past replies ({total})", "",
           "Real replies I wrote to comments on my news accounts. Use them to match my tone, length, emojis and "
           "phrasing in each language.", ""]
    for title, pairs in blocks.items():
        out += [f"## {title}", ""]
        for p in pairs:
            out += [f"- Video: {p['video']}", f"  Comment: {p['comment']}", f"  My reply: {p['reply']}", ""]
    if errors:
        out += ["<!-- couldn't read: " + " | ".join(errors) + " -->"]
    return "\n".join(out), total, errors


STYLE_PATH = os.path.join(DATA_DIR, "my_replies.md")
REPLY_RULES = """You write replies to comments on Fer's bilingual short-video news accounts (Australia, World, Europe,
Crypto "in 5 News" / "en 5 Noticias", and Best Free Tours Sydney), in Fer's own voice.
Copy the style of FER'S REAL REPLIES below: length, greetings, emojis, tone in each language.
Rules: reply in the comment's language (Spanish from Spain, Portuguese from Brazil, natural Indonesian); 5-25 words, max 2 sentences; never two identical
replies; don't invent facts, dates or figures; no political opinions; no financial, medical or legal advice;
no links or hashtags; don't promise videos. Write SKIP for spam, scams, bots, links, insults, hate, trolling,
emoji-only comments, or anything you're unsure about."""


SHARED_STYLE = "inbox/my_replies.md"        # stage 4a: shared with the Telegram desk (news-metrics, private)


def style_text(vault, cfg, max_chars=9000):
    from . import shared as SH
    try:
        text, _ = SH.read_text(SHARED_STYLE)
    except SH.Offline:
        text = None
    if text:
        with open(STYLE_PATH, "w", encoding="utf-8") as f:
            f.write(text)
    if not os.path.exists(STYLE_PATH):
        text, _, _ = style_export(vault, cfg)
        with open(STYLE_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        share_style(text)
    text = open(STYLE_PATH, encoding="utf-8").read()
    return text[:max_chars]


def share_style(text):
    """Puts your replies file in news-metrics (Newsdesk's 'Export my past replies' calls this)."""
    from . import shared as SH
    try:
        SH.modify_text(SHARED_STYLE, lambda old: text, "My replies (from Newsdesk)")
    except SH.Offline as e:
        print(f"⚠️ my_replies.md not shared ({e})")


def add_my_reply(item, reply):
    """A reply sent from Telegram goes into your replies file, so the next suggestions learn from it."""
    from . import shared as SH
    block = (f"- Video: {str(item.get('video') or item.get('post') or '')[:120]}\n  Comment: {item.get('text', '')}\n"
             f"  My reply: {reply}\n")

    def f(old):
        old = old or "# My past replies\n\n"
        head = "## Sent from Telegram"
        if head not in old:
            old = old.rstrip("\n") + f"\n\n{head}\n\n"
        return old.rstrip("\n") + "\n" + block
    try:
        SH.modify_text(SHARED_STYLE, f, "My replies: one more from Telegram")
    except SH.Offline:
        pass


RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "reply_instructions.txt")


def reply_rules():
    """Your own instructions (config/reply_instructions.txt) if the file exists, otherwise the built-in ones."""
    if os.path.exists(RULES_PATH):
        text = open(RULES_PATH, encoding="utf-8").read().strip()
        if text:
            return text
    return REPLY_RULES


def draft_with_ai(vault, cfg, data, refs=None, free_only=True):
    from .extras import ask_ai, extract_json
    items = [i for i in data["items"] if i["status"] == "new" and (not refs or i["ref"] in refs)]
    if not items:
        raise ValueError("No comments selected.")
    out = {}
    for start in range(0, len(items), 25):                      # 25 comments per request
        chunk = items[start:start + 25]
        prompt = (reply_rules() + "\n\nFER'S REAL REPLIES (my_replies.md):\n" + style_text(vault, cfg) + "\n\nCOMMENTS:\n" +
                  copy_text({**data, "items": chunk}).rsplit("\n", 1)[0] +
                  '\n\nReturn JSON only: {"replies": {"C1": "reply text or SKIP", ...}} with every reference above.')
        d = extract_json(ask_ai(vault, cfg, prompt, max_tokens=2500, free_only=free_only))
        out.update({str(k).upper(): v for k, v in (d.get("replies") or {}).items()})
    lines = [f"batch: {data['batch']}", "replies:"]
    for i in items:
        r = str(out.get(i["ref"], "SKIP")).strip().replace("\n", " ")
        lines.append(f"  {i['ref']}: SKIP" if r.upper() == "SKIP" or not r else f"  {i['ref']}: " + json.dumps(r, ensure_ascii=False))
    return "\n".join(lines)
