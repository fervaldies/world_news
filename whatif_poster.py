#!/usr/bin/env python3
"""What If scheduled Instagram posts (runs every hour in world_news/.github/workflows/whatif_poster.yml).

Newsdesk queues each scheduled Instagram post as a DRAFT release in the private repo QUEUE_REPO (body = manifest
JSON, assets = video + caption + meta + cover). This script:
    list → writes due=<n> to $GITHUB_OUTPUT (no download, a few seconds)
    post → for every item whose time has come: downloads it, posts it with world-news-reels' upload_instagram.py
           (the same uploader as every other post), writes done/<tag>.json to QUEUE_REPO, deletes the release
           (the video is really gone) and tells Telegram. A failure is retried next hour, 3 tries at most.
Never prints tokens or raw API answers.
"""
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

API = os.environ.get("ND_GITHUB_API", "https://api.github.com")
TG = os.environ.get("ND_TELEGRAM_API", "https://api.telegram.org")
REPO = os.environ.get("QUEUE_REPO", "fervaldies/whatif-queue")
TOKEN = os.environ.get("GH_PAT", "")
UPLOADER = os.environ.get("UPLOADER_DIR", "wnr")          # sparse checkout of world-news-reels
MAX_TRIES = 3
TOO_LATE_H = 12                                           # more than this late (workflow was off): don't post, tell Fer
FLAGS = {"en": "🇬🇧", "es": "🇪🇸"}


def call(method, url, data=None, ctype="application/json", accept="application/vnd.github+json", timeout=120):
    if isinstance(data, (dict, list)):
        data = json.dumps(data).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")   # not sent on to the download redirect
    req.add_header("Accept", accept)
    if data is not None:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            b = r.read()
            return r.status, (json.loads(b) if b and accept.endswith("json") else b)
    except urllib.error.HTTPError as e:
        try:
            msg = (json.loads(e.read() or b"{}") or {}).get("message", "")
        except Exception:
            msg = ""
        return e.code, {"message": str(msg)[:150]}


def telegram(text):
    bot, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not bot or not chat:
        return
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text, "disable_web_page_preview": "true"}).encode()
        urllib.request.urlopen(f"{TG}/bot{bot}/sendMessage", data=data, timeout=20).read()
    except Exception:
        print("(Telegram message not sent)")


def queue():
    st, rels = call("GET", f"{API}/repos/{REPO}/releases?per_page=100")
    if st == 404:
        return []
    if st != 200 or not isinstance(rels, list):
        sys.exit(f"❌ can't read the queue ({st} {rels.get('message', '') if isinstance(rels, dict) else ''})")
    items = []
    for r in rels:
        if not r.get("draft") or not str(r.get("name", "")).startswith("ig-"):
            continue
        try:
            m = json.loads(r.get("body") or "{}")
            when = datetime.fromisoformat(m["when"].replace("Z", "+00:00"))
        except Exception:
            print(f"⚠️ {r.get('name')}: unreadable manifest, skipped")
            continue
        items.append((when, r, m))
    return sorted(items, key=lambda x: x[0])


def due_items(now=None):
    now = now or datetime.now(timezone.utc)
    return [x for x in queue() if x[0] <= now + timedelta(minutes=2)]


def write_done(tag, res):
    body = {"message": f"done {tag}", "content": base64.b64encode(json.dumps(res, ensure_ascii=False).encode()).decode()}
    st, old = call("GET", f"{API}/repos/{REPO}/contents/done/{tag}.json")
    if st == 200 and isinstance(old, dict) and old.get("sha"):
        body["sha"] = old["sha"]
    st, b = call("PUT", f"{API}/repos/{REPO}/contents/done/{tag}.json", body)
    if st not in (200, 201):
        print(f"⚠️ couldn't write done/{tag}.json ({st})")


def already_done(tag):
    st, _ = call("GET", f"{API}/repos/{REPO}/contents/done/{tag}.json")
    return st == 200


def save_attempts(r, m):
    call("PATCH", f"{API}/repos/{REPO}/releases/{r['id']}", {"body": json.dumps(m, ensure_ascii=False)})


def download(r, folder):
    names = {}
    for a in r.get("assets") or []:
        st, data = call("GET", f"{API}/repos/{REPO}/releases/assets/{a['id']}", accept="application/octet-stream",
                        timeout=600)
        if st != 200:
            raise RuntimeError(f"download of {a['name']} failed ({st})")
        path = os.path.join(folder, a["name"])
        with open(path, "wb") as f:
            f.write(data)
        names[a["name"]] = path
    return names


def post_one(when, r, m):
    tag, lang, key = r["name"], m["lang"], m.get("channel", "whatif")
    title = m.get("title", "")
    late_h = (datetime.now(timezone.utc) - when).total_seconds() / 3600
    if already_done(tag):                                  # posted before but the release wasn't deleted
        call("DELETE", f"{API}/repos/{REPO}/releases/{r['id']}")
        print(f"⏭️ {tag}: already posted")
        return
    if late_h > TOO_LATE_H:
        write_done(tag, {"status": "failed", "error": f"{late_h:.0f} h late (the hourly check didn't run)", "tag": tag})
        call("DELETE", f"{API}/repos/{REPO}/releases/{r['id']}")
        telegram(f"⚠️ What If {FLAGS.get(lang, '')} Instagram NOT posted ({late_h:.0f} h late): {title}\n"
                 "Post it from Newsdesk (What If → Post now) if you still want it.")
        return
    user_id = os.environ.get(f"{key.upper()}_INSTAGRAM_{lang.upper()}_USER_ID", "")
    if not user_id:
        raise RuntimeError(f"secret {key.upper()}_INSTAGRAM_{lang.upper()}_USER_ID is missing in world_news")
    with tempfile.TemporaryDirectory() as td:
        files = download(r, td)
        video, cap = files.get(m["video"]), files.get(m["caption"])
        if not video or not cap:
            raise RuntimeError("video or caption missing in the queue item")
        env = {k: v for k, v in os.environ.items() if not k.endswith("_USER_ID") and k not in ("GH_PAT",)}
        env.update({f"INSTAGRAM_{lang.upper()}_TOKEN": os.environ.get("INSTAGRAM_TOKEN", ""),
                    f"INSTAGRAM_{lang.upper()}_USER_ID": user_id,
                    "GITHUB_TOKEN": TOKEN, "GITHUB_REPOSITORY": "fervaldies/world-news-reels",
                    "IG_STORY": "1" if m.get("story") else "0"})
        if m.get("trial"):
            env["IG_TRIAL_GRADUATION"] = "SS_PERFORMANCE"
        cmd = [sys.executable, os.path.join(os.path.abspath(UPLOADER), "upload_instagram.py"), video, cap, lang] + \
            (["--trial"] if m.get("trial") else [])
        p = subprocess.run(cmd, cwd=os.path.abspath(UPLOADER), env=env, capture_output=True, text=True, timeout=1500)
        out = p.stdout + "\n" + p.stderr
        # world_news is public: only short status lines in the log (no captions, no API answers)
        print("\n".join(l for l in out.splitlines()[-40:] if l.startswith(("✅", "❌", "⚠️", "🆔", "📸", "⏳", "🖼️"))
                        and "access_token" not in l and "Bearer" not in l))
        ids = re.findall(r"🆔 Instagram media id:\s*(\S+)", out)
        if p.returncode != 0 or not ids:
            raise RuntimeError(f"upload_instagram.py ended with code {p.returncode}"
                               + ("" if ids else " and no media id"))
    write_done(tag, {"status": "ok", "media_id": ids[-1], "tag": tag,
                     "posted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    call("DELETE", f"{API}/repos/{REPO}/releases/{r['id']}")
    telegram(f"📸 What If {FLAGS.get(lang, '')} posted on Instagram"
             f"{' (trial reel)' if m.get('trial') else ''}"
             f"{f' ({late_h * 60:.0f} min after the planned time)' if late_h > 0.1 else ''}: {title}")
    print(f"✅ {tag}: media {ids[-1]}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    if not TOKEN:
        sys.exit("❌ GH_PAT is missing")
    if mode == "list":
        items = queue()
        due = due_items()
        print(f"Queue: {len(items)} item(s), {len(due)} due now")
        for when, r, m in items[:10]:                     # world_news is public: no titles in the log
            print(f"  {when:%d %b %H:%M} UTC · {m.get('lang', '?').upper()} · {m.get('target')}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"due={len(due)}\n")
        return
    failed = 0
    for when, r, m in due_items():
        try:
            post_one(when, r, m)
        except Exception as e:
            failed += 1
            m["attempts"] = int(m.get("attempts") or 0) + 1
            err = str(e)[:200]
            print(f"❌ {r['name']}: {err} (try {m['attempts']} of {MAX_TRIES})")
            if m["attempts"] >= MAX_TRIES:
                write_done(r["name"], {"status": "failed", "error": err, "tag": r["name"]})
                call("DELETE", f"{API}/repos/{REPO}/releases/{r['id']}")
                telegram(f"❌ What If {FLAGS.get(m.get('lang'), '')} Instagram failed 3 times: {m.get('title', '')}\n{err}\n"
                         "Post it from Newsdesk (What If → Post now).")
            else:
                save_attempts(r, m)
                telegram(f"⚠️ What If Instagram not posted yet ({err}): {m.get('title', '')}. Trying again next hour.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
