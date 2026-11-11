"""
trial_check.py — weekly: switch on Instagram trial reels for accounts that have enough followers.

Reads the follower counts the metrics already collect (news-metrics/index/latest.json), and for each
reels repo (Australia / World / Europe) adds every language whose Instagram account has reached
TRIAL_MIN_FOLLOWERS (default 1000) to the repository variable TRIAL_LANGS (e.g. "es" → "en,es").
Trials are only ever switched ON here — to switch one off, edit TRIAL_LANGS in the repo's
Settings → Secrets and variables → Actions → Variables. Tells Fer on Telegram when something changes.
GitHub workflows post EN/ES trials; PT/ID trials are posted from Newsdesk (it reads the same variable).

Env: GH_PAT, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRIAL_MIN_FOLLOWERS (optional), DRY_RUN=true (optional),
TELL_ALWAYS=1 (optional: send the Telegram summary even when nothing changed).
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

OWNER = "fervaldies"
REPOS = {"australia": "australia-news-reels", "world": "world-news-reels", "europe": "europe-news-reels"}
DEFAULT = {"australia": "es", "world": "es", "europe": ""}      # what the workflows assume when the variable isn't set
LANGS = ["en", "es", "pt", "id"]
NAMES = {"australia": "🇦🇺 Australia", "world": "🌍 World", "europe": "🇪🇺 Europe"}
MIN = int(os.environ.get("TRIAL_MIN_FOLLOWERS") or 1000)
DRY = (os.environ.get("DRY_RUN") or "").lower() == "true"
PAT = os.environ["GH_PAT"]


def gh(method, path, body=None, raw=False):
    req = urllib.request.Request(f"https://api.github.com{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"token {PAT}", "User-Agent": "trial-check",
                                          "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return data if raw else (json.loads(data) if data else {})


def get_var(repo):
    try:
        return gh("GET", f"/repos/{OWNER}/{repo}/actions/variables/TRIAL_LANGS")["value"], True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, False
        raise


def set_var(repo, value, exists):
    if exists:
        gh("PATCH", f"/repos/{OWNER}/{repo}/actions/variables/TRIAL_LANGS", {"name": "TRIAL_LANGS", "value": value})
    else:
        gh("POST", f"/repos/{OWNER}/{repo}/actions/variables", {"name": "TRIAL_LANGS", "value": value})


def telegram(text):
    bot, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (bot and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{bot}/sendMessage", data=data, timeout=20)
    except Exception as e:
        print(f"⚠️ Telegram: {e}")


def main():
    latest = json.loads(gh("GET", f"/repos/{OWNER}/news-metrics/contents/index/latest.json", raw=True))
    accounts = latest.get("accounts") or {}
    print(f"Follower counts from {latest.get('date')} · threshold {MIN:,}" + (" · DRY RUN" if DRY else ""))
    switched, near, status = [], [], []
    for ch, repo in REPOS.items():
        value, exists = get_var(repo)
        current = [x for x in (value if value is not None else DEFAULT[ch]).replace(" ", "").split(",") if x]
        new = list(current)
        for lang in LANGS:
            f = (accounts.get(f"instagram:{ch}:{lang}") or {}).get("followers")
            if f is None:
                continue
            if lang in current:
                continue
            if f >= MIN:
                new.append(lang)
                switched.append(f"{NAMES[ch]} {lang.upper()} ({f:,} followers)")
            else:
                near.append((f / MIN, f"{NAMES[ch]} {lang.upper()}: {f:,}"))
        new = [l for l in LANGS if l in new]
        status.append(f"{NAMES[ch]}: {', '.join(l.upper() for l in new) or 'none'}")
        if new != current or (not exists and new):
            print(f"{repo}: TRIAL_LANGS {','.join(current) or '(none)'} → {','.join(new)}")
            if not DRY:
                set_var(repo, ",".join(new), exists)
        else:
            print(f"{repo}: TRIAL_LANGS {','.join(current) or '(none)'} (no change)")
    near.sort(reverse=True)
    close = "; ".join(t for _, t in near[:3])
    if switched:
        msg = ("🧪 Trial reels switched ON for: " + ", ".join(switched) +
               f".\nThey reached {MIN:,} followers. GitHub posts the EN/ES trials from the next daily; "
               "PT/ID trials run from Newsdesk.\nTrials now: " + " · ".join(status) +
               (f"\nClosest next: {close} (need {MIN:,})" if close else ""))
        print(msg)
        if not DRY:
            telegram(msg)
    else:
        msg = f"🧪 Trial reels: no new account reached {MIN:,} followers. Trials now: " + " · ".join(status) + \
              (f"\nClosest next: {close}" if close else "")
        print(msg)
        if os.environ.get("TELL_ALWAYS") == "1" and not DRY:
            telegram(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Trial check failed: {e}")
        sys.exit(1)
