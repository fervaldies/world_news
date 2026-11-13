"""
shared.py — files Newsdesk and the Telegram desk share, in the PRIVATE news-metrics repo (stage 4a).

    series/takes.json    the YOUR TAKE series list (Part 1s waiting for their Part 2)
    inbox/my_replies.md  your replies, so AI-written replies sound like you

Reads and writes go through the GitHub contents API with the file's sha, so two writers can't overwrite
each other (a conflict is retried with the newer file). A local copy is kept for when GitHub can't be
reached; a 20-second memory cache keeps Newsdesk's tabs fast.
"""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request

from .runner import DATA_DIR

VAULT = None                     # set by Newsdesk's app.py / the desk
CFG = None                       # a function returning the config dict (or the dict)
CACHE_DIR = os.path.join(DATA_DIR, "shared_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
TTL = 20
_mem = {}
_lock = threading.RLock()


class Offline(Exception):
    pass


def _cfg():
    c = CFG() if callable(CFG) else CFG
    return c or {}


def repo():
    c = _cfg()
    return c.get("metrics_repo", f"{c.get('github_owner', 'fervaldies')}/news-metrics")


def _token():
    t = (VAULT.get("GH_PAT", "") if VAULT else "") or os.environ.get("GH_PAT", "")
    if not t:
        raise Offline("GH_PAT is missing")
    return t


def _req(method, path, body=None, raw=False):
    url = f"https://api.github.com/repos/{repo()}/contents/{path}"
    data = json.dumps(body).encode() if body is not None else None
    h = {"Authorization": f"token {_token()}", "User-Agent": "newsdesk-shared",
         "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
    if data:
        h["Content-Type"] = "application/json"
    with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=h), timeout=30) as r:
        b = r.read()
    return b if raw else (json.loads(b) if b else {})


def _cache(path):
    return os.path.join(CACHE_DIR, path.replace("/", "__"))


def read_text(path, fresh=False):
    """(text, sha) from GitHub; (None, None) if the file doesn't exist. Raises Offline if GitHub can't be reached."""
    with _lock:
        hit = _mem.get(path)
        if hit and not fresh and time.time() - hit[2] < TTL:
            return hit[0], hit[1]
    try:
        d = _req("GET", path)
        if d.get("content"):
            text = base64.b64decode(d["content"]).decode("utf-8")
        else:                                            # files over 1 MB come without content
            text = _req("GET", path, raw=True).decode("utf-8")
        sha = d.get("sha")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None
        raise Offline(f"GitHub HTTP {e.code}")
    except Offline:
        raise
    except Exception as e:
        raise Offline(str(e))
    with _lock:
        _mem[path] = (text, sha, time.time())
    with open(_cache(path), "w", encoding="utf-8") as f:
        f.write(text)
    return text, sha


def cached_text(path):
    p = _cache(path)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


def write_text(path, text, sha, message):
    body = {"message": message, "content": base64.b64encode(text.encode("utf-8")).decode()}
    if sha:
        body["sha"] = sha
    try:
        d = _req("PUT", path, body)
    except urllib.error.HTTPError as e:
        if e.code in (409, 422):
            raise Conflict()
        raise Offline(f"GitHub HTTP {e.code}")
    except Exception as e:
        raise Offline(str(e))
    new_sha = (d.get("content") or {}).get("sha")
    with _lock:
        _mem[path] = (text, new_sha, time.time())
    with open(_cache(path), "w", encoding="utf-8") as f:
        f.write(text)
    return new_sha


class Conflict(Exception):
    pass


def modify_text(path, fn, message, tries=5):
    """fn(old_text_or_None) → new text. Retries with the newer file when someone else wrote in between."""
    for i in range(tries):
        old, sha = read_text(path, fresh=True)
        new = fn(old)
        if new is None or new == old:
            return old
        try:
            write_text(path, new, sha, message)
            return new
        except Conflict:
            time.sleep(1 + i)
    raise Offline("news-metrics kept changing — try again")


def modify_json(path, fn, default, message):
    def f(old):
        data = json.loads(old) if old else default
        new = fn(data)
        return json.dumps(new, ensure_ascii=False, indent=1)
    return json.loads(modify_text(path, f, message))
