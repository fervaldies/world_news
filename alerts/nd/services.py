"""
services.py — everything the new panels need that isn't a queued job:

  • GitHub helpers (news-metrics records, workflows, Actions minutes)
  • Today board (what was posted today, by GitHub or by Newsdesk)
  • Headline editor (read / save the day files of each public news repo)
  • Token health (Instagram, YouTube, GitHub, Telegram, Anthropic, Gemini)
  • Claude helpers (translate a headline, write a special edition)
  • Presets and settings (small JSON files in ~/.local/share/newsdesk)

Secrets are read from the vault and never returned to the browser.
"""

import os
import re
import json
import time
import glob
import math
import subprocess
import threading
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta

import yaml

try:
    from zoneinfo import ZoneInfo
    SYDNEY = ZoneInfo("Australia/Sydney")
except Exception:            # pragma: no cover
    SYDNEY = None

from .runner import DATA_DIR, HISTORY_PATH

PRESETS_PATH = os.path.join(DATA_DIR, "presets.json")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
PAUSED_PATH = os.path.join(DATA_DIR, "paused_workflows.json")
EDITS_DIR = os.path.join(DATA_DIR, "edits")
os.makedirs(EDITS_DIR, exist_ok=True)

GRAPH = "https://graph.facebook.com/v25.0"
IG_SCOPES = ["instagram_basic", "instagram_content_publish", "instagram_manage_comments",
             "instagram_manage_insights", "pages_show_list", "pages_read_engagement",
             "pages_manage_posts", "read_insights", "business_management"]
DEFAULT_AI_MODEL = "claude-haiku-4-5-20251001"


# ── small helpers ────────────────────────────────────────────────────────────
def utc_today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def http(method, url, headers=None, data=None, timeout=30):
    """Returns (status, headers, body-bytes). Never raises for HTTP errors."""
    body = None
    if data is not None:
        if isinstance(data, (bytes, bytearray)):
            body = bytes(data)
        elif isinstance(data, str):
            body = data.encode()
        else:
            body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        out = e.code, e.headers, e.read() if e.fp else b""
    if "generativelanguage.googleapis.com" in url and ":generateContent" in url:
        GeminiCounter.record(url, out[0])
    return out


def jbody(raw):
    try:
        return json.loads(raw.decode() or "null")
    except Exception:
        return None


def history_entries():
    if not os.path.exists(HISTORY_PATH):
        return []
    out = []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def parse_iso(ts):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def sydney_hhmm(utc_h, utc_m, day=None):
    base = datetime.now(timezone.utc) if day is None else datetime.fromisoformat(day + "T00:00:00+00:00")
    t = base.replace(hour=utc_h, minute=utc_m, second=0, microsecond=0)
    if SYDNEY is None:
        return None
    return t.astimezone(SYDNEY).strftime("%H:%M")


# ── GitHub ───────────────────────────────────────────────────────────────────
class GitHub:
    def __init__(self, vault, cfg_getter):
        self.vault = vault
        self.cfg = cfg_getter
        self._tree = (0, None)
        self._records = {}
        self._wf = (0, None)
        self._minutes = (0, None)
        self._login = None
        self.lock = threading.Lock()

    @property
    def owner(self):
        return self.cfg().get("github_owner", "fervaldies")

    @property
    def metrics_repo(self):
        return self.cfg().get("metrics_repo", f"{self.owner}/news-metrics")

    def call(self, method, path, accept="application/vnd.github+json", data=None, extra=None, timeout=30):
        token = self.vault.get("GH_PAT")
        if not token:
            raise Exception("GH_PAT is missing in secrets.env")
        h = {"Authorization": f"token {token}", "Accept": accept, "X-GitHub-Api-Version": "2022-11-28",
             "User-Agent": "newsdesk"}
        if data is not None:
            h["Content-Type"] = "application/json"
        h.update(extra or {})
        url = path if path.startswith("http") else "https://api.github.com" + path
        return http(method, url, h, data, timeout)

    def login(self):
        if not self._login:
            st, _, b = self.call("GET", "/user")
            if st != 200:
                raise Exception(f"GitHub /user HTTP {st}")
            self._login = (jbody(b) or {}).get("login")
        return self._login

    # news-metrics: list of every file path (cached for a minute)
    def metrics_paths(self, max_age=60):
        with self.lock:
            now = time.time()
            if self._tree[1] is not None and now - self._tree[0] < max_age:
                return self._tree[1]
            st, _, b = self.call("GET", f"/repos/{self.metrics_repo}/git/trees/HEAD?recursive=1")
            if st != 200:
                raise Exception(f"news-metrics tree HTTP {st}")
            paths = [t["path"] for t in (jbody(b) or {}).get("tree", []) if t.get("type") == "blob"]
            self._tree = (now, paths)
            return paths

    def metrics_file(self, path):
        if path in self._records:
            return self._records[path]
        st, _, b = self.call("GET", f"/repos/{self.metrics_repo}/contents/{urllib.parse.quote(path)}",
                             accept="application/vnd.github.raw+json")
        data = jbody(b) if st == 200 else None
        if data is not None:
            self._records[path] = data
        return data

    # raw pass-through for the Metrics tab (read-only, news-metrics only)
    def metrics_proxy(self, path, accept, etag=None, ref="main"):
        extra = {"If-None-Match": etag} if etag else {}
        q = f"?ref={urllib.parse.quote(ref)}" if ref else ""
        return self.call("GET", f"/repos/{self.metrics_repo}/contents/{urllib.parse.quote(path)}{q}",
                         accept=accept or "application/vnd.github+json", extra=extra, timeout=60)

    # ── workflows ────────────────────────────────────────────────────────────
    def repos(self):
        cfg = self.cfg()
        out = []
        for key, c in cfg["channels"].items():
            out.append({"repo": c["reels_repo"], "channel": key, "name": c["name"], "flag": c["flag"], "role": "reels"})
            out.append({"repo": c["news_repo"], "channel": key, "name": c["name"], "flag": c["flag"], "role": "news"})
        f = cfg.get("facts")
        if f:
            out.append({"repo": f["reels_repo"], "channel": "facts", "name": f["name"], "flag": f["flag"], "role": "facts"})
        return out

    def workflows(self, refresh=False):
        now = time.time()
        if not refresh and self._wf[1] is not None and now - self._wf[0] < 300:
            return self._wf[1]
        cfg = self.cfg()
        repos_dir = cfg["repos_dir"]

        def one(r):
            st, _, b = self.call("GET", f"/repos/{self.owner}/{r['repo']}/actions/workflows?per_page=100")
            if st != 200:
                return {**r, "error": f"HTTP {st}", "workflows": []}
            wfs = []
            for w in (jbody(b) or {}).get("workflows", []):
                fname = os.path.basename(w.get("path", ""))
                wfs.append({"id": w["id"], "name": w.get("name"), "file": fname, "state": w.get("state"),
                            "posting": bool(re.match(r"^(daily|weekly)_", fname)) and r["role"] in ("reels", "facts"),
                            "cron": read_cron(os.path.join(repos_dir, r["repo"], w.get("path", "")))})
            return {**r, "workflows": wfs}

        with ThreadPoolExecutor(8) as ex:
            data = list(ex.map(one, self.repos()))
        self._wf = (now, data)
        return data

    def set_workflow(self, repo, wf_id, enable):
        st, _, b = self.call("PUT", f"/repos/{self.owner}/{repo}/actions/workflows/{int(wf_id)}/{'enable' if enable else 'disable'}")
        if st not in (200, 204):
            raise Exception(f"GitHub said HTTP {st}: {(jbody(b) or {}).get('message', '')}")
        self._wf = (0, None)

    def pause_posting(self, channels=None):
        paused = read_json(PAUSED_PATH, [])
        done = []
        for r in self.workflows(refresh=True):
            if channels and r["channel"] not in channels:
                continue
            for w in r["workflows"]:
                if w["posting"] and w["state"] == "active":
                    self.set_workflow(r["repo"], w["id"], False)
                    paused.append({"repo": r["repo"], "id": w["id"], "name": w["name"]})
                    done.append(f"{r['repo']} · {w['file']}")
        write_json(PAUSED_PATH, paused)
        return done

    def resume_posting(self):
        paused = read_json(PAUSED_PATH, [])
        done, left = [], []
        for p in paused:
            try:
                self.set_workflow(p["repo"], p["id"], True)
                done.append(f"{p['repo']} · {p['name']}")
            except Exception:
                left.append(p)
        write_json(PAUSED_PATH, left)
        return done

    # ── Actions minutes this month ───────────────────────────────────────────
    def minutes(self, refresh=False):
        now = time.time()
        if not refresh and self._minutes[1] is not None and now - self._minutes[0] < 600:
            return self._minutes[1]
        included = int(self.cfg().get("actions_included_minutes", 3000))
        month = datetime.now(timezone.utc)
        result = None
        try:
            login = self.login()
            st, _, b = self.call("GET", f"/users/{login}/settings/billing/usage?year={month.year}&month={month.month}")
            items = (jbody(b) or {}).get("usageItems") if st == 200 else None
            if items is not None:
                used = 0.0
                for it in items:
                    if str(it.get("product", "")).lower() != "actions":
                        continue
                    if "minute" not in str(it.get("unitType", "")).lower():
                        continue
                    if it.get("grossAmount", 1) == 0:      # public repos are free
                        continue
                    used += float(it.get("quantity", 0))
                result = {"used": round(used), "included": included, "source": "GitHub billing"}
            if result is None:
                st, _, b = self.call("GET", f"/users/{login}/settings/billing/actions")
                if st == 200:
                    d = jbody(b) or {}
                    result = {"used": round(d.get("total_minutes_used", 0)),
                              "included": d.get("included_minutes") or included, "source": "GitHub billing"}
        except Exception:
            result = None
        if result is None:
            result = self._estimate_minutes(included)
        self._minutes = (now, result)
        return result

    def _estimate_minutes(self, included):
        start = datetime.now(timezone.utc).strftime("%Y-%m-01")
        repos = sorted({r["repo"] for r in self.repos() if r["role"] in ("reels", "facts")} |
                       {self.metrics_repo.split("/", 1)[1]})
        per_repo = {}

        def one(repo):
            total, page = 0, 1
            while page <= 10:
                st, _, b = self.call("GET", f"/repos/{self.owner}/{repo}/actions/runs?per_page=100&page={page}"
                                            f"&created=%3E%3D{start}")
                if st != 200:
                    return repo, None
                runs = (jbody(b) or {}).get("workflow_runs", [])
                for r in runs:
                    a, z = parse_iso(r.get("run_started_at")), parse_iso(r.get("updated_at"))
                    if a and z and r.get("status") == "completed":
                        total += max(1, math.ceil((z - a).total_seconds() / 60))
                if len(runs) < 100:
                    break
                page += 1
            return repo, total

        with ThreadPoolExecutor(6) as ex:
            for repo, mins in ex.map(one, repos):
                if mins is not None:
                    per_repo[repo] = mins
        if not per_repo:
            raise Exception("GitHub didn't share the Actions usage — check the GH_PAT in Tokens and keys.")
        return {"used": sum(per_repo.values()), "included": included, "per_repo": per_repo,
                "source": "estimate from run times (private repos Newsdesk knows)"}


def read_cron(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    m = re.search(r"cron:\s*[\"']([^\"']+)[\"']", text)
    if not m:
        return None
    parts = m.group(1).split()
    if len(parts) != 5:
        return {"expr": m.group(1)}
    out = {"expr": m.group(1)}
    if parts[0].isdigit() and parts[1].isdigit():
        h, mi = int(parts[1]), int(parts[0])
        out.update({"utc": f"{h:02d}:{mi:02d}", "sydney": sydney_hhmm(h, mi), "h": h, "m": mi})
    if parts[4] not in ("*", "?"):
        names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        out["weekday"] = ",".join(names[int(x) % 7] if x.isdigit() else x for x in parts[4].split(","))
    return out


# ── Today board ──────────────────────────────────────────────────────────────
REC = re.compile(r"^(\d{4}-\d{2}-\d{2})_([a-z0-9]+)_(en|es)_(youtube|instagram|facebook)_([a-z]+(?:_[a-z]+)*?)(_trial)?_(.+)\.json$")
PLATFORMS = ["youtube", "instagram", "facebook"]


def today_board(gh, cfg, day=None):
    day = day or utc_today()
    errors = []
    records = []
    try:
        paths = [p for p in gh.metrics_paths() if p.startswith("posts/") and os.path.basename(p).startswith(day + "_")]

        def load(p):
            m = REC.match(os.path.basename(p))
            if not m:
                return None
            d = gh.metrics_file(p) or {}
            return {"channel": m.group(2), "lang": m.group(3), "platform": m.group(4), "kind": m.group(5),
                    "trial": bool(m.group(6)), "post_id": m.group(7), "url": d.get("url"),
                    "posted_at": d.get("posted_at") or d.get("created_at"), "title": d.get("title")}

        with ThreadPoolExecutor(8) as ex:
            records = [r for r in ex.map(load, paths) if r]
    except Exception as e:
        errors.append(f"Couldn't read news-metrics: {e}")

    hist = [h for h in history_entries() if str(h.get("at", ""))[:10] == day]

    def source_of(rec):
        t = parse_iso(rec.get("posted_at"))
        for h in hist:
            if (h.get("channel") == rec["channel"] and h.get("lang") == rec["lang"] and h.get("kind") == rec["kind"]
                    and bool(h.get("trial")) == rec["trial"]):
                ht = parse_iso(h.get("at"))
                if t is None or ht is None or abs((ht - t).total_seconds()) < 45 * 60:
                    return "newsdesk"
        return "github"

    for r in records:
        r["source"] = source_of(r)

    wf_state = {}
    try:
        for repo in gh.workflows():
            for w in repo["workflows"]:
                wf_state[(repo["repo"], w["file"])] = w["state"]
    except Exception as e:
        errors.append(f"Couldn't read GitHub workflows: {e}")

    rows = []
    for key, c in cfg["channels"].items():
        if not c.get("daily", True):             # weekly-only channel (e.g. Crypto): no daily row
            continue
        for lang in ("en", "es", *c.get("extra_langs", [])):
            cells = {}
            for pl in PLATFORMS:
                hit = next((r for r in records if r["channel"] == key and r["lang"] == lang and r["platform"] == pl
                            and r["kind"] == "daily" and not r["trial"]), None)
                cells[pl] = hit
            if lang in c.get("trials", []):
                cells["trial"] = next((r for r in records if r["channel"] == key and r["lang"] == lang
                                       and r["kind"] == "daily" and r["trial"]), None) or {"pending": True}
            # Newsdesk posts that left no record (e.g. Crypto): trust the local history
            local = next((h for h in hist if h.get("channel") == key and h.get("lang") == lang
                          and h.get("kind") == "daily" and not h.get("trial")), None)
            if local:
                for pl in PLATFORMS:
                    if not cells[pl]:
                        cells[pl] = {"source": "newsdesk", "url": None, "posted_at": local.get("at"), "no_record": True}
            wf_file = f"daily_{lang}.yml"
            cron = read_cron(os.path.join(cfg["repos_dir"], c["reels_repo"], ".github", "workflows", wf_file))
            rows.append({"channel": key, "name": c["name"], "flag": c["flag"], "color": c["color"], "lang": lang,
                         "cells": cells, "github": {"cron": cron, "state": wf_state.get((c["reels_repo"], wf_file))}})

    others = {}
    for r in records:
        if r["kind"] == "daily":
            continue
        k = (r["channel"], r["lang"], r["kind"], r["trial"])
        others.setdefault(k, {"channel": r["channel"], "lang": r["lang"], "kind": r["kind"], "trial": r["trial"],
                              "source": r["source"], "posts": []})["posts"].append(
            {"platform": r["platform"], "url": r["url"], "posted_at": r["posted_at"]})
    for h in hist:
        if h.get("channel") == "facts":
            k = ("facts", h.get("lang"), h.get("kind"), bool(h.get("trial")))
            others.setdefault(k, {"channel": "facts", "lang": h.get("lang"), "kind": h.get("kind"),
                                  "trial": bool(h.get("trial")), "source": "newsdesk", "posts": []})
    return {"day": day, "rows": rows, "others": list(others.values()), "errors": errors,
            "counts": {"records": len(records)}}


# ── Headline editor ──────────────────────────────────────────────────────────
def _day_files(cfg, key, day):
    c = cfg["channels"][key]
    cwd = os.path.join(cfg["repos_dir"], c["news_repo"])
    dname = datetime.fromisoformat(day).strftime("%A")
    return cwd, dname, os.path.join(cwd, f"{dname}NewsEN.yml"), os.path.join(cwd, f"{dname}NewsES.yml")


def _titles(d):
    out = []
    for n in (d or {}).get("news", []) or []:
        out.append(n.get("title", "") if isinstance(n, dict) else str(n))
    return out


def edits_path(key, day):
    return os.path.join(EDITS_DIR, f"{key}_{day}.json")


def load_edits(key, day=None):
    return read_json(edits_path(key, day or utc_today()), None)


def get_headlines(cfg, key, pull=True):
    day = utc_today()
    cwd, dname, pen, pes = _day_files(cfg, key, day)
    msg = None
    if pull:
        r = subprocess.run(["git", "pull", "--rebase", "--autostash"], cwd=cwd, capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            msg = "Couldn't update the news repo — showing the local copy."
    en = yaml.safe_load(open(pen, encoding="utf-8")) if os.path.exists(pen) else None
    es = yaml.safe_load(open(pes, encoding="utf-8")) if os.path.exists(pes) else None
    if en is None and es is None:
        return {"ok": False, "message": f"No {dname}NewsEN.yml / {dname}NewsES.yml yet. Fetch today's news first."}
    file_date = str((en or es or {}).get("date", ""))
    return {"ok": True, "day": day, "weekday": dname, "file_date": file_date,
            "stale": bool(file_date) and not file_date.startswith(day),
            "news_en": _titles(en), "news_es": _titles(es),
            "question_en": (en or {}).get("question", ""), "question_es": (es or {}).get("question", ""),
            "hashtags_en": (en or {}).get("hashtags", []) or [], "hashtags_es": (es or {}).get("hashtags", []) or [],
            "edited": os.path.exists(edits_path(key, day)), "message": msg}


def save_headlines(cfg, key, payload):
    day = utc_today()
    cwd, dname, pen, pes = _day_files(cfg, key, day)
    rows = payload.get("rows") or []
    if not 1 <= len(rows) <= 5:
        raise ValueError("Keep between 1 and 5 stories.")
    for r in rows:
        if not str(r.get("en", "")).strip() or not str(r.get("es", "")).strip():
            raise ValueError("Every story needs an English and a Spanish headline.")

    def tags(v):
        if isinstance(v, str):
            v = [t for t in re.split(r"[\s,]+", v) if t]
        return [t if t.startswith("#") else "#" + t for t in v]

    for path, lang in ((pen, "en"), (pes, "es")):
        d = yaml.safe_load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        d = d or {}
        old = d.get("news", []) or []
        new = []
        for r in rows:
            src = r.get("src")
            text = str(r[lang]).strip()
            base = old[src] if isinstance(src, int) and 0 <= src < len(old) else None
            new.append({**base, "title": text} if isinstance(base, dict) else {"title": text})
        d["news"] = new
        if f"question_{lang}" in payload:
            d["question"] = str(payload[f"question_{lang}"]).strip()
        if f"hashtags_{lang}" in payload:
            d["hashtags"] = tags(payload[f"hashtags_{lang}"])
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False, width=1000)

    write_json(edits_path(key, day), {"date": day, "news_en": [str(r["en"]).strip() for r in rows],
                                       "news_es": [str(r["es"]).strip() for r in rows]})
    cmd = (f"git add {dname}NewsEN.yml {dname}NewsES.yml && "
           f"(git diff --staged --quiet || git commit -m 'Headlines edited {dname} (Newsdesk)') && "
           f"(git push || (git pull --rebase --autostash && git push))")
    r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return {"ok": True, "pushed": False,
                "message": "Saved for Newsdesk videos, but pushing to GitHub failed — GitHub's runs will use the old headlines."}
    return {"ok": True, "pushed": True, "message": "Saved and pushed. Newsdesk and GitHub will use these headlines."}


def apply_edits_to_temp(key, temp_path, log):
    """Called by the daily/trial jobs right after make_daily_reels.py."""
    edits = load_edits(key)
    if not edits:
        return False
    with open(temp_path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    changed = []
    for lang in ("en", "es"):
        k = f"news_{lang}"
        if isinstance(d.get(k), list):
            sample = d[k][0] if d[k] else None
            d[k] = [{**sample, "title": t} if isinstance(sample, dict) else t for t in edits[k]]
            changed.append(k)
    if not changed:
        raise Exception("Couldn't apply your edited headlines: weeks/daily_temp.yml has no news_en / news_es lists.")
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False, width=1000)
    log(f"Using your edited headlines ({', '.join(changed)}): {edits['news_en'][0]}")
    return True


# ── Claude ───────────────────────────────────────────────────────────────────
def claude(vault, system, user, model, max_tokens=1500, web=False):
    key = vault.get("ANTHROPIC_API_KEY")
    if not key:
        raise Exception("ANTHROPIC_API_KEY is missing in secrets.env")
    msgs = [{"role": "user", "content": user}]
    body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": msgs}
    if web:
        body["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    h = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    text = []
    for _ in range(5):
        st, _, b = http("POST", "https://api.anthropic.com/v1/messages", h, body, timeout=180)
        d = jbody(b) or {}
        if st != 200:
            raise Exception(f"Claude API HTTP {st}: {(d.get('error') or {}).get('message', '')}")
        text += [c.get("text", "") for c in d.get("content", []) if c.get("type") == "text"]
        if d.get("stop_reason") != "pause_turn":
            break
        msgs.append({"role": "assistant", "content": d.get("content", [])})
        text = []
    return "".join(text).strip()


def translate(vault, cfg, text):
    return claude(vault, "You translate news headlines from English into natural Spain Spanish. "
                         "Keep it under 20 words, keep names and numbers exact, no quotes, no prefixes. "
                         "Reply with the translation only.",
                  text, cfg.get("ai_model", DEFAULT_AI_MODEL), max_tokens=200)


SPECIAL_NAMES = {"australia": ("Australia News", "Noticias de Australia"),
                 "world": ("World News", "Noticias del Mundo"),
                 "europe": ("Europe News", "Noticias de Europa"),
                 "crypto": ("Crypto News", "Noticias Cripto")}
MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
             "octubre", "noviembre", "diciembre"]


def gemini_write(vault, cfg, system, user, key, topic):
    """Gemini writes the special edition: Google Search grounding when the key allows it,
    plus the latest GNews articles on the topic as background. Free on Gemini's free tier."""
    gkey = vault.get("GEMINI_API_KEY")
    if not gkey:
        raise Exception("GEMINI_API_KEY is missing in secrets.env")
    region = {"australia": " Australia", "europe": " Europe"}.get(key, "")
    arts = []
    if vault.get("GNEWS_API_KEY"):
        q = urllib.parse.urlencode({"q": f"{topic}{region}", "lang": "en", "max": 10, "sortby": "publishedAt",
                                    "apikey": vault.get("GNEWS_API_KEY")})
        st, _, b = http("GET", f"https://gnews.io/api/v4/search?{q}", timeout=30)
        for a in (jbody(b) or {}).get("articles", []) if st == 200 else []:
            arts.append(f"- {a.get('publishedAt', '')[:10]} · {(a.get('source') or {}).get('name', '')}: "
                        f"{a.get('title', '')} — {a.get('description', '')}")
    context = ("\n\nLatest articles found on this topic (use them, and search for more if you can):\n" + "\n".join(arts)) if arts else ""
    model = cfg.get("ai_model_gemini", "gemini-3.5-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={urllib.parse.quote(gkey)}"
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user + context}]}],
            "generationConfig": {"maxOutputTokens": 2500}}
    note = None
    st, _, b = http("POST", url, {"Content-Type": "application/json"}, {**body, "tools": [{"google_search": {}}]}, timeout=120)
    if st != 200:                                   # no search quota / not supported → write from the GNews articles only
        if not arts:
            raise Exception(f"Gemini couldn't search ({(jbody(b) or {}).get('error', {}).get('message', st)}) and GNews found nothing.")
        note = "Gemini couldn't use Google Search right now, so it wrote this from the latest GNews articles only — check the facts."
        st, _, b = http("POST", url, {"Content-Type": "application/json"}, body, timeout=120)
    d = jbody(b) or {}
    if st != 200:
        raise Exception(f"Gemini HTTP {st}: {(d.get('error') or {}).get('message', '')[:200]}")
    try:
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    except Exception:
        raise Exception("Gemini returned an empty answer. Try again or use Claude.")
    return text, note


def special_from_topic(vault, cfg, app_dir, key, topic, writer=None):
    c = cfg["channels"][key]
    en_name, es_name = SPECIAL_NAMES.get(key, (f"{c['name']} News", f"Noticias de {c['name']}"))
    now = datetime.now()
    date_en = f"{now:%B} {now.day}, {now.year}"
    date_es = f"{now.day} de {MONTHS_ES[now.month - 1]} de {now.year}"
    custom = os.path.join(app_dir, "config", "special_prompts", f"{key}.txt")
    if os.path.exists(custom):
        system = open(custom, encoding="utf-8").read()
    else:
        system = (
            f"You write special editions for the short-video news channel '{en_name}' / '{es_name}'. "
            "Search the web for the latest reporting on the topic and pick the 5 most important and recent "
            "developments, each from a different angle. Write each as a punchy, factual headline under 18 words, "
            "no source names, no prefixes, no emojis, no citations, no markup or tags of any kind. Translate each "
            "one into natural Spain Spanish (Spain vocabulary, e.g. 'ordenador', not 'computadora'; numbers like "
            "'6.300 millones de dólares'), also under 18 words, in the same order. The topic in the titles is "
            "3 words at most.\n\n"
            "Reply with YAML only — no code fences, no commentary — exactly in this shape:\n"
            f'title_en: "{en_name}\\nSpecial Edition\\n<Topic in 2-4 words>\\n{date_en}"\n'
            f'title_es: "{es_name}\\nEdición Especial\\n<Tema en 2-4 palabras>\\n{date_es}"\n'
            "news_en:\n  - \"...\"\n  (5 items)\nnews_es:\n  - \"...\"\n  (5 items)\n")
    user = f"Topic: {topic}\nToday's date: {date_en}."
    model = cfg.get("ai_model_special", cfg.get("ai_model", DEFAULT_AI_MODEL))
    note = None
    writer = writer or get_settings().get("special_writer", "gemini")
    if writer == "gemini":
        out, note = gemini_write(vault, cfg, system, user, key, topic)
    else:
        try:
            out = claude(vault, system, user, model, max_tokens=2500, web=True)
        except Exception as e:
            if "web" not in str(e).lower() and "tool" not in str(e).lower():
                raise
            note = "Web search isn't enabled for your Anthropic key, so Claude wrote this from what it already knows — check the facts."
            out = claude(vault, system, user, model, max_tokens=2000, web=False)
    out = re.sub(r"</?cite\b[^>]*>", "", out)            # web search adds citation tags; the video must not show them
    block = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", out, flags=re.S)
    if block:                                              # keep only the YAML if Claude added previews or notes
        out = block.group(1)
    out = re.sub(r"^```(?:yaml)?\s*|\s*```$", "", out.strip(), flags=re.M)
    i = out.find("title_en:")
    if i > 0:
        out = out[i:]
    d = yaml.safe_load(out) or {}
    missing = [k for k in ("title_en", "title_es", "news_en", "news_es") if k not in d]
    if missing:
        raise Exception(f"Claude's answer is missing {', '.join(missing)}. Try again or rephrase the topic.")
    out, n2 = add_extra_languages(vault, cfg, key, out.strip() + "\n", writer)
    if n2:
        note = f"{note} {n2}".strip() if note else n2
    return {"yaml": out, "note": note, "writer": writer}


# ── Token health ─────────────────────────────────────────────────────────────
class Health:
    def __init__(self, vault, gh, cfg_getter):
        self.vault, self.gh, self.cfg = vault, gh, cfg_getter
        self.result = None
        self.running = False

    def run_background(self):
        if not self.running:
            threading.Thread(target=self.check, daemon=True).start()

    def check(self):
        self.running = True
        try:
            items = []
            for fn in (self._instagram, self._reminder_file, self._instagram_1140, self._github, self._youtube, self._telegram, self._anthropic, self._gemini):
                try:
                    r = fn()
                    items += r if isinstance(r, list) else [r]
                except Exception as e:
                    names = {"_instagram": "Instagram / Facebook token", "_github": "GitHub token (GH_PAT)",
                             "_youtube": "YouTube tokens", "_telegram": "Telegram bot",
                             "_anthropic": "Anthropic (Claude) key", "_gemini": "Gemini key"}
                    items.append({"name": names.get(fn.__name__, fn.__name__), "level": "bad",
                                  "detail": f"Couldn't check: {getattr(e, 'reason', e)}"})
            bad = sum(1 for i in items if i["level"] == "bad")
            warn = sum(1 for i in items if i["level"] == "warn")
            self.result = {"at": datetime.now(timezone.utc).isoformat(), "items": items, "bad": bad, "warn": warn}
            return self.result
        finally:
            self.running = False

    def _instagram(self):
        tok = self.vault.get("INSTAGRAM_TOKEN")
        if not tok:
            return {"name": "Instagram / Facebook token", "level": "bad", "detail": "INSTAGRAM_TOKEN is missing"}
        q = urllib.parse.urlencode({"input_token": tok, "access_token": tok})
        st, _, b = http("GET", f"{GRAPH}/debug_token?{q}")
        d = (jbody(b) or {}).get("data", {})
        if st != 200 or not d.get("is_valid"):
            msg = ((jbody(b) or {}).get("error") or {}).get("message") or (d.get("error") or {}).get("message") or f"HTTP {st}"
            return {"name": "Instagram / Facebook token", "level": "bad", "detail": f"Not valid: {msg}. Renew it."}
        exp = d.get("expires_at") or 0                  # 0 = Meta says the token itself never expires
        data_exp = d.get("data_access_expires_at") or 0  # data access ends ~90 days after you last renewed
        missing = [s for s in IG_SCOPES if s not in (d.get("scopes") or [])]
        limits = [(t, why) for t, why in ((exp, "token expires"), (data_exp, "data access ends")) if t]
        if limits:
            when, why = min(limits)
            days = (when - time.time()) / 86400
            level = "bad" if days < 5 else "warn" if days < 12 else "ok"
            detail = f"{days:.0f} days left ({why} {datetime.fromtimestamp(when):%d %b %Y})"
            if not exp:
                detail += " · Meta reports the token itself as never expiring"
        else:
            level, detail, days = "ok", "Meta reports no expiry date for this token", None
        if missing:
            level = "warn" if level == "ok" else level
            detail += f" · missing permissions: {', '.join(missing)}"
        return {"name": "Instagram / Facebook token", "level": level, "detail": detail, "days": days}

    def _reminder_file(self):
        """The GitHub token reminder (reminder_state.json): last renewal and when it will start reminding."""
        rem = self.cfg().get("token_reminder_repo")
        if not rem:
            return []
        import base64
        repo = rem if "/" in rem else f"{self.cfg().get('github_owner', 'fervaldies')}/{rem}"
        st, _, b = self.gh.call("GET", f"/repos/{repo}/contents/reminder_state.json")
        if st != 200:
            return {"name": "Token reminder (GitHub)", "level": "warn", "detail": f"Couldn't read reminder_state.json in {repo} (HTTP {st})"}
        state = json.loads(base64.b64decode((jbody(b) or {}).get("content", "")).decode() or "{}")
        last = str(state.get("last_updated", ""))[:10]
        try:
            d0 = datetime.strptime(last, "%Y-%m-%d")
        except ValueError:
            return {"name": "Token reminder (GitHub)", "level": "warn", "detail": f"last_updated looks odd: {last!r}"}
        age = (datetime.now() - d0).days
        first, daily = d0 + timedelta(days=50), d0 + timedelta(days=55)
        if state.get("reminding"):
            level, extra = "warn", "reminding you now"
        elif age >= 50:
            level, extra = "warn", f"reminders due (daily from {daily:%d %b})"
        else:
            level, extra = "ok", f"first reminder {first:%d %b}, daily from {daily:%d %b}"
        return {"name": "Token reminder (GitHub)", "level": level,
                "detail": f"Last renewed {d0:%d %b %Y} ({age} day{'s' if age != 1 else ''} ago) · {extra}"}

    def _instagram_1140(self):
        tok = self.vault.get("INSTAGRAM_TOKEN_1140")
        if not tok:
            return []
        q = urllib.parse.urlencode({"input_token": tok, "access_token": tok})
        st, _, b = http("GET", f"{GRAPH}/debug_token?{q}")
        d = (jbody(b) or {}).get("data", {})
        if st != 200 or not d.get("is_valid"):
            return {"name": "Project 1140 token", "level": "bad", "detail": "Not valid any more — renew it in the wizard."}
        limits = [t for t in (d.get("expires_at") or 0, d.get("data_access_expires_at") or 0) if t]
        if not limits:
            return {"name": "Project 1140 token", "level": "ok", "detail": "No expiry date reported"}
        days = (min(limits) - time.time()) / 86400
        return {"name": "Project 1140 token", "level": "bad" if days < 5 else "warn" if days < 12 else "ok",
                "detail": f"{days:.0f} days left (expires {datetime.fromtimestamp(min(limits)):%d %b %Y})", "days": days}

    def _github(self):
        st, hd, b = self.gh.call("GET", "/user")
        if st != 200:
            return {"name": "GitHub token (GH_PAT)", "level": "bad", "detail": f"Rejected (HTTP {st}). Create a new classic token with 'repo'."}
        login = (jbody(b) or {}).get("login")
        scopes = [s.strip() for s in (hd.get("X-OAuth-Scopes") or "").split(",") if s.strip()]
        exp = hd.get("github-authentication-token-expiration")
        level, detail, days = "ok", f"Signed in as {login}", None
        if scopes and "repo" not in scopes:
            level, detail = "bad", detail + " · missing the 'repo' scope"
        if exp:
            try:
                when = datetime.strptime(exp.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                days = (when - datetime.now(timezone.utc)).total_seconds() / 86400
                detail += f" · {days:.0f} days left"
                if days < 5:
                    level = "bad"
                elif days < 14 and level == "ok":
                    level = "warn"
            except Exception:
                detail += f" · expires {exp}"
        else:
            detail += " · no expiry date"
        return {"name": "GitHub token (GH_PAT)", "level": level, "detail": detail, "days": days}

    def _youtube(self):
        cfg = self.cfg()
        want = {}
        for c in cfg["channels"].values():
            for lang, sec in c["youtube_secret"].items():
                want.setdefault(sec, []).append(f"{c['flag']} {c['name']} {lang.upper()}")
        if cfg.get("facts"):
            want.setdefault(cfg["facts"]["youtube_secret"], []).append(f"{cfg['facts']['flag']} Facts")
        cid, csec = self.vault.get("YOUTUBE_CLIENT_ID"), self.vault.get("YOUTUBE_CLIENT_SECRET")

        def one(item):
            sec, labels = item
            stems = {l.rsplit(" ", 1)[0] for l in labels}
            name = "YouTube · " + (f"{stems.pop()} EN + ES" if len(labels) == 2 and len(stems) == 1 else " / ".join(labels))
            rt = self.vault.get(sec)
            if not rt:
                return {"name": name, "level": "bad", "detail": f"{sec} is missing"}
            if not cid or not csec:
                return {"name": name, "level": "bad", "detail": "YOUTUBE_CLIENT_ID / SECRET missing"}
            data = urllib.parse.urlencode({"client_id": cid, "client_secret": csec, "refresh_token": rt,
                                           "grant_type": "refresh_token"}).encode()
            st, _, b = http("POST", "https://oauth2.googleapis.com/token",
                            {"Content-Type": "application/x-www-form-urlencoded"}, data)
            d = jbody(b) or {}
            if st != 200:
                return {"name": name, "level": "bad",
                        "detail": f"Refresh failed: {d.get('error_description') or d.get('error') or st}. Run get_youtube_tokens.py again."}
            st2, _, b2 = http("GET", "https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                              {"Authorization": f"Bearer {d['access_token']}"})
            items = (jbody(b2) or {}).get("items") or []
            title = items[0]["snippet"]["title"] if items else None
            return {"name": name, "level": "ok", "detail": f"Works · channel: {title}" if title else "Works"}

        with ThreadPoolExecutor(6) as ex:
            return list(ex.map(one, want.items()))

    def _telegram(self):
        bot = self.vault.get("TELEGRAM_BOT_TOKEN")
        if not bot:
            return {"name": "Telegram bot", "level": "bad", "detail": "TELEGRAM_BOT_TOKEN is missing"}
        st, _, b = http("GET", f"https://api.telegram.org/bot{bot}/getMe")
        d = (jbody(b) or {}).get("result") or {}
        if st != 200:
            return {"name": "Telegram bot", "level": "bad", "detail": f"Rejected (HTTP {st})"}
        chat = "" if self.vault.get("TELEGRAM_CHAT_ID") else " · TELEGRAM_CHAT_ID missing"
        return {"name": "Telegram bot", "level": "warn" if chat else "ok", "detail": f"@{d.get('username')}{chat}"}

    def _anthropic(self):
        key = self.vault.get("ANTHROPIC_API_KEY")
        if not key:
            return {"name": "Anthropic (Claude) key", "level": "warn", "detail": "Missing — weekly picks, translation and special editions won't work"}
        st, _, _ = http("GET", "https://api.anthropic.com/v1/models?limit=1",
                        {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return {"name": "Anthropic (Claude) key", "level": "ok" if st == 200 else "bad",
                "detail": "Works" if st == 200 else f"Rejected (HTTP {st})"}

    def _gemini(self):
        key = self.vault.get("GEMINI_API_KEY")
        if not key:
            return {"name": "Gemini key", "level": "warn", "detail": "Missing — fetching falls back to Claude"}
        st, _, _ = http("GET", f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1&key={urllib.parse.quote(key)}")
        return {"name": "Gemini key", "level": "ok" if st == 200 else "bad",
                "detail": "Works" if st == 200 else f"Rejected (HTTP {st})"}


# ── presets & settings ───────────────────────────────────────────────────────
def get_presets():
    return read_json(PRESETS_PATH, [])


def add_preset(p):
    name = str(p.get("name", "")).strip()[:60]
    if not name:
        raise ValueError("Give the preset a name.")
    sel = {"channels": [str(c) for c in p.get("channels", [])], "langs": [l for l in p.get("langs", []) if l in ("en", "es", "pt", "id")],
           "actions": [a for a in p.get("actions", []) if a in ("fetch", "daily", "trial", "weekly")]}
    if not sel["channels"] or not sel["actions"] or (not sel["langs"] and sel["actions"] != ["fetch"]):
        raise ValueError("Pick at least one channel, language and action.")
    presets = [x for x in get_presets() if x["name"] != name]
    presets.append({"id": f"p{int(time.time() * 1000)}", "name": name, **sel})
    write_json(PRESETS_PATH, presets)
    return presets


def delete_preset(pid):
    presets = [x for x in get_presets() if x["id"] != pid]
    write_json(PRESETS_PATH, presets)
    return presets


DEFAULT_SETTINGS = {"notify": True, "telegram_bot": False, "telegram_confirm": True, "telegram_bot_video": True,
                    "special_writer": "gemini", "cal_remind_time": "18:00", "cal_wake": True,
                    "auto_skip_youtube": True, "style_force": "", "charts_week_reminder": True}


def get_settings():
    return {**DEFAULT_SETTINGS, **read_json(SETTINGS_PATH, {})}


def set_settings(patch):
    s = get_settings()
    for k in DEFAULT_SETTINGS:
        if k in patch:
            s[k] = patch[k]
    write_json(SETTINGS_PATH, s)
    return s


# ── combined special edition (one YAML with a section per channel) ───────────
MULTI_ALIASES = {"australia": "australia", "aus": "australia", "au": "australia",
                 "world": "world", "mundo": "world", "europe": "europe", "europa": "europe", "eu": "europe",
                 "crypto": "crypto", "cripto": "crypto"}


def parse_multi(cfg, text, kind="special"):
    """Returns ({channel_key: section_dict}, [problems])."""
    data = yaml.safe_load(text or "") or {}
    if not isinstance(data, dict):
        raise ValueError("Paste the combined YAML (world:, europe:, australia: sections).")
    if "title_en" in data:
        raise ValueError("This is a single channel's YAML — use the button in that channel's row, or paste the combined file.")
    out, problems = {}, []
    for k, v in data.items():
        key = MULTI_ALIASES.get(str(k).strip().lower())
        if not key or key not in cfg["channels"]:
            problems.append(f"Unknown section '{k}' (ignored)")
            continue
        if not isinstance(v, dict):
            problems.append(f"{k}: not a section")
            continue
        need = ((POLL_FIELDS if isinstance(v, dict) and v.get("type") == "poll" else TAKE_FIELDS) if kind == "take"
                else ("title_en", "title_es", "intro_en", "intro_es", "facts_en", "facts_es") if kind == "explainer"
                else ("title_en", "title_es", "news_en", "news_es"))
        miss = [f for f in need if f not in v]
        if miss:
            problems.append(f"{cfg['channels'][key]['name']}: missing {', '.join(miss)}")
            continue
        out[key] = v
    return out, problems


# ── YOUR TAKE (context + 2–3 open questions) ─────────────────────────────────
TAKE_FIELDS = ("title_en", "title_es", "context_en", "context_es", "questions_en", "questions_es")
TAKE_NAMES = {"australia": ("Australia News", "Noticias de Australia"), "world": ("World News", "Noticias del Mundo"),
              "europe": ("Europe News", "Noticias de Europa")}


POLL_FIELDS = ("title_en", "title_es", "context_en", "context_es", "question_en", "question_es", "options_en", "options_es")


def check_take(d):
    d = d or {}
    if d.get("type") == "poll":
        miss = [f for f in POLL_FIELDS if f not in d]
        if miss:
            raise ValueError(f"The poll YAML is missing {', '.join(miss)}")
        oe, os_ = d["options_en"], d["options_es"]
        if not isinstance(oe, list) or not isinstance(os_, list) or not 2 <= len(oe) <= 3 or len(oe) != len(os_):
            raise ValueError("A poll needs 2 or 3 options, the same number in English and Spanish.")
        return d
    miss = [f for f in TAKE_FIELDS if f not in d]
    if miss:
        raise ValueError(f"The YAML is missing {', '.join(miss)}")
    qe, qs = d["questions_en"], d["questions_es"]
    if not isinstance(qe, list) or not isinstance(qs, list) or not 2 <= len(qe) <= 3 or len(qe) != len(qs):
        raise ValueError("Use 2 or 3 questions, the same number in English and Spanish.")
    return d


def take_title(d, lang, flag):
    lines = [l.strip() for l in str(d[f"title_{lang}"]).split("\n") if l.strip()
             and l.strip().lower() not in ("your take", "tu opinión", "tu opinion")]
    channel, topic = (lines[0] if lines else ""), (lines[1] if len(lines) >= 3 else (lines[-1] if lines else ""))
    t = (f"{topic}: what's your take? 🗣️ | {channel} {flag} #Shorts" if lang == "en"
         else f"{topic}: ¿tú qué opinas? 🗣️ | {channel} {flag} #Shorts")
    return t[:100]


def take_from_topic(vault, cfg, app_dir, key, topic, writer=None, fmt="questions"):
    en_name, es_name = TAKE_NAMES.get(key, (f"{cfg['channels'][key]['name']} News", f"Noticias de {cfg['channels'][key]['name']}"))
    now = datetime.now()
    date_en, date_es = f"{now:%B} {now.day}, {now.year}", f"{now.day} de {MONTHS_ES[now.month - 1]} de {now.year}"
    region = {"australia": "Australia and Australians", "europe": "Europe, the EU and European countries"}.get(key, "the world")
    custom = os.path.join(app_dir, "config", "take_prompts", f"{key}.txt")
    if fmt == "poll":
        custom = os.path.join(app_dir, "config", "take_prompts", f"{key}_poll.txt")
    poll_system = (
        f"You write 'Your Take' POLL videos for the short-video news channel '{en_name}' / '{es_name}', "
        f"which covers {region}. Search for the latest reporting on the topic. Then write:\n"
        "• context: ONE factual sentence (max 18 words) with the news behind the poll, from the last few days.\n"
        "• ONE poll question (max 14 words) people can answer by picking an option.\n"
        "• 3 short options (max 6 words each), clearly different, covering the main positions, none loaded.\n"
        "Rules: neutral and civil; never about politicians personally, religion or elections; no loaded, panic or "
        "conspiracy framing. Spanish from Spain, same order and meaning. No source names, citations, markup or emojis.\n\n"
        "Reply with YAML only — no code fences, no commentary — exactly:\n"
        "type: poll\n"
        f'title_en: "{en_name}\\nYour Take\\n<Topic in 2-3 words>\\n{date_en}"\n'
        f'title_es: "{es_name}\\nTu Opinión\\n<Tema en 2-3 palabras>\\n{date_es}"\n'
        'context_en: "..."\ncontext_es: "..."\nquestion_en: "..."\nquestion_es: "..."\n'
        'options_en:\n  - "..."\n  - "..."\n  - "..."\noptions_es:\n  - "..."\n  - "..."\n  - "..."\n')
    system = open(custom, encoding="utf-8").read() if os.path.exists(custom) else poll_system if fmt == "poll" else (
        f"You write 'Your Take' engagement videos for the short-video news channel '{en_name}' / '{es_name}', "
        f"which covers {region}. Search for the latest reporting on the topic. Then write:\n"
        "• context: ONE factual sentence (max 18 words) with the news behind the debate, from the last few days.\n"
        "• 3 open questions (max 12 words each) that invite people to comment. Mix: what happens next, how it "
        "affects people where they live, what they would do, whether something is worth it.\n"
        "Rules: neutral and civil; never 'who is right', never about politicians personally, religion or elections; "
        "no loaded, panic or conspiracy framing; easy to answer in a few words. Spanish from Spain (vocabulary, "
        "accents, number formats), same order, same meaning. No source names, citations, markup or emojis.\n\n"
        "Reply with YAML only — no code fences, no commentary — exactly:\n"
        f'title_en: "{en_name}\\nYour Take\\n<Topic in 2-3 words>\\n{date_en}"\n'
        f'title_es: "{es_name}\\nTu Opinión\\n<Tema en 2-3 palabras>\\n{date_es}"\n'
        'context_en: "..."\ncontext_es: "..."\nquestions_en:\n  - "..."\n  - "..."\n  - "..."\n'
        'questions_es:\n  - "..."\n  - "..."\n  - "..."\n')
    user = f"Topic: {topic}\nToday's date: {date_en}."
    writer = writer or get_settings().get("special_writer", "gemini")
    note = None
    if writer == "gemini":
        out, note = gemini_write(vault, cfg, system, user, key, topic)
    else:
        try:
            out = claude(vault, system, user, cfg.get("ai_model_special", cfg.get("ai_model", DEFAULT_AI_MODEL)), 1500, web=True)
        except Exception as e:
            if "web" not in str(e).lower() and "tool" not in str(e).lower():
                raise
            note = "Claude couldn't search the web, so it wrote this from what it already knows — check the facts."
            out = claude(vault, system, user, cfg.get("ai_model", DEFAULT_AI_MODEL), 1200, web=False)
    out = re.sub(r"</?cite\b[^>]*>", "", out)
    block = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", out, flags=re.S)
    out = block.group(1) if block else out
    i = out.find("type:") if fmt == "poll" and "type:" in out else out.find("title_en:")
    out = out[i:] if i > 0 else out
    if fmt == "poll" and not out.lstrip().startswith("type:"):
        out = "type: poll\n" + out
    check_take(yaml.safe_load(out))
    return {"yaml": out.strip() + "\n", "note": note, "writer": writer}


def take_ideas(vault, cfg, key):
    """Three debate-worthy topics from today's headlines of this channel (Gemini first — free)."""
    from .extras import ask_ai, extract_json
    c = cfg["channels"][key]
    day = datetime.now(timezone.utc).strftime("%A")
    path = os.path.join(cfg["repos_dir"], c["news_repo"], f"{day}NewsEN.yml")
    heads = _titles(yaml.safe_load(open(path, encoding="utf-8"))) if os.path.exists(path) else []
    if not heads:
        raise Exception("No headlines for today yet — fetch the news first.")
    prompt = ("These are today's top headlines of a news channel:\n" + "\n".join(f"- {h}" for h in heads) +
              "\n\nPick the 3 that would spark the most civil discussion in the comments (impact on people, what "
              "happens next, what would you do). Avoid tragedies with victims and stories about politicians personally. "
              'Return JSON only: {"ideas":[{"topic":"<2-3 word topic>","why":"<max 12 words>"}]}')
    return extract_json(ask_ai(vault, cfg, prompt, max_tokens=600)).get("ideas", [])[:3]


# ── secrets.env editing (token wizard) ───────────────────────────────────────
def write_secret(vault, key, value):
    """Replace one line in secrets.env (kept chmod 600); Newsdesk re-reads the file by itself."""
    import tempfile
    path = vault.path
    lines = open(path, encoding="utf-8").read().splitlines()
    done = False
    for i, l in enumerate(lines):
        if re.match(rf"^\s*{key}\s*=", l):
            lines[i], done = f"{key}={value}", True
    if not done:
        lines.append(f"{key}={value}")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".secrets.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


INBOX_SCOPES = ["pages_read_user_content", "pages_manage_engagement"]


def token_targets(cfg):
    """GitHub secrets that hold the Instagram token (plus any extra listed in channels.yml)."""
    owner = cfg.get("github_owner", "fervaldies")
    out = [(f"{owner}/{c['reels_repo']}", s) for c in cfg["channels"].values()
           for s in ["INSTAGRAM_EN_TOKEN", "INSTAGRAM_ES_TOKEN"] + [f"INSTAGRAM_{l.upper()}_TOKEN" for l in c.get("extra_langs", [])]]
    out += [(f"{owner}/{c['news_repo']}", f"INSTAGRAM_{l.upper()}_TOKEN")      # PT / ID workflows run in the public repos
            for c in cfg["channels"].values() for l in c.get("extra_langs", []) if c.get("news_repo")]
    if cfg.get("facts"):
        out.append((f"{owner}/{cfg['facts']['reels_repo']}", "INSTAGRAM_FACTS_TOKEN"))
    for e in cfg.get("token_secrets_extra", []) or []:
        for sec in e.get("secrets", []):
            out.append((e["repo"] if "/" in e["repo"] else f"{owner}/{e['repo']}", sec))
    return out


def token_profile(cfg, profile):
    """Which Meta app, secrets and GitHub targets a token belongs to."""
    owner = cfg.get("github_owner", "fervaldies")
    if profile == "p1140":
        repo = cfg.get("p1140_repo", "project-1140")
        return {"label": "Project 1140", "app_id": str(cfg.get("meta_app_id_1140", "2758367021182915")),
                "secret": "META_APP_SECRET_1140", "env": "INSTAGRAM_TOKEN_1140",
                "targets": [(f"{owner}/{repo}" if "/" not in repo else repo, x)
                            for x in cfg.get("p1140_secrets", ["INSTAGRAM_EN_TOKEN", "INSTAGRAM_ES_TOKEN"])],
                "required": cfg.get("p1140_scopes", ["instagram_basic", "instagram_content_publish"]),
                "optional": [], "reminder": None}
    return {"label": "News (aus-news)", "app_id": str(cfg.get("meta_app_id", "1957862705169242")),
            "secret": "META_APP_SECRET", "env": "INSTAGRAM_TOKEN", "targets": token_targets(cfg),
            "required": IG_SCOPES, "optional": INBOX_SCOPES, "reminder": cfg.get("token_reminder_repo")}


def renew_instagram_token(vault, cfg, gh, short_token, log=print, profile="news"):
    """Graph Explorer token → long-lived token → secrets.env + every GitHub secret + reminder_state.json."""
    import shutil
    report = []
    prof = token_profile(cfg, profile)
    app_id = prof["app_id"]
    secret = vault.get(prof["secret"])
    if not secret:
        raise Exception(f"Add {prof['secret']}=… to secrets.env first (the {prof['label']} Meta app → Settings → Basic → App secret).")
    q = urllib.parse.urlencode({"grant_type": "fb_exchange_token", "client_id": app_id, "client_secret": secret,
                                "fb_exchange_token": short_token.strip()})
    st, _, b = http("GET", f"{GRAPH}/oauth/access_token?{q}", timeout=30)
    d = jbody(b) or {}
    if st != 200 or not d.get("access_token"):
        raise Exception(f"Meta didn't accept the token: {(d.get('error') or {}).get('message', st)}")
    token = d["access_token"]
    q = urllib.parse.urlencode({"input_token": token, "access_token": token})
    st, _, b = http("GET", f"{GRAPH}/debug_token?{q}", timeout=30)
    info = (jbody(b) or {}).get("data", {})
    scopes = info.get("scopes") or []
    if str(info.get("app_id", app_id)) != app_id:
        raise Exception(f"This token was made with another Meta app (ID {info.get('app_id')}), not {prof['label']}. "
                        "Pick the right app in Graph Explorer.")
    missing = [x for x in prof["required"] if x not in scopes]
    if missing:
        raise Exception(f"The token is missing permissions: {', '.join(missing)}. Tick them in Graph Explorer and try again.")
    miss_inbox = [x for x in prof["optional"] if x not in scopes]
    write_secret(vault, prof["env"], token)
    report.append(f"✓ secrets.env (Newsdesk) updated ({prof['env']})")
    report.append("Permissions on the new token: " + ", ".join(sorted(scopes)))
    if miss_inbox:
        report.append(f"⚠️ Inbox on Facebook needs {', '.join(miss_inbox)} — add them next time")
    exe = shutil.which("gh")
    if not exe:
        report.append("✖ GitHub secrets not updated: the GitHub CLI (gh) isn't installed — sudo apt install gh")
    else:
        for repo, name in prof["targets"]:
            r = subprocess.run([exe, "secret", "set", name, "-R", repo], input=token, capture_output=True, text=True, timeout=60)
            report.append(f"{'✓' if r.returncode == 0 else '✖'} {repo} · {name}"
                          + ("" if r.returncode == 0 else f" — {(r.stderr or r.stdout).strip()[:120]}"))
            log(report[-1])
    rem = prof["reminder"]
    if rem:
        repo = rem if "/" in rem else f"{cfg.get('github_owner', 'fervaldies')}/{rem}"
        try:
            import base64
            st, _, b = gh.call("GET", f"/repos/{repo}/contents/reminder_state.json")
            d = jbody(b) or {}
            state = json.loads(base64.b64decode(d.get("content", "")).decode() or "{}")
            old = str(state.get("last_updated", ""))
            today = datetime.now()
            state["last_updated"] = today.isoformat(timespec="seconds") if "T" in old else today.strftime("%Y-%m-%d")
            state["reminding"] = False
            body = {"message": "Token renewed (Newsdesk)", "sha": d.get("sha"),
                    "content": base64.b64encode(json.dumps(state, indent=2).encode()).decode()}
            st, _, b = gh.call("PUT", f"/repos/{repo}/contents/reminder_state.json", data=body)
            report.append(f"{'✓' if st in (200, 201) else '✖'} {repo} · reminder_state.json"
                          + ("" if st in (200, 201) else f" (HTTP {st})"))
        except Exception as e:
            report.append(f"✖ reminder_state.json: {e}")
    elif profile == "news":
        report.append("ℹ️ reminder_state.json not updated: add token_reminder_repo: <repo name> to channels.yml")
    else:
        report.append("ℹ️ Newsdesk watches this token and warns you before it expires")
    days = (info.get("expires_at") or 0) and (info["expires_at"] - time.time()) / 86400
    return {"report": report, "days": round(days) if days else None}


# ── EXPLAINER (what you need to know) ────────────────────────────────────────
def check_explainer(d):
    d = d or {}
    miss = [f for f in ("title_en", "title_es", "intro_en", "intro_es", "facts_en", "facts_es") if not d.get(f)]
    if miss:
        raise ValueError(f"The explainer YAML is missing {', '.join(miss)}")
    fe, fs = d["facts_en"], d["facts_es"]
    if not isinstance(fe, list) or not 3 <= len(fe) <= 5 or len(fe) != len(fs):
        raise ValueError("Use 3 to 5 facts, the same number in English and Spanish.")
    return d


def explainer_from_topic(vault, cfg, app_dir, key, topic, writer=None):
    en_name, es_name = TAKE_NAMES.get(key, (f"{cfg['channels'][key]['name']} News", f"Noticias de {cfg['channels'][key]['name']}"))
    now = datetime.now()
    date_en, date_es = f"{now:%B} {now.day}, {now.year}", f"{now.day} de {MONTHS_ES[now.month - 1]} de {now.year}"
    region = {"australia": "Australia and Australians", "europe": "Europe and the EU"}.get(key, "the world")
    custom = os.path.join(app_dir, "config", "explainer_prompts", f"{key}.txt")
    system = open(custom, encoding="utf-8").read() if os.path.exists(custom) else (
        f"You write 'Explainer' videos (what you need to know) for the short-video news channel '{en_name}' / "
        f"'{es_name}', which covers {region}. Search for the latest reporting and the essential background on the "
        "topic. Then write 4 key facts (3–5 allowed) that let someone who missed the story understand it: what "
        "happened, why it matters, the key numbers, what happens next. Each fact max 16 words, plain and factual, "
        "with figures exactly as reported; hedge anything unconfirmed. No opinions, no source names, citations, "
        "markup or emojis. Spanish from Spain, natural, same order and meaning.\n\n"
        "Reply with YAML only — no code fences, no commentary — exactly:\n"
        f'title_en: "{en_name}\\n<Topic in 2-4 words>\\n{date_en}"\n'
        f'title_es: "{es_name}\\n<Tema en 2-4 palabras>\\n{date_es}"\n'
        'intro_en: "<Topic>: what you need to know."\nintro_es: "<Tema>: lo que debes saber."\n'
        'facts_en:\n  - "..."\n  - "..."\n  - "..."\n  - "..."\nfacts_es:\n  - "..."\n  - "..."\n  - "..."\n  - "..."\n')
    user = f"Topic: {topic}\nToday's date: {date_en}."
    writer = writer or get_settings().get("special_writer", "gemini")
    note = None
    if writer == "gemini":
        out, note = gemini_write(vault, cfg, system, user, key, topic)
    else:
        try:
            out = claude(vault, system, user, cfg.get("ai_model_special", cfg.get("ai_model", DEFAULT_AI_MODEL)), 1500, web=True)
        except Exception as e:
            if "web" not in str(e).lower() and "tool" not in str(e).lower():
                raise
            note = "Claude couldn't search the web, so it wrote this from what it already knows — check the facts."
            out = claude(vault, system, user, cfg.get("ai_model", DEFAULT_AI_MODEL), 1200, web=False)
    out = re.sub(r"</?cite\b[^>]*>", "", out)
    block = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", out, flags=re.S)
    out = block.group(1) if block else out
    i = out.find("title_en:")
    out = out[i:] if i > 0 else out
    check_explainer(yaml.safe_load(out))
    return {"yaml": out.strip() + "\n", "note": note, "writer": writer}



# ── YouTube uploads today (the daily upload limit per Google Cloud project) ──
class YouTubeCounter:
    """Counts uploads since midnight Pacific time (when YouTube resets the quota), across every channel."""
    def __init__(self, vault, cfg_fn):
        self.vault, self.cfg = vault, cfg_fn
        self.cache, self.t, self.playlists = None, 0, {}

    def secrets(self):
        cfg, out = self.cfg(), {}
        for k, c in cfg["channels"].items():
            for lang, sec in (c.get("youtube_secret") or {}).items():
                out.setdefault(sec, f"{c['flag']} {c['name']}" + ("" if len(set((c.get("youtube_secret") or {}).values())) == 1 else f" {lang.upper()}"))
        if cfg.get("facts") and cfg["facts"].get("youtube_secret"):
            out.setdefault(cfg["facts"]["youtube_secret"], f"{cfg['facts']['flag']} {cfg['facts']['name']}")
        return out

    @staticmethod
    def reset_times():
        from zoneinfo import ZoneInfo
        la = ZoneInfo("America/Los_Angeles")
        now = datetime.now(la)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1)

    def count(self, max_age=300, force=False):
        if self.cache and not force and time.time() - self.t < max_age:
            return self.cache
        from .inbox import yt_token, _yt_get
        start, nxt = self.reset_times()
        rows, total, errors = [], 0, []
        for sec, name in self.secrets().items():
            if not self.vault.get(sec):
                continue
            try:
                tok = yt_token(self.vault, sec)
                if sec not in self.playlists:
                    ch = _yt_get(tok, "channels", part="contentDetails", mine="true")["items"][0]
                    self.playlists[sec] = ch["contentDetails"]["relatedPlaylists"]["uploads"]
                items = _yt_get(tok, "playlistItems", part="snippet", playlistId=self.playlists[sec], maxResults=50).get("items", [])
                n = sum(1 for it in items if parse_iso(it["snippet"].get("publishedAt")) and parse_iso(it["snippet"]["publishedAt"]) >= start)
                rows.append({"channel": name, "uploads": n})
                total += n
            except Exception as e:
                errors.append(f"{name}: {str(e)[:120]}")
        limit = int(self.cfg().get("youtube_daily_limit", 100))
        self.cache = {"total": total, "limit": limit, "left": max(0, limit - total), "channels": rows, "errors": errors,
                      "resets_at": nxt.astimezone(timezone.utc).isoformat(),
                      "level": "bad" if total >= limit else "warn" if total >= 0.8 * limit else "ok",
                      "checked_at": datetime.now(timezone.utc).isoformat()}
        self.t = time.time()
        return self.cache

    def bump(self, n=1):
        """Count an upload we just made, without waiting for the next check."""
        if self.cache:
            self.cache["total"] += n
            self.cache["left"] = max(0, self.cache["limit"] - self.cache["total"])



# ── Gemini usage (Gemini has no "quota left" endpoint, so Newsdesk counts its own calls) ──
class GeminiCounter:
    """Counts Gemini requests made by Newsdesk per day (Pacific time, when the free daily limits reset),
    per model, and remembers 'limit reached' (HTTP 429) answers."""
    PATH = None
    _lock = threading.Lock()

    @staticmethod
    def day():
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")

    @classmethod
    def record(cls, url, status):
        try:
            from .runner import DATA_DIR
            path = cls.PATH or os.path.join(DATA_DIR, "gemini_usage.json")
            model = re.search(r"/models/([^:/?]+):generateContent", url)
            model = model.group(1) if model else "?"
            with cls._lock:
                data = read_json(path, {})
                d = data.setdefault(cls.day(), {"count": 0, "models": {}, "limited": []})
                d["count"] += 1
                d["models"][model] = d["models"].get(model, 0) + 1
                if status == 429:
                    d["limited"].append(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                for k in sorted(data)[:-14]:          # keep two weeks
                    data.pop(k, None)
                write_json(path, data)
        except Exception:
            pass

    @classmethod
    def summary(cls, cfg):
        from .runner import DATA_DIR
        from zoneinfo import ZoneInfo
        data = read_json(cls.PATH or os.path.join(DATA_DIR, "gemini_usage.json"), {})
        d = data.get(cls.day(), {"count": 0, "models": {}, "limited": []})
        limit = int(cfg.get("gemini_daily_limit", 1000))
        # GitHub's daily news fetch also uses Gemini (≈5 calls per news repo: pick, rewrite, ES, PT, ID, extras)
        github_est = 6 * len([c for c in cfg["channels"].values() if c.get("news_repo")])
        used = d["count"] + github_est
        la = ZoneInfo("America/Los_Angeles")
        nxt = datetime.now(la).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        level = "bad" if d["limited"] or used >= 0.95 * limit else "warn" if used >= 0.8 * limit else "ok"
        week = [{"day": k, "count": v.get("count", 0), "limited": len(v.get("limited", []))} for k, v in sorted(data.items())[-7:]]
        return {"newsdesk": d["count"], "github_estimate": github_est, "used": used, "limit": limit,
                "left": max(0, limit - used), "models": d["models"], "limited": d["limited"][-5:],
                "level": level, "resets_at": nxt.isoformat(), "week": week}



# ── Special editions in the extra languages (PT, ID) ──────────────────────────
EXTRA_SPECIAL = {
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


def ai_text(vault, cfg, prompt, writer=None, max_tokens=1500):
    """One plain prompt to Gemini (free) or Claude, following the writer choice."""
    writer = writer or get_settings().get("special_writer", "gemini")
    if writer == "gemini" and vault.get("GEMINI_API_KEY"):
        model = cfg.get("ai_model_gemini", "gemini-3.5-flash-lite")
        st, _, b = http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                                f"?key={urllib.parse.quote(vault.get('GEMINI_API_KEY'))}",
                        {"Content-Type": "application/json"},
                        {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens}},
                        timeout=90)
        if st == 200:
            d = jbody(b) or {}
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        if not vault.get("ANTHROPIC_API_KEY"):
            raise Exception(f"Gemini HTTP {st}")
    return claude(vault, "You translate news headlines. Reply with JSON only.", prompt,
                  cfg.get("ai_model", DEFAULT_AI_MODEL), max_tokens)


def add_extra_languages(vault, cfg, key, yml_text, writer=None):
    """Adds title_pt/news_pt and title_id/news_id (translated from English) for channels that have them."""
    langs = [l for l in cfg["channels"].get(key, {}).get("extra_langs", []) if l in EXTRA_SPECIAL]
    if not langs:
        return yml_text, None
    d = yaml.safe_load(yml_text) or {}
    lines = [l.strip() for l in str(d.get("title_en", "")).split("\n") if l.strip()]
    topic = lines[2] if len(lines) >= 4 else (lines[-2] if len(lines) >= 2 else "")
    news = [str(x) for x in d.get("news_en") or []]
    if not news:
        return yml_text, None
    now = datetime.now()
    added, failed = [], []
    for l in langs:
        x = EXTRA_SPECIAL[l]
        try:
            prompt = (f"Translate this news topic and these {len(news)} headlines into {x['style']}. Natural headline "
                      "style, not literal; keep every number, name, place and organisation exactly; same order; "
                      "each headline under 18 words; the topic in 3 words max. Return ONLY raw JSON, no markdown:\n"
                      '{"topic": "...", "news": ["...", "..."]}\n\n'
                      f"Topic: {topic}\nHeadlines:\n" + "\n".join(f"- {h}" for h in news))
            out = ai_text(vault, cfg, prompt, writer)
            j = json.loads(re.search(r"\{.*\}", out, re.S).group(0))
            tr = [str(h).strip() for h in j.get("news", []) if str(h).strip()]
            if len(tr) != len(news):
                raise ValueError(f"got {len(tr)} headlines")
            date = x["date"].format(d=now.day, m=x["months"][now.month - 1], y=now.year)
            d[f"title_{l}"] = f"{x['names'].get(key, key)}\n{x['label']}\n{j.get('topic') or topic}\n{date}"
            d[f"news_{l}"] = tr
            added.append(l.upper())
        except Exception as e:
            failed.append(f"{l.upper()} ({e})")
    if not added:
        return yml_text, ("Portuguese / Indonesian versions weren't written: " + "; ".join(failed)) if failed else None
    order = ["title_en", "title_es", "title_pt", "title_id", "news_en", "news_es", "news_pt", "news_id"]
    d = {**{k: d[k] for k in order if k in d}, **{k: v for k, v in d.items() if k not in order}}
    return tidy_yaml(d), (f"Not written in: {'; '.join(failed)}." if failed else None)



def tidy_yaml(d):
    """YAML the way people edit it: titles on one line with \\n, one quoted headline per line."""
    q = lambda t: '"' + str(t).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    out = []
    for k, v in d.items():
        if isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v):   # e.g. Charts of the Week's charts
            out.append(yaml.safe_dump({k: v}, allow_unicode=True, sort_keys=False, width=1000,
                                      default_flow_style=None).rstrip("\n"))
        elif isinstance(v, dict):
            out.append(yaml.safe_dump({k: v}, allow_unicode=True, sort_keys=False, width=1000,
                                      default_flow_style=None).rstrip("\n"))
        elif isinstance(v, list):
            out.append(f"{k}:")
            out += [f"  - {q(x)}" for x in v]
        elif isinstance(v, (int, float, bool)) or v is None:
            out.append(f"{k}: {json.dumps(v)}")
        else:
            out.append(f"{k}: {q(v)}")
    return "\n".join(out) + "\n"



# ── Daily style test and the bottom space of the new style ────────────────────
STYLE_CYCLE = ["new", "current-voice", "new", "current-silent"]


def style_schedule(days=10):
    from datetime import date, timedelta
    today = date.today()
    return [{"date": (today + timedelta(days=i)).isoformat(),
             "style": STYLE_CYCLE[(today + timedelta(days=i)).toordinal() % 4]} for i in range(days)]


LAYOUT_SIDES = ("top", "bottom", "left", "right")


def _look(cfg):
    """look.py from the first reels repo that has it (the repos own the layout rules), or None."""
    import importlib.util
    for c in cfg["channels"].values():
        p = os.path.join(cfg["repos_dir"], c["reels_repo"], "look.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("nd_look", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m, os.path.dirname(p)
    return None, None


def read_layout(cfg):
    """{'ready': bool, 'layout': {'new': {...}, 'current': {...}}, 'defaults': ..., 'limits': ...}"""
    look, repo = _look(cfg)
    if look:
        env = os.environ.pop("LAYOUT_OVERRIDE", None)
        try:
            lay = look.load_layout(os.path.join(repo, "layout.yml"))
        finally:
            if env is not None:
                os.environ["LAYOUT_OVERRIDE"] = env
        return {"ready": True, "layout": lay, "defaults": look.DEFAULTS, "limits": look.LIMITS}
    return {"ready": False, "layout": {"new": {"bottom": read_bottom_margin(cfg)}}, "defaults": {}, "limits": {}}


def clean_layout(cfg, lay):
    """Keep only known keys, inside the limits."""
    look, _ = _look(cfg)
    if not look:
        raise Exception("look.py isn't in the reels repos yet — install stage 1 first.")
    out = {}
    for style in ("new", "current"):
        src = (lay or {}).get(style) or {}
        out[style] = dict(look.DEFAULTS[style])
        for k in out[style]:
            if k in src and src[k] not in (None, ""):
                out[style][k] = look._clamp(k, src[k])
    return out


def layout_file_text(cfg, lay):
    look, _ = _look(cfg)
    return look.layout_yaml(clean_layout(cfg, lay))


def read_bottom_margin(cfg):
    for c in cfg["channels"].values():
        p = os.path.join(cfg["repos_dir"], c["reels_repo"], "layout.yml")
        if os.path.exists(p):
            try:
                d = yaml.safe_load(open(p, encoding="utf-8")) or {}
                return int((d.get("new") or {}).get("bottom") or d.get("bottom_margin", 280))
            except Exception:
                pass
    return 280


def background_file(cfg, key, n):
    """Path of background number n (1 = prefix.ext, 2 = prefix2.ext, …) in the channel's reels repo, or None."""
    c = cfg["channels"][key]
    spec = c.get("daily_video") or {}
    if not spec.get("prefix"):
        return None
    n = max(1, int(n or 1))
    name = f"{spec['prefix']}.{spec['ext']}" if n == 1 else f"{spec['prefix']}{n}.{spec['ext']}"
    p = os.path.join(cfg["repos_dir"], c["reels_repo"], name)
    return p if os.path.exists(p) else None


def layout_preview(cfg, key, lang, style, values, bg, out_png):
    """Still frame of `style` with these (unsaved) values, the free space shaded. Returns the log lines."""
    import sys as _sys
    c = cfg["channels"][key]
    cwd = os.path.join(cfg["repos_dir"], c["reels_repo"])
    yml = os.path.join(cwd, "weeks", "daily_temp.yml")
    lay = clean_layout(cfg, {style: values})
    env = {**os.environ, "LAYOUT_OVERRIDE": json.dumps({style: lay[style]})}
    cmd = [_sys.executable, "make_reels_es_en.py", "--yml", yml if os.path.exists(yml) else "none.yml",
           "--lang", lang, "--style", style, "--preview-layout", out_png]
    video = background_file(cfg, key, bg)
    if video:
        cmd += ["--video", video]
    if os.path.exists(out_png):
        os.remove(out_png)
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not os.path.exists(out_png):
        raise Exception((r.stderr or r.stdout).strip()[-400:] or "Preview failed")
    notes = [l for l in r.stdout.splitlines() if l.startswith("🎨")]
    return {"notes": notes, "background": os.path.basename(video) if video else None}


def cards_preview(cfg, key, lang, margin, out_png):
    """Older call (bottom space of the new style only)."""
    return layout_preview(cfg, key, lang, "new", {"bottom": margin}, 1, out_png)


# ── Location tags (Instagram) — locations.yml in the reels repos ────────────────
def read_locations(cfg):
    """Text of locations.yml from the first reels repo that has it (None if stage 3 isn't installed)."""
    for c in cfg["channels"].values():
        p = os.path.join(cfg["repos_dir"], c["reels_repo"], "locations.yml")
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    return None


def location_names(text):
    d = yaml.safe_load(text or "") or {}
    if not isinstance(d, dict):
        raise Exception("locations.yml must be a list of  \"Place\": \"ID\"  lines")
    out = {}
    for k, v in d.items():
        v = "" if v is None else str(v).strip()
        if v and not v.isdigit():
            raise Exception(f"'{k}': the ID must be digits only (got '{v}')")
        out[str(k).strip()] = v
    return out


def recent_places(cfg, have):
    """Places of this week's stories (as the upload scripts would tag them) that locations.yml doesn't list yet."""
    found = []
    for key, c in cfg["channels"].items():
        if key not in ("australia", "world", "europe"):
            continue
        for p in glob.glob(os.path.join(cfg["repos_dir"], c["news_repo"], "*NewsEN.yml")):
            try:
                items = (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("news") or []
            except Exception:
                continue
            for it in items:
                v = (it or {}).get("visual") or {} if isinstance(it, dict) else {}
                if v.get("type") != "map":
                    continue
                place, country = str(v.get("place") or "").strip(), str(v.get("country") or "").strip()
                name = (place if country in ("", "Australia") else None) if key == "australia" else \
                       country if key == "europe" else (place or country)
                if name and name.lower() not in {h.lower() for h in have} and name not in found:
                    found.append(name)
    return sorted(found)


def find_location_ids(cfg, token, text, limit=150):
    """For every place without an ID: Wikidata's Facebook Places ID first (free), then Meta's place search
    (stops asking Meta after its first refusal). Returns (new text, found, note or None)."""
    import importlib.util
    px = None
    for c in cfg["channels"].values():
        p = os.path.join(cfg["repos_dir"], c["reels_repo"], "post_extras.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("nd_px2", p)
            px = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(px)
            break
    if not px:
        raise Exception("post_extras.py isn't in the reels repos yet — install stage 3 first.")
    names = [k for k, v in location_names(text).items() if not v][:limit]
    found, notes, meta_ok, wd_errors = [], [], bool(token), 0
    for n in names:
        r = None
        if hasattr(px, "wikidata_location"):
            try:
                r = px.wikidata_location(n)
            except Exception as e:
                wd_errors += 1
                if wd_errors == 3:
                    notes.append(f"Wikidata didn't answer ({str(e)[:120]})")
        if not r and meta_ok:
            try:
                r = px.search_location(n, token)
            except urllib.error.HTTPError as e:
                meta_ok = False
                notes.append(f"Meta's place search refused (HTTP {e.code}) — it needs the 'Page Public Metadata "
                             "Access' feature, so only Wikidata was used for the rest.")
            except Exception:
                meta_ok = False
        if r:
            pid, title = r
            text = re.sub(rf'^(\s*"?{re.escape(n)}"?\s*:\s*)"?"?[^\n]*$', lambda m: f'{m.group(1)}"{pid}"   # {title}',
                          text, count=1, flags=re.M)
            found.append(f"{n} → {title}: {pid}")
    missing = len(names) - len(found)
    if missing:
        notes.append(f"{missing} place{'s' if missing != 1 else ''} still without an ID — paste those by hand "
                     "(Facebook page → About → Page transparency → Page ID), or leave them untagged.")
    return text, found, " ".join(notes) or None


# ── VIRAL CLAIM CHECK (fact check) ───────────────────────────────────────────
FACT_CHECKERS = {
    "australia": ["AAP FactCheck", "RMIT FactLab"],
    "world": ["AFP Fact Check", "Full Fact"],
    "europe": ["AFP Fact Check", "Full Fact"],
    "es": ["Maldita", "Newtral", "EFE Verifica"],
}
CLAIM_KEYS = ("title_en", "title_es", "claim_en", "claim_es", "verdict_en", "verdict_es", "points_en", "points_es",
              "checker", "url")


def check_claim(d):
    d = d or {}
    miss = [k for k in CLAIM_KEYS if not d.get(k)]
    if miss:
        raise ValueError(f"The fact-check YAML is missing {', '.join(miss)}")
    pe, ps = d["points_en"], d["points_es"]
    if not isinstance(pe, list) or not 2 <= len(pe) <= 3 or len(pe) != len(ps):
        raise ValueError("Use 2 or 3 points, the same number in English and Spanish.")
    if not str(d["url"]).startswith("http"):
        raise ValueError("url must be the link to the fact-checker's article")
    return d


def claim_from_topic(vault, cfg, key, topic="", writer=None):
    """Find (or check) a viral claim that a real fact-checker has already ruled on, and write the video text.
    Empty topic = find this week's most-shared claim for the channel's region."""
    en_name, es_name = TAKE_NAMES.get(key, (f"{cfg['channels'][key]['name']} News", f"Noticias de {cfg['channels'][key]['name']}"))
    now = datetime.now()
    date_en, date_es = f"{now:%B} {now.day}, {now.year}", f"{now.day} de {MONTHS_ES[now.month - 1]} de {now.year}"
    checkers = FACT_CHECKERS.get(key, FACT_CHECKERS["world"])
    region = {"australia": "Australia", "europe": "Europe"}.get(key, "the world")
    system = (
        f"You write 'Fact check' videos for the short-video news channel '{en_name}' / '{es_name}' ({region}). "
        "Search the web. Use ONLY a claim that a real, established fact-checker has ALREADY published a ruling on, "
        f"preferably {', '.join(checkers)} (for the Spanish side, if one of {', '.join(FACT_CHECKERS['es'])} ruled on "
        "the same claim, name it in checker_es). Never rule on a claim yourself and never invent a verdict, a "
        "checker or a link: copy the fact-checker's own rating (False, Misleading, Missing context, True…) and "
        "give the exact URL of their article. Describe the claim neutrally (\"Posts claim that…\"), without "
        "repeating insults or naming private people. Avoid claims about ongoing elections. Points: 2 or 3 short "
        "facts (max 16 words) from the fact-check that explain the verdict. Spanish from Spain, same meaning.\n\n"
        "Reply with YAML only — no code fences, no commentary — exactly:\n"
        f'title_en: "{en_name}\\n<Topic in 2-4 words>\\n{date_en}"\n'
        f'title_es: "{es_name}\\n<Tema en 2-4 palabras>\\n{date_es}"\n'
        'claim_en: "Posts claim that ..."\nclaim_es: "Circulan mensajes que dicen que ..."\n'
        'verdict_en: "False"\nverdict_es: "Falso"\n'
        'points_en:\n  - "..."\n  - "..."\npoints_es:\n  - "..."\n  - "..."\n'
        'checker: "<fact-checker name>"\nchecker_es: "<Spanish fact-checker if it also ruled, else same as checker>"\n'
        'url: "https://<the fact-check article>"\nurl_es: "https://<the Spanish article, if any>"\n')
    user = (f"Claim or topic to check: {topic}" if topic.strip() else
            f"Find the most shared viral claim of the last 7 days about {region} that one of these fact-checkers ruled "
            f"on: {', '.join(checkers)}.") + f"\nToday's date: {date_en}."
    writer = writer or get_settings().get("special_writer", "gemini")
    note = None
    if writer == "gemini":
        out, note = gemini_write(vault, cfg, system, user, key, topic.strip() or "fact check")
    else:
        try:
            out = claude(vault, system, user, cfg.get("ai_model_special", cfg.get("ai_model", DEFAULT_AI_MODEL)), 1500, web=True)
        except Exception as e:
            if "web" not in str(e).lower() and "tool" not in str(e).lower():
                raise
            raise Exception("Fact checks need web search, and it isn't enabled for your Anthropic key — use Gemini.")
    out = re.sub(r"</?cite\b[^>]*>", "", out)
    block = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", out, flags=re.S)
    out = block.group(1) if block else out
    i = out.find("title_en:")
    out = out[i:] if i > 0 else out
    d = check_claim(yaml.safe_load(out))
    note = ((note + " ") if note else "") + "Open the fact-check link and confirm the verdict before posting."
    return {"yaml": out.strip() + "\n", "note": note, "writer": writer, "url": d.get("url"), "url_es": d.get("url_es")}
