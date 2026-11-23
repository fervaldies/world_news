"""
desk_trials.py — 🧪 Trials becomes a trial-video maker (change 3, 23 Nov 2026).

🧪 Trials → pick the accounts that can post trial reels (TRIAL_LANGS of each reels repo, i.e. the accounts with enough
followers) and how many trials per account (1–5) → type the topic → the special-edition text is written (Gemini,
Claude on request) → 🎬 Make → the videos come to approve → posted ONLY as Instagram trial reels that are never
promoted (graduation MANUAL: they stay with non-followers; nothing on YouTube / Facebook, no Story).
Each trial looks like a special edition but in its own colour (never red), with its own background and music.
The bot checks the views every hour for 48 h and tells Telegram when one goes viral:
views ≥ VIRAL_X (3) × the account's usual views (median of its last 30 reels) and ≥ VIRAL_MIN (300), with the topic
and the settings (colour, background, music) so you know which one worked. At 48 h: a short ranking.
Button codes: tr:<account>|tr:n<k>|tr:status (start), trm:make|trm:claude (draft). ⚡ Auto Post All: au:trials.
Public-log rule: only short status lines (no titles) are printed.
"""
import json
import os
import random
import statistics
import urllib.parse
import urllib.request
import uuid
from datetime import timedelta

A = None                                       # alerts module
D = None                                       # desk module
TINTS = [("blue", "#1F5FC8"), ("teal", "#007C80"), ("purple", "#7A3CC2"), ("green", "#1E8A46"),
         ("orange", "#C25A0E"), ("magenta", "#B02878"), ("navy", "#23358F"), ("olive", "#6B7A12")]
VIRAL_X = float(os.environ.get("TRIAL_VIRAL_X") or 3)
VIRAL_MIN = int(os.environ.get("TRIAL_VIRAL_MIN") or 300)
WATCH_H = 48
AUTO_N = int(os.environ.get("AUTO_TRIALS_N") or 3)
GRAPH = "https://graph.facebook.com/v21.0"
_USED = {"bg": set(), "music": set()}


def accounts():
    """[(channel, lang)] that may post trial reels: TRIAL_LANGS repo variable of each reels repo."""
    out = []
    for ck, c in A.CHANNELS.items():
        default = "es" if ck == "world" else ""                  # world-news-reels' workflow default
        try:
            v = (A.gh("GET", f"/repos/{A.OWNER}/{c['repo']}/actions/variables/TRIAL_LANGS") or {}).get("value", "")
        except Exception:
            v = default
        out += [(ck, l.strip()) for l in str(v or "").split(",") if l.strip() in ("en", "es")]
    return out


def label(ck, lang):
    return f"{A.CHANNELS[ck]['flag']} {A.CHANNELS[ck]['name']} {lang.upper()}"


def begin(st):
    acc = accounts()
    if not acc:
        D.menu("🧪 No account can post trial reels yet (TRIAL_LANGS is empty: they need "
               "enough followers first).\nTrial batches made in Newsdesk: send /trials")
        return None
    f = D.start_flow(st, "trials", step="accounts", acc=[list(a) for a in acc], pick=[f"{c}:{l}" for c, l in acc])
    m = D.ask_buttons("🧪 Trials — which accounts, and how many trials each?\n(Instagram trial reels only, never "
                      "promoted; same look as a special edition in other colours, backgrounds and music)", rows(f))
    f["msg"] = (m or {}).get("message_id")
    return None


def rows(f):
    r = [[(("✅ " if f"{c}:{l}" in f["pick"] else "⬜ ") + label(c, l), f"tr:{c}:{l}")] for c, l in f["acc"]]
    r.append([(f"{n}×", f"tr:n{n}") for n in range(1, 6)])
    r.append([("📋 Trial batches status", "tr:status")])
    r.append(D.AUTO.button("trials") if hasattr(D, "AUTO") else [])
    r.append([("✖ Cancel", "x:cancel")])
    return [x for x in r if x]


def tap(st, parts, msg):
    if parts[1] == "status":
        A.drop_buttons(msg)
        st.pop("desk", None)
        return D.trials_status()
    f = D.flow(st)
    if not f or f.get("flow") != "trials":
        A.drop_buttons(msg)
        A.say("⌛ That choice has expired — start again with 🧪 Trials.")
        return None
    if parts[1].startswith("n") and parts[1][1:].isdigit():
        if not f["pick"]:
            A.tg("sendMessage", chat_id=A.CHAT, text="Tick at least one account.")
            return None
        f["n"], f["step"] = max(1, min(5, int(parts[1][1:]))), "topic"
        A.drop_buttons(msg)
        A.say(f"🧪 {f['n']} trial{'s' if f['n'] > 1 else ''} for: {', '.join(label(*k.split(':')) for k in f['pick'])}\n"
              "✍️ Type the topic (a few words).")
        return None
    if len(parts) == 3:
        k = f"{parts[1]}:{parts[2]}"
        f["pick"] = [x for x in [f"{c}:{l}" for c, l in f["acc"]] if (x in f["pick"]) != (x == k)]
        D.edit_buttons(msg.get("message_id"), rows(f))
    return None


def got_topic(st, f, topic, prefer="gemini"):
    f["topic"], f["step"] = topic, "drafted"
    chans = sorted({k.split(":")[0] for k in f["pick"]})
    A.say(f"✍️ Writing{' with Claude' if prefer == 'claude' else ''}… (about a minute"
          f"{' per channel' if len(chans) > 1 else ''})")
    f["drafts"] = {}
    for ck in chans:
        try:
            y, who = A.write_special(ck, topic, topic, prefer)
            f["drafts"][ck] = y
            A.say(A.preview({"channel": ck, "yaml": y, "writer": "Claude" if who == "claude" else "Gemini"}))
        except Exception as e:
            A.log(f"trial draft failed ({type(e).__name__})")
            A.say(f"⚠️ {A.CHANNELS[ck]['flag']} couldn't write it ({type(e).__name__}).")
    if not f["drafts"]:
        f["step"] = "topic"
        A.say("Type another topic, or ✖ Cancel.")
        return None
    n = f["n"] * len([k for k in f["pick"] if k.split(":")[0] in f["drafts"]])
    D.ask_buttons("Check the facts, then:", [[(f"🎬 Make the {n} trial{'s' if n > 1 else ''}", "trm:make")],
                                             [("✍️ Rewrite with Claude", "trm:claude"), ("✖ Cancel", "x:cancel")]])
    return None


def make_tasks(drafts, pick, n):
    bid = uuid.uuid4().hex[:6]
    tasks = []
    for key in pick:
        ck, lang = key.split(":")
        y = drafts.get(ck)
        if not y or not y.get(f"news_{lang}"):
            continue
        for slot, (tname, tint) in enumerate(random.sample(TINTS, n), 1):
            tasks.append({"type": "special", "kind": "trial", "trial": True, "channel": ck, "lang": lang, "yaml": y,
                          "tint": tint, "tint_name": tname, "slot": slot, "batch": bid,
                          "label": f"{A.CHANNELS[ck]['flag']} Trial {lang.upper()} #{slot} ({tname})"})
    return tasks


def draft_tap(st, parts, msg):
    f = D.flow(st)
    A.drop_buttons(msg)
    if not f or f.get("flow") != "trials" or not f.get("drafts"):
        A.say("⌛ That draft has expired — start again with 🧪 Trials.")
        return None
    if parts[1] == "claude":
        return got_topic(st, f, f["topic"], "claude")
    tasks = make_tasks(f["drafts"], f["pick"], f["n"])
    if not tasks:
        A.say("Nothing to make (no language has a text).")
        return None
    jid = D.new_job(st, tasks, "trials")
    st["jobs"][jid]["topic"] = f["topic"]
    st.pop("desk", None)
    mins = 4 + 2 * len(tasks)
    D.menu(f"🎬 Making {len(tasks)} trial{'s' if len(tasks) > 1 else ''} — they'll come here to approve "
           f"(about {mins}–{mins + 5} min).")
    return ("post", jid)


# ── render + post (in the post job) ─────────────────────────────────────────
def pick_assets(c, cwd):
    """A background and a music file not used yet by this batch (random)."""
    out = []
    for kind, ext in (("video", "mp4"), ("music", "mp3")):
        key = "bg" if kind == "video" else "music"
        got = None
        for _ in range(12):
            got = A.pick(c[kind], ext, cwd)
            if got not in _USED[key]:
                break
        _USED[key].add(got)
        out.append(got)
    return out


def post_trial(task, vid, cap, logf, cwd, jid, st):
    """Instagram trial reel only, never promoted. Returns True when posted."""
    env = {k: v for k, v in D.post_env(task["channel"], task["lang"]).items() if not k.startswith("FACEBOOK_")}
    env.update({"IG_TRIAL_GRADUATION": "MANUAL", "IG_STORY": "0"})
    rv, rc = os.path.relpath(vid, cwd), os.path.relpath(cap, cwd)
    ig_file = os.path.join(os.path.dirname(rv), f"reel_{task['lang']}_instagram.mp4")
    enc = A.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", rv, "-vf",
                "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30", "-c:v", "libx264",
                "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p", "-b:v", "3500k", "-maxrate", "3500k",
                "-bufsize", "7000k", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart",
                "-t", "90", ig_file], logf, cwd=cwd)
    ok = enc and A.sh(["python", "upload_instagram.py", ig_file, rc, task["lang"], "--trial"], logf, env, cwd=cwd)
    mid = None
    try:
        import re
        ids = re.findall(r"🆔 Instagram media id:\s*(\d+)", open(logf, encoding="utf-8", errors="ignore").read())
        mid = ids[-1] if ids else None
    except Exception:
        pass
    if ok and mid:
        job = st["jobs"][jid]
        st.setdefault("trial_posts", []).append({
            "media": mid, "t": A.iso(), "channel": task["channel"], "lang": task["lang"], "topic": job.get("topic", ""),
            "batch": task["batch"], "slot": task["slot"], "tint": task["tint"], "tint_name": task["tint_name"],
            "bg": task.get("bg"), "music": task.get("music"), "job": jid})
        st["trial_posts"] = st["trial_posts"][-120:]
    return bool(ok and mid)


# ── the viral watch (bot job, every hour) ────────────────────────────────────
def _get(path, **params):
    params["access_token"] = os.environ.get("INSTAGRAM_TOKEN", "")
    with urllib.request.urlopen(f"{GRAPH}/{path}?{urllib.parse.urlencode(params)}", timeout=30) as r:
        return json.loads(r.read())


def views(media):
    for metric in ("views", "plays"):
        try:
            d = _get(f"{media}/insights", metric=metric)
            for it in d.get("data", []):
                v = (it.get("values") or [{}])[0].get("value")
                if v is None:
                    v = (it.get("total_value") or {}).get("value")
                if v is not None:
                    return int(v)
        except Exception:
            continue
    return None


def baseline(st, ck, lang):
    """The account's usual views: median of its last 30 reels (cached for a day)."""
    key = f"{ck}:{lang}"
    cache = st.setdefault("trial_base", {})
    b = cache.get(key)
    if b and A.now() - A.parse_iso(b["t"]) < timedelta(days=1):
        return b["v"]
    uid = os.environ.get(f"IG_{ck.upper()}_{lang.upper()}", "")
    if not uid:
        return None
    try:
        items = _get(f"{uid}/media", fields="id,media_product_type", limit=30).get("data", [])
    except Exception:
        return None
    mine = {p["media"] for p in st.get("trial_posts", [])}
    vals = [v for v in (views(i["id"]) for i in items if i.get("media_product_type") == "REELS" and i["id"] not in mine)
            if v is not None]
    if len(vals) < 5:
        return None
    cache[key] = {"t": A.iso(), "v": statistics.median(vals)}
    return cache[key]["v"]


def settings_text(p):
    return (f"{label(p['channel'], p['lang'])} · trial #{p['slot']} · colour {p['tint_name']} ({p['tint']})"
            + (f" · background {p['bg']}" if p.get("bg") else "") + (f" · music {p['music']}" if p.get("music") else ""))


def tick(st):
    last = st.get("_trial_tick")
    if last and A.now() - A.parse_iso(last) < timedelta(minutes=55):
        return
    st["_trial_tick"] = A.iso()
    batches = {}
    for p in st.get("trial_posts", []):
        age_h = (A.now() - A.parse_iso(p["t"])).total_seconds() / 3600
        if p.get("final") or age_h < 1:
            continue
        v = views(p["media"])
        if v is not None:
            p["views"] = v
        base = baseline(st, p["channel"], p["lang"])
        if base:
            p["usual"] = base
        if v is not None and base and not p.get("viral") and v >= max(VIRAL_MIN, VIRAL_X * base):
            p["viral"] = A.iso()
            A.say(f"🔥 A trial went viral: {v:,} views in {age_h:.0f} h (usual {base:,.0f})\n"
                  f"Topic: {p['topic']}\n{settings_text(p)}")
        if age_h >= WATCH_H:
            p["final"] = True
            batches.setdefault(p["batch"], []).append(p)
    for bid, ps in batches.items():                       # 48 h: a short ranking of the batch
        allp = [p for p in st.get("trial_posts", []) if p["batch"] == bid]
        if not all(p.get("final") for p in allp):
            continue
        allp.sort(key=lambda p: -(p.get("views") or 0))
        A.say(f"🧪 Trials after 48 h — {allp[0]['topic']}\n" + "\n".join(
            f"{i}. {p.get('views', '?')} views · {label(p['channel'], p['lang'])} #{p['slot']} {p['tint_name']}"
            + (" 🔥" if p.get("viral") else "") for i, p in enumerate(allp, 1)))
