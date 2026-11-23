"""
desk_auto.py — ⚡ Auto Post All (change 4, 23 Nov 2026).

Under 🚨 Breaking check, 🔴 Special edition and 📚 Explainer there is a ⚡ Auto Post All button. It asks ONLY for the
topic, then: picks the channel (Australia / World / Europe) from the topic, writes it (Gemini; Claude if Gemini fails
or its answer doesn't pass the checks), makes the videos and posts them in the saved languages (EN + ES) — no preview,
no approval. Safeguards:
  • the writer is told this goes out unchecked: only facts its sources confirm, "reports say" for the rest
  • the existing YAML checks + a length check; if the text fails twice it stops and nothing is posted
  • a daily cap (AUTO_DAILY_CAP, default 3 a day, Sydney time)
  • right after posting: "✅ Auto-posted" with the links and a 🗑 Delete button per platform for 30 minutes
Button codes: au:<kind> (start), dl:<job>:<n>:<platform> (delete). Public-log rule: only short status lines.
"""
import os
import re
import urllib.parse
import urllib.request
import json
from datetime import timedelta

A = None                                       # alerts module (set by desk.py)
D = None                                       # desk module
KINDS = {"breaking": "🚨 Breaking news", "special": "🔴 Special edition", "explainer": "📚 Explainer", "trials": "🧪 Trials"}
CAP = int(os.environ.get("AUTO_DAILY_CAP") or 3)
LANGS = [l.strip() for l in (os.environ.get("AUTO_LANGS") or "en,es").split(",") if l.strip()]
DELETE_MIN = 30
HEDGE = ("\n\n(AUTOMATIC POST — nobody checks this before it is published. Use only facts that your sources confirm; "
         "for anything not confirmed by two sources write \"reports say\" / \"según informes\". No guesses, no numbers "
         "you can't source.)")
GRAPH = "https://graph.facebook.com/v21.0"


def button(kind):
    return [("⚡ Auto Post All (just type the topic)", f"au:{kind}")]


def used_today(st):
    today = A.now().astimezone(A.SYD).date()
    return [t for t in st.get("auto_log", []) if A.parse_iso(t).astimezone(A.SYD).date() == today]


def start(st, kind, msg):
    A.drop_buttons(msg)
    n = len(used_today(st))
    if n >= CAP:
        D.menu(f"⚡ Auto Post All: today's limit is reached ({n} of {CAP}). Use the normal flow (with preview) instead.")
        return None
    D.start_flow(st, "auto", kind=kind, step="topic", channels=[], langs=list(LANGS))
    A.say(f"⚡ {KINDS[kind]} — Auto Post All\nType the topic (a few words). I'll pick the channel, write it, make the "
          f"videos and post them in {' + '.join(l.upper() for l in LANGS)} without asking.\n"
          f"Today: {n} of {CAP} used. ✖ Cancel stops it.")
    return None


def pick_channel(topic):
    try:
        out, _ = A.ask("Which news channel fits this topic best? australia = it happens in or mainly affects Australia; "
                       "europe = it happens in or mainly affects a European country or the EU; world = anything else. "
                       f"Answer with one word only.\n\nTopic: {topic}", max_tokens=10)
        w = re.findall(r"australia|europe|world", out.lower())
        return w[0] if w else "world"
    except Exception:
        return "world"


def check_special(y, langs):
    """Extra checks on top of write_special's: 5 headlines per language, each 4–30 words, a 3–4 line title."""
    for l in langs:
        news = y.get(f"news_{l}") or []
        if len(news) < 5:
            return f"{l.upper()}: fewer than 5 headlines"
        for h in news:
            n = len(str(h).split())
            if n < 4 or n > 30:
                return f"{l.upper()}: a headline of {n} words"
        lines = [x for x in str(y.get(f"title_{l}", "")).split("\n") if x.strip()]
        if not 2 <= len(lines) <= 4:
            return f"{l.upper()}: the title has {len(lines)} lines"
    return None


def check_explainer(y, langs):
    for l in langs:
        facts = y.get(f"facts_{l}") or []
        if not 3 <= len(facts) <= 6 or any(len(str(x).split()) > 30 for x in facts):
            return f"{l.upper()}: the facts don't fit (3–6 facts, 30 words max)"
    return None


def write(kind, ck, topic, langs, headline=None):
    """(yaml, writer). Gemini first; if the text fails the checks, Claude once. Raises if both fail."""
    last = None
    for prefer in ("gemini", "claude"):
        try:
            if kind == "explainer":
                y, who = D.write_explainer(ck, topic, prefer)     # its system text already says: hedge the unconfirmed
                bad = check_explainer(y, langs)
            else:
                y, who = A.write_special(ck, topic, (headline or topic) + HEDGE, prefer)
                if any(l in langs for l in ("pt", "id")):
                    A.add_extra(ck, y, prefer)
                bad = check_special(y, langs)
            if not bad:
                return y, who
            last = bad
            A.log(f"auto: {prefer} text failed a check")
        except Exception as e:
            last = str(e)[:100] or type(e).__name__
            A.log(f"auto: {prefer} failed ({type(e).__name__})")
    raise Exception(f"the text didn't pass the checks ({last})")


def got_topic(st, f, topic):
    kind, langs = f["kind"], f["langs"]
    if len(used_today(st)) >= CAP:
        st.pop("desk", None)
        D.menu(f"⚡ Today's Auto Post All limit is reached ({CAP}).")
        return None
    if kind == "trials":
        return trials_topic(st, topic)
    ck = pick_channel(topic)
    A.say(f"⚡ {D.CH[ck]} · {KINDS[kind]} · {' + '.join(l.upper() for l in langs)}\n✍️ Writing… (about a minute)")
    try:
        y, who = write(kind, ck, topic, langs)
    except Exception as e:
        st.pop("desk", None)
        D.menu(f"⚠️ Auto Post All stopped: {str(e)[:150]}. Nothing was posted.")
        return None
    tasks = []
    for lang in langs:
        if kind == "explainer":
            tasks.append(D.explainer_task(ck, lang, y))
        elif y.get(f"news_{lang}"):
            if kind == "breaking":
                y["breaking_lead"] = True                     # the small red BREAKING tag on the lead story
            tasks.append({"type": "special", "kind": "special", "channel": ck, "lang": lang, "yaml": y,
                          "label": f"{D.CH[ck].split()[0]} {'Breaking' if kind == 'breaking' else 'Special'} {lang.upper()}"})
    if not tasks:
        st.pop("desk", None)
        D.menu("⚠️ Auto Post All stopped: no language has a text. Nothing was posted.")
        return None
    jid = D.new_job(st, tasks, f"auto {kind}")
    st["jobs"][jid]["auto"] = True
    st.setdefault("auto_log", []).append(A.iso())
    st["auto_log"] = st["auto_log"][-30:]
    st.pop("desk", None)
    head = (str(y.get("title_en", "")).split("\n") + [""])[2] if kind != "explainer" else topic
    mins = 4 + 2 * len(tasks)
    D.menu(f"⚡ Written by {'Claude' if who == 'claude' else 'Gemini'}: {head or topic}\n🎬 Making and posting "
           f"{len(tasks)} video{'s' if len(tasks) > 1 else ''} (about {mins}–{mins + 6} min). You'll get the links "
           f"and a 🗑 Delete button for {DELETE_MIN} min.")
    return ("post", jid)


def alert_all(st, items):
    """⚡ Auto post all: every story of a breaking alert, written, made and posted without review (one job)."""
    n = len(used_today(st))
    if n >= CAP:
        D.menu(f"⚡ Today's Auto Post All limit is reached ({n} of {CAP}). Use 📝 Draft all instead (with review).")
        return None
    langs = list(LANGS)
    A.say(f"⚡ Auto posting {len(items)} stor{'y' if len(items) == 1 else 'ies'} in {' + '.join(l.upper() for l in langs)}, "
          f"no review.\n✍️ Writing… (about a minute each)")
    tasks, done, failed = [], [], []
    for i, it in enumerate(items):
        ck = it["channel"]
        try:
            y, who = write("breaking", ck, it["topic"], langs, it.get("headline"))
        except Exception as e:
            failed.append(f"{D.CH[ck].split()[0]} {it['topic']} ({str(e)[:60]})")
            continue
        y["breaking_lead"] = True                              # the small red BREAKING tag on the lead story
        for lang in langs:
            if y.get(f"news_{lang}"):
                tasks.append({"type": "special", "kind": "special", "channel": ck, "lang": lang, "yaml": y, "slot": f"b{i + 1}",
                              "label": f"{D.CH[ck].split()[0]} Breaking {lang.upper()} · {it['topic'][:28]}"})
        done.append(f"{D.CH[ck].split()[0]} {it['topic']} ({'Claude' if who == 'claude' else 'Gemini'})")
    if not tasks:
        D.menu("⚠️ Auto post all stopped: no story passed the checks. Nothing was posted."
               + ("\n" + "\n".join("• " + f for f in failed) if failed else ""))
        return None
    jid = D.new_job(st, tasks, "auto breaking all")
    st["jobs"][jid]["auto"] = True
    st.setdefault("auto_log", []).append(A.iso())              # one press = one of the daily Auto Posts
    st["auto_log"] = st["auto_log"][-30:]
    mins = 4 + 2 * len(tasks)
    D.menu("⚡ Written:\n" + "\n".join("• " + x for x in done)
           + (f"\n⚠️ Skipped (text failed the checks):\n" + "\n".join("• " + f for f in failed) if failed else "")
           + f"\n🎬 Making and posting {len(tasks)} video{'s' if len(tasks) > 1 else ''} (about {mins}–{mins + 8} min). "
           f"You'll get the links and a 🗑 Delete button for {DELETE_MIN} min.")
    return ("post", jid)


# ── after posting: links + delete buttons ────────────────────────────────────
def ids_from_log(logf):
    try:
        text = open(logf, encoding="utf-8", errors="ignore").read()
    except Exception:
        return {}
    out = {}
    m = re.findall(r"Uploaded! Video ID:\s*([\w-]+)", text)
    if m:
        out["yt"] = m[-1]
    m = re.findall(r"🆔 Instagram media id:\s*(\d+)", text)
    if m:
        out["ig"] = m[-1]
    m = re.findall(r"(?m)^\s+Video ID:\s*(\d+)", text)              # Facebook (printed indented by upload_instagram.py)
    if m:
        out["fb"] = m[-1]
    return out


def after_post(st, jid, posted):
    """posted = [(task, logf)] that went out. Sends the links and the delete buttons."""
    job = st["jobs"][jid]
    job["posted"], rows, lines = [], [], []
    for n, (task, logf) in enumerate(posted):
        ids = ids_from_log(logf)
        job["posted"].append({"channel": task["channel"], "lang": task["lang"], **ids})
        links = []
        if ids.get("yt"):
            links.append(f"YouTube https://youtube.com/shorts/{ids['yt']}")
        if ids.get("fb"):
            links.append(f"Facebook https://www.facebook.com/watch/?v={ids['fb']}")
        if ids.get("ig"):
            links.append("Instagram ✓")
        lines.append(f"{task['label']}\n   " + "\n   ".join(links or ["(no ids in the log)"]))
        rows.append([(f"🗑 {p.upper()} {task['lang'].upper()}", f"dl:{jid}:{n}:{p}") for p in ("yt", "ig", "fb") if ids.get(p)])
    job["delete_until"] = A.iso(A.now() + timedelta(minutes=DELETE_MIN))
    rows = [r for r in rows if r] + [[("✔ Keep all", f"dl:{jid}:0:keep")]]
    A.say("✅ Auto-posted\n" + "\n".join(lines) + f"\n\nWrong? Delete within {DELETE_MIN} min:", rows)


def _http(method, url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _err(b):
    try:
        d = json.loads(b or b"{}")
        e = d.get("error")
        return (e.get("message") if isinstance(e, dict) else e) or ""
    except Exception:
        return ""


def delete_one(ck, lang, plat, pid):
    L, C = lang.upper(), ck.upper()
    if plat == "yt":
        data = urllib.parse.urlencode({"client_id": os.environ.get("YOUTUBE_CLIENT_ID", ""),
                                       "client_secret": os.environ.get("YOUTUBE_CLIENT_SECRET", ""),
                                       "refresh_token": os.environ.get(f"YT_{C}_{L}", ""),
                                       "grant_type": "refresh_token"}).encode()
        st, b = _http("POST", "https://oauth2.googleapis.com/token", data, {"Content-Type": "application/x-www-form-urlencoded"})
        if st != 200:
            raise Exception("YouTube login failed")
        tok = json.loads(b)["access_token"]
        st, b = _http("DELETE", f"https://www.googleapis.com/youtube/v3/videos?id={pid}", headers={"Authorization": f"Bearer {tok}"})
        if st not in (200, 204):
            raise Exception(f"YouTube said no ({st} {_err(b)[:80]})")
        return
    user = os.environ.get("INSTAGRAM_TOKEN", "")
    if plat == "ig":
        st, b = _http("DELETE", f"{GRAPH}/{pid}?access_token={urllib.parse.quote(user)}")
        if st != 200:
            raise Exception(f"Instagram said no ({st} {_err(b)[:80]})")
        return
    page = os.environ.get(f"FB_{C}_{L}", "")
    st, b = _http("GET", f"{GRAPH}/{page}?fields=access_token&access_token={urllib.parse.quote(user)}")
    if st != 200:
        raise Exception("Facebook page login failed")
    ptok = json.loads(b).get("access_token", "")
    st, b = _http("DELETE", f"{GRAPH}/{pid}?access_token={urllib.parse.quote(ptok)}")
    if st != 200:
        raise Exception(f"Facebook said no ({st} {_err(b)[:80]})")


def delete_tap(st, parts, msg):
    jid, n, plat = parts[1], int(parts[2]), parts[3]
    job = (st.get("jobs") or {}).get(jid) or {}
    if plat == "keep":
        A.drop_buttons(msg)
        return None
    until = job.get("delete_until")
    if not until or A.now() > A.parse_iso(until):
        A.drop_buttons(msg)
        A.say("⌛ The delete buttons have expired: delete it on the platform by hand.")
        return None
    item = (job.get("posted") or [])[n] if n < len(job.get("posted") or []) else None
    if not item or not item.get(plat):
        A.say("Already deleted, or no id for that one.")
        return None
    name = {"yt": "YouTube", "ig": "Instagram", "fb": "Facebook"}[plat]
    try:
        delete_one(item["channel"], item["lang"], plat, item[plat])
        item.pop(plat)
        A.say(f"🗑 Deleted from {name} ({item['lang'].upper()}).")
    except Exception as e:
        A.say(f"⚠️ {name} {item['lang'].upper()} not deleted: {str(e)[:120]}. Delete it in the app.")
    return None


def trials_topic(st, topic):
    """⚡ for 🧪 Trials: every account that can post trials, AUTO_TRIALS_N (3) each, on this topic."""
    T = D.TRIALS
    acc = T.accounts()
    if not acc:
        st.pop("desk", None)
        D.menu("⚠️ No account can post trial reels yet. Nothing was posted.")
        return None
    drafts, who = {}, "gemini"
    for ck in sorted({c for c, _ in acc}):
        A.say(f"⚡ 🧪 {D.CH[ck]} · {T.AUTO_N} trials per account\n✍️ Writing…")
        try:
            drafts[ck], who = write("special", ck, topic, [l for c, l in acc if c == ck])
        except Exception as e:
            A.say(f"⚠️ {D.CH[ck]}: {str(e)[:120]} — skipped.")
    tasks = T.make_tasks(drafts, [f"{c}:{l}" for c, l in acc], T.AUTO_N)
    st.pop("desk", None)
    if not tasks:
        D.menu("⚠️ Auto Post All stopped: no text passed the checks. Nothing was posted.")
        return None
    jid = D.new_job(st, tasks, "auto trials")
    st["jobs"][jid].update(auto=True, topic=topic)
    st.setdefault("auto_log", []).append(A.iso())
    mins = 4 + 2 * len(tasks)
    D.menu(f"⚡ Making and posting {len(tasks)} trial reels (Instagram only, never promoted; about {mins}–{mins + 6} "
           "min). I'll tell you if one goes viral.")
    return ("post", jid)
