"""
desk_more.py — YOUR TAKE (Part 1 + Part 2) and COMMENTS in the Telegram desk (stage 4).

Uses exact copies of Newsdesk's code (alerts/nd/: services, series, inbox, shared), so prompts, answer
collection, poll counting, Part 2 and replies work the same as in Newsdesk. The series list and your reply
style are shared with Newsdesk through the private news-metrics repo.

  🗣️ Your take → channel → Open questions / Poll → EN/ES → type a topic or 💡 Ideas from today
               → draft (Gemini; Claude if Gemini fails) → 🎬 Make the videos / 🔄 Rewrite / ✖ → videos → Post
  Part 2       → reminder when due (from the series list) → Collect and write → the text (approval 1)
               → 🎬 Make the video → the video (approval 2, the desk's usual Post / Cancel) → posted, series done
  💬 Comments  → channel, days, writer → 10 cards at a time: comment + suggested reply → ✅ Send / ✏️ Edit / ⏭️ Skip

Public-log rule: comment texts, names and drafts go only to Telegram and the private news-metrics repo.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import types
import urllib.request
from datetime import datetime, timedelta, timezone

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MENU_TEXTS = {"🗣️ Your take": "take", "/yourtake": "take", "💬 Comments": "comments", "/comments": "comments"}
FLOWS = ("take", "comments", "reply")
TAKE_CHANNELS = ("australia", "world", "europe")
PAGE = 10
TICK_MIN = 30
_ND = None


def D():
    import desk
    return desk


def A():
    return D().A


# ── Newsdesk's code, with secrets from the workflow's environment ────────────
class Vault:
    """Newsdesk's secret names → the names the workflow passes (YT_/IG_/FB_<CHANNEL>_<LANG>…)."""
    def __init__(self, hide=()):
        self.hide = set(hide)

    def get(self, key, default=None):
        if key in self.hide:
            return default
        v = os.environ.get(key)
        if v:
            return v
        for pat, env in ((r"([A-Z]+)_INSTAGRAM_([A-Z]{2})_USER_ID", "IG_{}_{}"), (r"([A-Z]+)_FACEBOOK_([A-Z]{2})_PAGE_ID", "FB_{}_{}"),
                         (r"([A-Z]+)_YOUTUBE_([A-Z]{2})_REFRESH_TOKEN", "YT_{}_{}")):
            m = re.fullmatch(pat, key)
            if m:
                return os.environ.get(env.format(*m.groups())) or default
        m = re.fullmatch(r"([A-Z]+)_YOUTUBE_REFRESH_TOKEN", key)              # Australia: one channel for EN + ES
        if m:
            return os.environ.get(f"YT_{m.group(1)}_EN") or default
        return default

    def has(self, key):
        return bool(self.get(key))

    def mask(self, text):
        return text


def nd():
    global _ND
    if _ND:
        return _ND
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from nd import services as S, series as SR, inbox as IB, shared as SH
    cfg = yaml.safe_load(open(os.path.join(HERE, "nd", "config", "channels.yml"), encoding="utf-8")) or {}
    for key, c in cfg.get("channels", {}).items():            # the same defaults as Newsdesk's load_config
        c.setdefault("extra_langs", [])
        for l in c["extra_langs"]:
            c.setdefault("youtube_secret", {}).setdefault(l, f"{key.upper()}_YOUTUBE_{l.upper()}_REFRESH_TOKEN")
    cfg["repos_dir"] = os.path.join(S.DATA_DIR if hasattr(S, "DATA_DIR") else "/tmp", "repos")
    vault = Vault()
    SH.VAULT, SH.CFG = vault, (lambda: cfg)
    _ND = types.SimpleNamespace(S=S, SR=SR, IB=IB, SH=SH, cfg=cfg, vault=vault, gh=S.GitHub(vault, lambda: cfg),
                                app_dir=os.path.join(HERE, "nd"))
    return _ND


def flag(ck):
    return D().CH[ck].split()[0]


def say(text, rows=None):
    return A().tg("sendMessage", chat_id=A().CHAT, text=text[:4000],
                  **({"reply_markup": D().kb(rows)} if rows else {}))


def edit(msg_id, text, rows=None):
    if msg_id:
        A().tg("editMessageText", chat_id=A().CHAT, message_id=msg_id, text=text[:4000],
               reply_markup=D().kb(rows) if rows else {"inline_keyboard": []})


# ── routing (called by desk.route) ───────────────────────────────────────────
def begin(st, name):
    if name == "take":
        f = D().start_flow(st, "take", step="channel", langs=["en", "es"], fmt="questions")
        m = say("🗣️ Your take — which channel?", [[(D().CH[k], f"y:c:{k}")] for k in TAKE_CHANNELS] + [[("✖ Cancel", "x:cancel")]])
        f["msg"] = (m or {}).get("message_id")
        return None
    if name == "comments":
        f = D().start_flow(st, "comments", step="pick", ch="all", days=7, writer="gemini")
        m = say(comments_text(f), comments_rows(f))
        f["msg"] = (m or {}).get("message_id")
        return None


def text(st, f, t):
    """A typed message while one of these flows waits for it."""
    if f["flow"] == "take" and f.get("step") == "topic":
        return draft_take(st, f, t.strip())
    if f["flow"] == "reply" and f.get("step") == "text":
        return send_reply(st, f["ref"], t.strip(), typed=True)
    return None


def tap(st, parts, msg):
    kind, mid = parts[0], (msg or {}).get("message_id")
    try:
        if kind == "y":
            return take_tap(st, parts, mid)
        if kind in ("p", "p2"):
            return part2_tap(st, parts, mid)
        if kind == "m":
            return comments_tap(st, parts, mid)
    except Exception as e:
        A().log(f"Desk ({kind}): {type(e).__name__}")
        say(f"⚠️ {str(e)[:300]}")
    return None


# ══ YOUR TAKE — Part 1 ═══════════════════════════════════════════════════════
def take_tap(st, parts, mid):
    f = D().flow(st)
    if not f or f.get("flow") != "take":
        D().A.drop_buttons({"message_id": mid})
        say("⌛ That choice has expired — start again from the menu.")
        return None
    what = parts[1]
    if what == "c":
        f["channel"], f["step"] = parts[2], "format"
        edit(mid, f"🗣️ Your take — {D().CH[parts[2]]}\nOpen questions or a poll?",
             [[("❓ Open questions", "y:f:questions"), ("🗳️ Poll", "y:f:poll")], [("✖ Cancel", "x:cancel")]])
    elif what == "f":
        f["fmt"], f["step"] = parts[2], "langs"
        edit(mid, take_head(f) + "\nWhich languages?", lang_rows(f))
    elif what == "l":
        if parts[2] == "next":
            if not f["langs"]:
                return None
            f["step"] = "topic"
            edit(mid, take_head(f) + "\n\n✍️ Type the topic (a few words), or tap 💡 for ideas from today's headlines.",
                 [[("💡 Ideas from today", "y:i")], [("✖ Cancel", "x:cancel")]])
        else:
            l = parts[2]
            f["langs"] = [x for x in ("en", "es") if (x in f["langs"]) != (x == l)]
            edit(mid, take_head(f) + "\nWhich languages?", lang_rows(f))
    elif what == "i":
        ideas = take_ideas(f["channel"])
        f["ideas"] = [i.get("topic", "") for i in ideas][:3]
        edit(mid, take_head(f) + "\n\n💡 Today's ideas:\n" + "\n".join(f"• {i.get('topic', '')} — {i.get('why', '')}" for i in ideas)
             + "\n\nTap one, or type your own topic.",
             [[(t[:40], f"y:t:{n}")] for n, t in enumerate(f["ideas"])] + [[("✖ Cancel", "x:cancel")]])
    elif what == "t":
        return draft_take(st, f, f["ideas"][int(parts[2])], mid=mid)
    elif what == "g":
        if parts[2] in ("rewrite", "claude"):
            return draft_take(st, f, f["topic"], mid=mid, writer="claude" if parts[2] == "claude" else None)
        if parts[2] == "make":
            return make_take(st, f, mid)
    return None


def take_head(f):
    return (f"🗣️ Your take — {D().CH[f['channel']]} · {'🗳️ Poll' if f.get('fmt') == 'poll' else '❓ Open questions'}"
            + (f" · {'+'.join(l.upper() for l in f['langs'])}" if f.get("step") not in ("format",) else ""))


def lang_rows(f):
    return [[(f"{'✅' if l in f['langs'] else '⬜'} {l.upper()}", f"y:l:{l}") for l in ("en", "es")],
            [("Next ➡️", "y:l:next"), ("✖ Cancel", "x:cancel")]]


def take_ideas(ck):
    """Three debate-worthy topics from today's headlines — the same prompt as Newsdesk's 'Ideas from today'."""
    N = nd()
    c = N.cfg["channels"][ck]
    day = datetime.now(timezone.utc).strftime("%A")
    url = f"https://raw.githubusercontent.com/{A().OWNER}/{c['news_repo']}/main/{day}NewsEN.yml"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "news-alerts"}), timeout=30) as r:
        heads = N.S._titles(yaml.safe_load(r.read().decode("utf-8")))
    if not heads:
        raise Exception("No headlines for today yet — the daily fetch hasn't run.")
    from nd.extras import ask_ai, extract_json
    prompt = ("These are today's top headlines of a news channel:\n" + "\n".join(f"- {h}" for h in heads) +
              "\n\nPick the 3 that would spark the most civil discussion in the comments (impact on people, what "
              "happens next, what would you do). Avoid tragedies with victims and stories about politicians personally. "
              'Return JSON only: {"ideas":[{"topic":"<2-3 word topic>","why":"<max 12 words>"}]}')
    return extract_json(ask_ai(N.vault, N.cfg, prompt, max_tokens=600)).get("ideas", [])[:3]


def write_take(ck, topic, fmt, writer=None):
    """Newsdesk's writer. Gemini first; Claude if Gemini doesn't work (or when asked)."""
    N = nd()
    if writer != "claude":
        try:
            return N.S.take_from_topic(N.vault, N.cfg, N.app_dir, ck, topic, writer="gemini", fmt=fmt)
        except Exception as e:
            A().log(f"Your take: Gemini didn't work ({type(e).__name__}) — trying Claude")
    r = N.S.take_from_topic(N.vault, N.cfg, N.app_dir, ck, topic, writer="claude", fmt=fmt)
    r["writer"] = "claude"
    return r


def take_summary(d, langs):
    out = []
    for l in langs:
        out.append(f"— {l.upper()} —")
        out.append(f"ℹ️ {d.get(f'context_{l}', '')}")
        if d.get("type") == "poll":
            out.append(f"❓ {d.get(f'question_{l}', '')}")
            out += [f"{n}) {o}" for n, o in enumerate(d.get(f"options_{l}") or [], 1)]
        else:
            out += [f"{n}) {q}" for n, q in enumerate(d.get(f"questions_{l}") or [], 1)]
        out.append("")
    return "\n".join(out)


def draft_take(st, f, topic, mid=None, writer=None):
    if not topic:
        return None
    f["topic"], f["step"] = topic, "drafting"
    note_msg = say(f"✍️ Writing “{topic}”… (about 30 seconds)")
    try:
        r = write_take(f["channel"], topic, f.get("fmt", "questions"), writer)
    except Exception as e:
        f["step"] = "topic"
        say(f"⚠️ Couldn't write it: {str(e)[:250]}\nType another topic, or try again.",
            [[("💡 Ideas from today", "y:i")], [("✖ Cancel", "x:cancel")]])
        return None
    f["yaml"], f["writer"], f["step"] = r["yaml"], r.get("writer"), "review"
    d = yaml.safe_load(r["yaml"])
    A().drop_buttons({"message_id": mid}) if mid else None
    edit((note_msg or {}).get("message_id"),
         take_head(f) + f"\nTopic: {topic} · written by {'Claude' if r.get('writer') == 'claude' else 'Gemini'}\n\n"
         + take_summary(d, f["langs"]) + (f"⚠️ {r['note']}\n" if r.get("note") else "")
         + "Make the videos?",
         [[("🎬 Make the videos", "y:g:make")], [("🔄 Rewrite", "y:g:rewrite"), ("🔄 With Claude", "y:g:claude")],
          [("✖ Cancel", "x:cancel")]])
    return None


def make_take(st, f, mid):
    N = nd()
    d = N.S.check_take(yaml.safe_load(f["yaml"]))
    ck = f["channel"]
    tasks = [{"type": "take", "kind": "question", "channel": ck, "lang": l, "yaml": f["yaml"],
              "title": N.S.take_title(d, l, flag(ck)), "label": f"{flag(ck)} Your take {l.upper()}"} for l in f["langs"]]
    jid = D().new_job(st, tasks, "your take")
    st.pop("desk", None)
    A().drop_buttons({"message_id": mid})
    say(f"🎬 Making {len(tasks)} video{'s' if len(tasks) > 1 else ''}… they'll come here for your OK.")
    return ("post", jid)


# ══ YOUR TAKE — Part 2 ═══════════════════════════════════════════════════════
def tick(st):
    """Bot run: send the Telegram reminder once when a Part 2 is due (the shared series list)."""
    last = st.get("p2_tick")
    now = datetime.now(timezone.utc)
    if last and now - A().parse_iso(last) < timedelta(minutes=TICK_MIN):
        return
    st["p2_tick"] = A().iso()
    SR = nd().SR
    for t in SR.load():
        if t.get("status") not in ("waiting", "due") or t.get("tg_notified"):
            continue
        if nd().S.parse_iso(t["due_at"]) > now:
            continue
        say(f"🗣️ Part 2 is due: {t['flag']} {t['name']} {t['lang'].upper()} — “{t['topic']}”"
            f" ({'poll' if t['type'] == 'poll' else 'open questions'}, Part 1 posted {t['posted_at'][:10]})",
            [[("🗣️ Collect the answers and write Part 2", f"p2:{t['id']}:go")],
             [("⏰ Tomorrow", f"p2:{t['id']}:later"), ("⏭️ Skip it", f"p2:{t['id']}:skip")]])
        SR.update(t["id"], tg_notified=True, status="due")


def part2_tap(st, parts, mid):
    N = nd()
    tid, what = parts[1], parts[2]
    t = N.SR.get(tid)
    if not t:
        say("⚠️ That Part 1 isn't in the series list any more.")
        return None
    if t.get("status") == "posted":
        A().drop_buttons({"message_id": mid})
        say("✅ That Part 2 was already posted.")
        return None
    if what == "later":
        N.SR.update(tid, due_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), tg_notified=False,
                    status="waiting")
        edit(mid, f"⏰ OK — I'll remind you tomorrow about “{t['topic']}”.")
        return None
    if what == "skip":
        N.SR.update(tid, status="skipped")
        edit(mid, f"⏭️ Skipped Part 2 of “{t['topic']}”.")
        return None
    if what == "cancel":
        edit(mid, f"✖ Part 2 of “{t['topic']}” not made. It stays in the series list (Newsdesk → Your take).")
        return None
    if what == "go":
        A().drop_buttons({"message_id": mid})
        say(f"🔎 Collecting the answers to “{t['topic']}” from Instagram, Facebook and YouTube…")
        try:
            got = N.SR.collect(N.vault, N.cfg, N.gh, tid)
        except Exception as e:
            say(f"⚠️ {str(e)[:250]}", [[("🔁 Try again", f"p2:{tid}:go"), ("⏰ Tomorrow", f"p2:{tid}:later")]])
            return None
        n = len(got.get("comments") or [])
        if not n:
            say(f"🤷 No answers yet on “{t['topic']}”." + (f"\n⚠️ {'; '.join(got.get('errors') or [])[:300]}" if got.get("errors") else ""),
                [[("⏰ Tomorrow", f"p2:{tid}:later"), ("⏭️ Skip it", f"p2:{tid}:skip")]])
            return None
        votes = got.get("votes")
        say(f"📥 {n} answer{'s' if n > 1 else ''} collected" + (f" · {votes['total']} poll vote{'s' if votes['total'] != 1 else ''}"
                                                                if votes else "") + ". Writing Part 2…")
        return write_p2(tid, None)
    if what in ("rewrite", "claude"):
        A().drop_buttons({"message_id": mid})
        return write_p2(tid, "claude" if what == "claude" else None)
    if what == "make":
        d, clean = N.SR.check_part2(t, t.get("part2_yaml") or "")
        jid = D().new_job(st, [{"type": "take2", "kind": "answers", "channel": t["channel"], "lang": t["lang"], "tid": tid,
                                "yaml": clean, "title": N.SR.yt_title(t, d, t["lang"]),
                                "label": f"{t['flag']} Your take Part 2 {t['lang'].upper()}"}], "your take part 2")
        A().drop_buttons({"message_id": mid})
        say("🎬 Making the Part 2 video… it'll come here for your OK.")
        return ("post", jid)
    return None


def write_p2(tid, writer):
    """Newsdesk's Part 2 writer: Gemini first, Claude if Gemini doesn't work (or when asked). Approval 1: the text."""
    N = nd()
    out, used = None, writer or "gemini"
    if used == "gemini":
        try:
            out = N.SR.write_part2(N.vault, N.cfg, tid, writer="gemini")
        except Exception as e:
            A().log(f"Part 2: Gemini didn't work ({type(e).__name__}) — trying Claude")
            used = "claude"
    if out is None:
        try:
            out = N.SR.write_part2(N.vault, N.cfg, tid, writer="claude")
        except Exception as e:
            say(f"⚠️ Couldn't write Part 2: {str(e)[:250]}", [[("🔁 Try again", f"p2:{tid}:rewrite")]])
            return None
    t = N.SR.get(tid)
    d = yaml.safe_load(out)
    lang = t["lang"]
    lines = [f"🗣️ Part 2 — {t['flag']} {t['name']} {lang.upper()}: “{t['topic']}” · by {'Claude' if used == 'claude' else 'Gemini'}",
             "", d.get(f"intro_{lang}", "")]
    if t["type"] == "poll":
        lines += [f"❓ {d.get(f'question_{lang}', '')}"] + [f"• {r.get('percent', 0)}% {r.get(f'option_{lang}', '')}"
                                                            for r in d.get("results") or []]
        if d.get(f"takeaway_{lang}"):
            lines.append(d[f"takeaway_{lang}"])
    else:
        for a in d.get("answers") or []:
            lines += ["", f"❓ {a.get(f'question_{lang}', '')}"] + [f"  “{r}”" for r in a.get(f"replies_{lang}") or []]
    lines += ["", d.get(f"outro_{lang}", ""), "", "Make the video?"]
    say("\n".join(lines), [[("🎬 Make the video", f"p2:{tid}:make")],
                           [("🔄 Rewrite", f"p2:{tid}:rewrite"), ("🔄 With Claude", f"p2:{tid}:claude")],
                           [("✖ Cancel", f"p2:{tid}:cancel")]])
    return None


# ── rendering (post job) ─────────────────────────────────────────────────────
def render(task, logf):
    ck, lang = task["channel"], task["lang"]
    cwd = D().reels_dir(ck)
    c = A().CHANNELS[ck]
    v, m = A().pick(c["video"], "mp4", cwd), A().pick(c["music"], "mp3", cwd)
    if not (v and m):
        return None
    out = f"tg_{task['kind']}_{lang}"
    subprocess.run(["rm", "-rf", os.path.join(cwd, out)])
    os.makedirs(os.path.join(cwd, out), exist_ok=True)
    if task["type"] == "take":                     # the same command as Newsdesk's YOUR TAKE
        with open(os.path.join(cwd, "tg_take.yml"), "w", encoding="utf-8") as f:
            f.write(task["yaml"])
        ok = A().sh(["python", "make_reels_es_en.py", "--yml", "tg_take.yml", "--video", v, "--music", m, "--lang", lang,
                     "--output-dir", out, "--your-take"], logf, cwd=cwd)
        vids = sorted(x for x in glob.glob(f"{cwd}/{out}/reel_{lang}_*.mp4") if not x.endswith("_instagram.mp4"))
        caps = sorted(glob.glob(f"{cwd}/{out}/caption_{lang}_*.txt"))
        return (vids[0], caps[0]) if ok and vids and caps else None
    # Part 2: Newsdesk's scenes (series.scenes), each with make_reels --scene, joined with one music track
    N = nd()
    t = N.SR.get(task["tid"])
    d, _ = N.SR.check_part2(t, task["yaml"])
    parts = []
    for i, sc in enumerate(N.SR.scenes(t, d)):
        with open(os.path.join(cwd, out, f"scene{i}.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(sc, f, allow_unicode=True, sort_keys=False, width=1000)
        if not A().sh(["python", "make_reels_es_en.py", "--yml", f"{out}/scene{i}.yml", "--video", v, "--lang", lang,
                       "--output-dir", f"{out}/s{i}", "--scene"], logf, cwd=cwd):
            return None
        src = sorted(glob.glob(f"{cwd}/{out}/s{i}/reel_{lang}_*.mp4"))
        if not src:
            return None
        has_a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                                src[0]], capture_output=True, text=True).stdout.strip()
        if not has_a:
            fixed = src[0].replace(".mp4", "_a.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src[0], "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                            "-shortest", "-c:v", "copy", "-c:a", "aac", fixed], check=True)
            src = [fixed]
        parts.append(src[0])
    durs = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p_],
                                 capture_output=True, text=True).stdout.strip() or 0) for p_ in parts]
    total, n = sum(durs), len(parts)
    ins = sum((["-i", p_] for p_ in parts), []) + ["-stream_loop", "-1", "-i", os.path.join(cwd, m)]
    fc = "".join(f"[{i}:v]fps=30,format=yuv420p,setsar=1[v{i}];[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}];"
                 for i in range(n))
    fc += "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][voi];"
    fc += (f"[{n}:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.04[mus];"
           f"[voi][mus]amix=inputs=2:duration=first:normalize=0,afade=t=out:st={max(0, total - 2):.2f}:d=2[a]")
    final = f"{cwd}/{out}/reel_{lang}_answers.mp4"
    if not A().sh(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                   "-t", f"{total:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k",
                   "-movflags", "+faststart", final], logf, cwd="."):
        return None
    cover = None
    for cp in sorted(glob.glob(f"{cwd}/{out}/s0/cover_*.jpg"))[:1]:
        cover = f"cover_{lang}_answers.jpg"
        with open(cp, "rb") as a_, open(f"{cwd}/{out}/{cover}", "wb") as b_:
            b_.write(a_.read())
    cap = f"{cwd}/{out}/caption_{lang}_answers.txt"
    with open(cap, "w", encoding="utf-8") as f:
        f.write(N.SR.caption(t, d, lang) + "\n")
    with open(f"{cwd}/{out}/meta_{lang}_answers.json", "w", encoding="utf-8") as f:
        json.dump({"channel": ck, "lang": lang, "kind": "answers", "trial": False, "voice": True, "duration_s": round(total, 2),
                   "background": v, "music": m, "cover": cover, "source": "telegram",
                   "comment": "What would you add? 👇" if lang == "en" else "¿Qué añadirías tú? 👇",
                   "part1_topic": t["topic"], "series_id": t["id"], "created_at": A().iso()}, f, ensure_ascii=False)
    return final, cap


def after_post(task, jid):
    """Posted: Part 1 goes into the shared series list; Part 2 marks its series as done."""
    N = nd()
    try:
        if task["type"] == "take":
            d = yaml.safe_load(task["yaml"])
            e = N.SR.remember(N.cfg, task["channel"], task["lang"], d, f"tg-{jid}")
            A().log(f"Your take {task['lang']}: Part 2 due {e['due_at'][:10]}")
        elif task["type"] == "take2":
            N.SR.update(task["tid"], status="posted", part2_posted_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        A().log(f"Series list not updated ({type(e).__name__})")
        say(f"⚠️ Posted, but the series list couldn't be updated: {str(e)[:200]}")


# ══ COMMENTS ═════════════════════════════════════════════════════════════════
def comments_text(f):
    return ("💬 Comments — new ones you haven't replied to.\n"
            f"Channel: {'all' if f['ch'] == 'all' else D().CH[f['ch']]} · last {f['days']} days · "
            f"replies written by {'Gemini (Claude if it fails)' if f['writer'] == 'gemini' else 'Claude'}")


def comments_rows(f):
    mark = lambda on, t: f"{'✅' if on else '⬜'} {t}"
    return [[(mark(f["ch"] == "all", "All"), "m:c:all")] + [(mark(f["ch"] == k, D().CH[k].split()[0]), f"m:c:{k}") for k in TAKE_CHANNELS],
            [(mark(f["days"] == d, f"{d} days"), f"m:d:{d}") for d in (3, 7, 14)],
            [(mark(f["writer"] == "gemini", "Gemini"), "m:w:gemini"), (mark(f["writer"] == "claude", "Claude"), "m:w:claude")],
            [("🔎 Collect", "m:go"), ("✖ Cancel", "x:cancel")]]


def comments_tap(st, parts, mid):
    what = parts[1]
    if what in ("c", "d", "w", "go"):
        f = D().flow(st)
        if not f or f.get("flow") != "comments":
            say("⌛ That choice has expired — start again from the menu.")
            return None
        if what == "c":
            f["ch"] = parts[2]
        elif what == "d":
            f["days"] = int(parts[2])
        elif what == "w":
            f["writer"] = parts[2]
        if what != "go":
            edit(mid, comments_text(f), comments_rows(f))
            return None
        edit(mid, comments_text(f) + "\n\n🔎 Collecting…")
        return collect_comments(st, f)
    ib = st.get("inbox") or {}
    if what == "more":
        A().drop_buttons({"message_id": mid})
        return show_page(st)
    item = next((i for i in ib.get("items", []) if i["ref"] == parts[2]), None) if len(parts) > 2 else None
    if not item:
        say("⌛ That comment isn't in the current round any more — start again with 💬 Comments.")
        return None
    if item["status"] != "shown":
        A().drop_buttons({"message_id": mid})
        return None
    if what == "s":
        return send_reply(st, item["ref"], item.get("suggest") or "")
    if what == "k":
        item["status"] = "skipped"
        edit(mid, item["card"] + "\n\n⏭️ Skipped")
        return None
    if what == "e":
        D().start_flow(st, "reply", step="text", ref=item["ref"])
        say(f"✏️ Type your reply to {item['ref']} ({item['flag']} {item['platform']}) — your next message is sent as it is.")
        return None
    return None


def collect_comments(st, f):
    N = nd()
    data = N.IB.collect(N.vault, N.cfg, days=f["days"])
    items = [i for i in data["items"] if f["ch"] == "all" or i.get("channel") == f["ch"]]
    for n, it in enumerate(items, 1):
        it["ref"], it["status"] = f"C{n}", "new"
        it["text"] = str(it.get("text", ""))[:600]
    st["inbox"] = {"batch": data["batch"], "at": A().iso(), "writer": f["writer"], "items": items}
    st.pop("desk", None)
    errs = data.get("errors") or []
    A().log(f"Comments: {len(items)} new")
    if not items:
        D().menu("🎉 No new comments to answer." + (f"\n⚠️ Couldn't read: {len(errs)} account(s)." if errs else ""))
        return None
    say(f"📥 {len(items)} new comment{'s' if len(items) > 1 else ''}. Here come the first {min(PAGE, len(items))} with suggested replies…"
        + (f"\n⚠️ Couldn't read {len(errs)} account(s): " + "; ".join(e.split(':')[0] for e in errs)[:300] if errs else ""))
    return show_page(st)


def show_page(st):
    N = nd()
    ib = st.get("inbox") or {}
    page = [i for i in ib.get("items", []) if i["status"] == "new"][:PAGE]
    if not page:
        D().menu("✅ That's all the comments in this round.")
        return None
    vault = N.vault if ib.get("writer") != "claude" else Vault(hide=("GEMINI_API_KEY",))   # no Gemini key → Claude
    try:
        text = N.IB.draft_with_ai(vault, N.cfg, {"batch": ib["batch"], "items": page}, free_only=False)
        replies = (yaml.safe_load(text) or {}).get("replies") or {}
    except Exception as e:
        say(f"⚠️ Couldn't write suggestions ({str(e)[:150]}) — you can still write your own.")
        replies = {}
    for it in page:
        r = str(replies.get(it["ref"], "SKIP") or "SKIP").strip()
        it["suggest"] = None if r.upper() == "SKIP" else r
        when = (it.get("at") or "")[:10]
        it["card"] = (f"{it['ref']} · {it.get('flag', '')} {it.get('name', '')} {str(it.get('lang', '')).upper()} · "
                      f"{str(it.get('platform', '')).title()} · {when}\n"
                      + (f"🎬 {str(it.get('video') or it.get('post') or '')[:90]}\n" if (it.get('video') or it.get('post')) else "")
                      + f"💬 {it.get('author') or 'Someone'}: {it['text']}\n\n"
                      + (f"✍️ {it['suggest']}" if it["suggest"] else "🤖 Suggestion: skip it (spam, unclear or not worth a reply)"))
        rows = ([[("✅ Send", f"m:s:{it['ref']}"), ("✏️ Edit", f"m:e:{it['ref']}"), ("⏭️ Skip", f"m:k:{it['ref']}")]]
                if it["suggest"] else [[("✏️ Write a reply", f"m:e:{it['ref']}"), ("⏭️ Skip", f"m:k:{it['ref']}")]])
        m = say(it["card"], rows)
        it["msg"], it["status"] = (m or {}).get("message_id"), "shown"
    left = sum(1 for i in ib["items"] if i["status"] == "new")
    say(f"{len(page)} shown · {left} more waiting." if left else f"{len(page)} shown — that's all of them.",
        [[(f"➡️ Next {min(PAGE, left)}", "m:more")]] if left else None)
    return None


def send_reply(st, ref, reply, typed=False):
    N = nd()
    ib = st.get("inbox") or {}
    item = next((i for i in ib.get("items", []) if i["ref"] == ref), None)
    if typed:
        st.pop("desk", None)
    if not item or not reply:
        say("⚠️ Nothing to send.")
        return None
    try:
        N.IB.post_reply(N.vault, item, reply)
    except Exception as e:
        say(f"⚠️ {ref} not sent: {str(e)[:200]}", [[("✏️ Try again", f"m:e:{ref}"), ("⏭️ Skip", f"m:k:{ref}")]])
        return None
    item["status"], item["sent"] = "sent", reply
    edit(item.get("msg"), item["card"].rsplit("\n\n", 1)[0] + f"\n\n✅ Sent: {reply}")
    if typed:
        say(f"✅ {ref} sent.")
    N.IB.add_my_reply(item, reply)                  # your style file learns from it (shared with Newsdesk)
    A().log("Comments: one reply sent")
    return None
