"""
series.py — YOUR TAKE as a two-part series.

  Part 1 (questions or poll) is remembered when it's posted (takes.json).
  After N days Newsdesk reminds you, collects the comments on that post from
  Instagram, Facebook and YouTube, and Gemini (or Claude, or you via the
  export for your Claude project) writes Part 2: the most popular answers,
  anonymous and rewritten, or the poll results. Part 2 is built from scenes
  (make_reels_es_en.py --scene) joined into one video.
"""

import os
import re
import json
import time
import uuid
import urllib.parse
from datetime import datetime, timedelta, timezone

import yaml

from . import services as S
from . import inbox as IB
from .runner import DATA_DIR

TAKES_PATH = os.path.join(DATA_DIR, "takes.json")
WORDS = {"en": ["one", "two", "three"], "es": ["uno", "dos", "tres"]}
REPLY_LEADS = {"en": ["Many of you said:", "Others said:", "And some of you said:"],
               "es": ["Muchos dijisteis:", "Otros dijeron:", "Y algunos dijisteis:"]}
MONTHS_ES = S.MONTHS_ES


SHARED_PATH = "series/takes.json"           # stage 4a: shared with the Telegram desk (news-metrics, private)
_SNAP = {}                                  # id → entry as last read (to merge only what changed)


def _remember_snapshot(items):
    _SNAP.clear()
    _SNAP.update({t["id"]: json.dumps(t, sort_keys=True, ensure_ascii=False) for t in items if t.get("id")})


def load():
    from . import shared as SH
    try:
        text, sha = SH.read_text(SHARED_PATH)
        if text is None:                        # first time: copy this computer's list to news-metrics
            items = S.read_json(TAKES_PATH, [])
            if items:
                SH.write_text(SHARED_PATH, json.dumps(items, ensure_ascii=False, indent=1), None,
                              "Your take: series list (copied from Newsdesk)")
        else:
            items = json.loads(text)
            S.write_json(TAKES_PATH, items)     # the local copy, for when GitHub can't be reached
    except SH.Offline:
        items = S.read_json(TAKES_PATH, [])
    _remember_snapshot(items)
    return items


def save(items):
    """Saves the local copy, then merges into the shared list: only entries changed here replace the shared ones."""
    from . import shared as SH
    S.write_json(TAKES_PATH, items)
    mine = {t["id"]: t for t in items}
    changed = {i: t for i, t in mine.items() if json.dumps(t, sort_keys=True, ensure_ascii=False) != _SNAP.get(i)}
    removed = set(_SNAP) - set(mine)

    def merge(remote):
        out = [changed.get(t["id"], t) for t in (remote or []) if t["id"] not in removed]
        have = {t["id"] for t in out}
        new = [t for t in items if t["id"] in changed and t["id"] not in have]
        return (new + out)[:200]
    try:
        merged = SH.modify_json(SHARED_PATH, merge, [], "Your take: series list")
        S.write_json(TAKES_PATH, merged)
        _remember_snapshot(merged)
    except SH.Offline as e:
        print(f"⚠️ Series list saved on this computer only ({e}) — it syncs at the next save")


def get(tid):
    return next((t for t in load() if t["id"] == tid), None)


def update(tid, **fields):
    items = load()
    for t in items:
        if t["id"] == tid:
            t.update(fields)
    save(items)
    return get(tid)


def topic_of(title):
    lines = [l.strip() for l in str(title).split("\n") if l.strip()
             and l.strip().lower() not in ("your take", "tu opinión", "tu opinion", "your answers", "vuestras respuestas")]
    return lines[1] if len(lines) >= 3 else (lines[-1] if lines else "")


def remember(cfg, key, lang, data, job_id):
    """Called when a Part 1 has been posted."""
    days = int(data.get("part2_days", 2) or 2)
    now = datetime.now(timezone.utc)
    c = cfg["channels"][key]
    entry = {"id": f"t{uuid.uuid4().hex[:8]}", "channel": key, "name": c["name"], "flag": c["flag"], "color": c["color"],
             "lang": lang, "type": "poll" if data.get("type") == "poll" else "questions",
             "topic": topic_of(data.get(f"title_{lang}", "")), "part1": data, "job": job_id,
             "posted_at": now.isoformat(), "due_at": (now + timedelta(days=days)).isoformat(),
             "status": "waiting", "comments": [], "votes": None, "part2_yaml": "", "notified": False}
    items = load()
    items.insert(0, entry)
    save(items[:200])
    return entry


# ── collecting the answers ───────────────────────────────────────────────────
def _records(gh, t):
    """news-metrics records of this Part 1 (one per platform)."""
    t0 = S.parse_iso(t["posted_at"])
    days = {(t0 + timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1, 0, 1)}
    out = {}
    for p in gh.metrics_paths(max_age=0):
        base = os.path.basename(p)
        if not p.startswith("posts/") or base[:10] not in days:
            continue
        m = S.REC.match(base)
        if not m or m.group(2) != t["channel"] or m.group(3) != t["lang"] or m.group(5) != "question" or m.group(6):
            continue
        rec = gh.metrics_file(p) or {}
        rt = S.parse_iso(rec.get("posted_at"))
        if rt and abs((rt - t0).total_seconds()) > 4 * 3600:
            continue
        out[m.group(4)] = {"post_id": m.group(7), "url": rec.get("url")}
    return out


def _comments(vault, cfg, t, platform, post_id):
    key, lang = t["channel"], t["lang"]
    P, L = key.upper(), lang.upper()
    if platform == "instagram":
        tok = vault.get("INSTAGRAM_TOKEN")
        me = IB._g(vault.get(f"{P}_INSTAGRAM_{L}_USER_ID"), tok, fields="username")["username"]
        cs = IB._g(f"{post_id}/comments", tok, fields="id,text,username,timestamp", limit=100).get("data", [])
        return [c.get("text", "") for c in cs if c.get("username") != me]
    if platform == "facebook":
        page = vault.get(f"{P}_FACEBOOK_{L}_PAGE_ID")
        tok = IB._g(page, vault.get("INSTAGRAM_TOKEN"), fields="access_token")["access_token"]
        cs = IB._g(f"{post_id}/comments", tok, fields="id,message,from", filter="toplevel", limit=100).get("data", [])
        return [c.get("message", "") for c in cs if (c.get("from") or {}).get("id") != page]
    if platform == "youtube":
        secret = cfg["channels"][key]["youtube_secret"][lang]
        tok = IB.yt_token(vault, secret)
        cid = IB._yt_get(tok, "channels", part="id", mine="true")["items"][0]["id"]
        th = IB._yt_get(tok, "commentThreads", part="snippet", videoId=post_id, maxResults=100).get("items", [])
        out = []
        for x in th:
            s = x["snippet"]["topLevelComment"]["snippet"]
            if (s.get("authorChannelId") or {}).get("value") != cid:
                out.append(s.get("textOriginal", ""))
        return out
    return []


VOTE = re.compile(r"^\s*(?:option|opción|opcion|la|el|#|nº|n°)?\s*([1-3])(?:\b|[).:!,\s])", re.I)


def count_votes(texts, n):
    counts = [0] * n
    for t in texts:
        m = VOTE.match(t or "")
        if m and 1 <= int(m.group(1)) <= n:
            counts[int(m.group(1)) - 1] += 1
    total = sum(counts)
    return {"counts": counts, "total": total,
            "percent": [round(100 * c / total) for c in counts] if total else [0] * n}


def collect(vault, cfg, gh, tid):
    t = get(tid)
    if not t:
        raise ValueError("Unknown Part 1")
    recs = _records(gh, t)
    if not recs:
        raise Exception("Couldn't find the Part 1 posts in news-metrics yet.")
    comments, errors = [], []
    for platform, r in recs.items():
        try:
            for text in _comments(vault, cfg, t, platform, r["post_id"]):
                text = re.sub(r"@\w[\w.]*", "@…", text or "").strip()     # no handles in what we keep
                if text:
                    comments.append({"platform": platform, "text": text[:600]})
        except Exception as e:
            errors.append(f"{platform}: {e}")
    votes = None
    if t["type"] == "poll":
        votes = count_votes([c["text"] for c in comments], len(t["part1"]["options_en"]))
    t = update(tid, comments=comments, votes=votes, posts=recs, collected_at=datetime.now(timezone.utc).isoformat(),
               status="collected" if t["status"] in ("waiting", "due") else t["status"])
    return {**t, "errors": errors}


# ── writing Part 2 ───────────────────────────────────────────────────────────
def part2_format(t):
    d = datetime.now()
    names = S.TAKE_NAMES.get(t["channel"], (f"{t['name']} News", f"Noticias de {t['name']}"))
    p1 = t["part1"]
    head = (f'title_en: "{names[0]}\\nYour Answers\\n<Topic in English>\\n{d:%B} {d.day}, {d.year}"\n'
            f'title_es: "{names[1]}\\nVuestras Respuestas\\n<Tema en español>\\n{d.day} de {MONTHS_ES[d.month - 1]} de {d.year}"\n'
            'intro_en: "<one sentence: two days ago we asked you about …, here is what you said>"\n'
            'intro_es: "…"\n')
    tail = ('outro_en: "<one sentence: thanks + invite to follow for the next debate>"\noutro_es: "…"\n')
    if t["type"] == "poll":
        opts = "\n".join(f'  - option_en: "{o}"\n    option_es: "{p1["options_es"][i]}"\n    percent: <number>'
                         for i, o in enumerate(p1["options_en"]))
        return ("type: poll\n" + head + f'question_en: "{p1["question_en"]}"\nquestion_es: "{p1["question_es"]}"\n'
                "results:\n" + opts + '\ntakeaway_en: "<one short sentence about the result>"\ntakeaway_es: "…"\n' + tail)
    qs = "\n".join(f'  - question_en: "{q}"\n    question_es: "{p1["questions_es"][i]}"\n'
                   '    replies_en: ["…", "…", "…"]\n    replies_es: ["…", "…", "…"]'
                   for i, q in enumerate(p1["questions_en"]))
    return "type: questions\n" + head + "answers:\n" + qs + "\n" + tail


RULES = """RULES FOR PART 2
• Use ONLY what the comments say. Never invent answers or opinions nobody wrote.
• Answers are anonymous: no names, usernames or handles, and nothing that identifies a person.
• Group similar comments and rewrite them as clear, short sentences (max 16 words), keeping their meaning.
  Prefer the most repeated ideas, and show a mix of views when people disagree.
• 2–3 replies per question. If a question got fewer real answers, write fewer (at least 1), or "replies: []" if none.
• Comments like "2" or "1!" are votes for that question/option number; for open questions, use them only to
  know which question people engaged with.
• Leave out spam, insults, hate, personal attacks and misinformation (don't repeat false claims as if true).
• Keep it neutral: no political side-taking, no judgement of the commenters.
• Spanish from Spain, natural (not literal), same meaning as the English; Spanish replies use "dijisteis/opináis".
• Poll: copy the percentages exactly as given; do not change them.
• Reply with YAML only, exactly in the format given (no code fences, no comments)."""


def prompt_for(t):
    p1 = t["part1"]
    lines = [f"YOUR TAKE — PART 2 for {t['flag']} {t['name']} ({'English' if t['lang'] == 'en' else 'Spanish'} account)",
             f"Topic: {t['topic']} · Part 1 posted {t['posted_at'][:10]}", "", "PART 1 (what we asked)",
             f"Context: {p1.get('context_en')} / {p1.get('context_es')}"]
    if t["type"] == "poll":
        lines.append(f"Poll question: {p1['question_en']} / {p1['question_es']}")
        v = t.get("votes") or {"counts": [], "percent": [], "total": 0}
        for i, o in enumerate(p1["options_en"]):
            c = v["counts"][i] if i < len(v["counts"]) else 0
            pc = v["percent"][i] if i < len(v["percent"]) else 0
            lines.append(f"  {i + 1}) {o} / {p1['options_es'][i]} — {c} votes = {pc}%")
        lines.append(f"Total votes counted: {v.get('total', 0)}")
    else:
        for i, q in enumerate(p1["questions_en"], 1):
            lines.append(f"Q{i}: {q} / {p1['questions_es'][i - 1]}")
    lines += ["", f"COMMENTS ({len(t['comments'])}, anonymous)"]
    lines += [f"- [{c['platform']}] {c['text']}" for c in t["comments"]] or ["(none)"]
    lines += ["", RULES, "", "FORMAT (fill in the <…> and …)", part2_format(t)]
    return "\n".join(lines)


def check_part2(t, text):
    text = re.sub(r"^```(?:ya?ml)?\s*|\s*```\s*$", "", (text or "").strip(), flags=re.M)
    i = text.find("type:")
    text = text[i:] if i > 0 else text
    d = yaml.safe_load(text) or {}
    for k in ("title_en", "title_es", "intro_en", "intro_es", "outro_en", "outro_es"):
        if not d.get(k):
            raise ValueError(f"Part 2 is missing {k}")
    if t["type"] == "poll":
        if not d.get("results") or not d.get("question_en"):
            raise ValueError("Poll results need question_en/question_es and results")
    else:
        ans = d.get("answers") or []
        if not ans or not any((a.get("replies_en") or []) for a in ans):
            raise ValueError("Part 2 needs at least one question with replies")
        for a in ans:
            if len(a.get("replies_en") or []) != len(a.get("replies_es") or []):
                raise ValueError("Each question needs the same number of replies in EN and ES")
    return d, text.strip() + "\n"


def write_part2(vault, cfg, tid, writer=None):
    from .extras import ask_ai
    t = get(tid)
    if not t["comments"]:
        raise ValueError("No answers collected yet — press Collect answers first.")
    writer = writer or S.get_settings().get("special_writer", "gemini")
    prompt = prompt_for(t)
    if writer == "gemini":
        key = vault.get("GEMINI_API_KEY")
        model = cfg.get("ai_model_gemini", "gemini-3.5-flash-lite")
        st, _, b = S.http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                                  f"?key={urllib.parse.quote(key or '')}", {"Content-Type": "application/json"},
                          {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 3000}},
                          timeout=120)
        d = S.jbody(b) or {}
        if st != 200:
            raise Exception(f"Gemini HTTP {st}: {(d.get('error') or {}).get('message', '')[:200]}")
        out = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    else:
        out = S.claude(vault, "You write Part 2 of YOUR TAKE videos. Reply with YAML only.", prompt,
                       cfg.get("ai_model", S.DEFAULT_AI_MODEL), 3000)
    _, clean = check_part2(t, out)
    update(tid, part2_yaml=clean, status="drafted" if t["status"] != "posted" else "posted")
    return clean


# ── building the Part 2 scenes ───────────────────────────────────────────────
def scenes(t, d):
    """Part 2 YAML → a list of scene dicts for make_reels_es_en.py --scene."""
    out = []
    rib = {"ribbon_en": "YOUR ANSWERS · PART 2", "ribbon_es": "VUESTRAS RESPUESTAS · PARTE 2"}
    title = {"title_en": d["title_en"], "title_es": d["title_es"]}
    cta = {"cta_en": ["Thanks for your answers!", "Follow for the next debate"],
           "cta_es": ["¡Gracias por vuestras respuestas!", "Síguenos para el próximo debate"]}
    if d.get("type") == "poll" or t["type"] == "poll":
        res = d["results"]
        items = {L: [d[f"question_{L}"]] + [f"{r.get('percent', 0)}% · {r[f'option_{L}']}" for r in res] for L in ("en", "es")}
        voice = {}
        for L in ("en", "es"):
            v = [d[f"intro_{L}"], d[f"question_{L}"]]
            for i, r in enumerate(res):
                w = WORDS[L][i]
                v.append(f"Option {w}, {r['option_en']}: {r.get('percent', 0)} percent." if L == "en"
                         else f"Opción {w}, {r['option_es']}: {r.get('percent', 0)} por ciento.")
            v.append(f"{d.get(f'takeaway_{L}', '')} {d[f'outro_{L}']}".strip())
            voice[f"voice_{L}"] = v
        out.append({**rib, **title, **cta, "hook_en": "Poll results", "hook_es": "Resultados de la encuesta",
                    "labels": ["?"] + [str(i + 1) for i in range(len(res))],
                    "items_en": items["en"], "items_es": items["es"], **voice})
        return out
    ans = [a for a in d["answers"] if (a.get("replies_en") or [])]
    n = len(ans)
    # intro: the questions that were asked
    out.append({**rib, **title, "hook_en": "Part 2 of 2: your answers", "hook_es": "Parte 2 de 2: vuestras respuestas",
                "labels": [str(i + 1) for i in range(n)],
                "items_en": [a["question_en"] for a in ans], "items_es": [a["question_es"] for a in ans],
                "voice_en": [d["intro_en"]] + [f"Question {WORDS['en'][i]}: {a['question_en']}" for i, a in enumerate(ans)],
                "voice_es": [d["intro_es"]] + [f"Pregunta {WORDS['es'][i]}: {a['question_es']}" for i, a in enumerate(ans)]})
    # one scene per question: the question, then the most popular answers
    for i, a in enumerate(ans):
        last = i == n - 1
        sc = {**rib, **title, "hook_en": f"Question {i + 1} of {n}", "hook_es": f"Pregunta {i + 1} de {n}",
              "labels": ["?"] + ["“"] * len(a["replies_en"]),
              "items_en": [a["question_en"]] + a["replies_en"], "items_es": [a["question_es"]] + a["replies_es"]}
        for L in ("en", "es"):
            v = [f"Question {WORDS['en'][i]}." if L == "en" else f"Pregunta {WORDS['es'][i]}.", a[f"question_{L}"]]
            v += [f"{REPLY_LEADS[L][k]} {r}" for k, r in enumerate(a[f"replies_{L}"])]
            if last:
                v.append(d[f"outro_{L}"])
            sc[f"voice_{L}"] = v
        if last:
            sc.update(cta)
        out.append(sc)
    return out


def caption(t, d, lang):
    topic = topic_of(d.get(f"title_{lang}", "")) or t["topic"]
    if t["type"] == "poll":
        lines = [f"🗳️ Poll results: {topic}" if lang == "en" else f"🗳️ Resultados de la encuesta: {topic}", "",
                 d[f"question_{lang}"]]
        lines += [f"{i}) {r[f'option_{lang}']} — {r.get('percent', 0)}%" for i, r in enumerate(d["results"], 1)]
        if d.get(f"takeaway_{lang}"):
            lines += ["", d[f"takeaway_{lang}"]]
        tags = "#Poll #YourTake" if lang == "en" else "#Encuesta #TuOpinión"
    else:
        lines = [f"🗣️ Your answers: {topic} — here's what you said 👇" if lang == "en"
                 else f"🗣️ Vuestras respuestas: {topic} — esto es lo que dijisteis 👇"]
        for i, a in enumerate(d["answers"], 1):
            if not a.get(f"replies_{lang}"):
                continue
            lines += ["", f"{i}) {a[f'question_{lang}']}"] + [f"• {r}" for r in a[f"replies_{lang}"]]
        tags = "#YourTake #Debate" if lang == "en" else "#TuOpinión #Debate"
    lines += ["", "What would you add? 👇" if lang == "en" else "¿Qué añadirías tú? 👇", "", tags]
    return "\n".join(lines)


def yt_title(t, d, lang):
    topic = topic_of(d.get(f"title_{lang}", "")) or t["topic"]
    names = S.TAKE_NAMES.get(t["channel"], (t["name"], t["name"]))
    if t["type"] == "poll":
        s = (f"{topic}: poll results 🗳️ | {names[0]} {t['flag']} #Shorts" if lang == "en"
             else f"{topic}: resultados de la encuesta 🗳️ | {names[1]} {t['flag']} #Shorts")
    else:
        s = (f"{topic}: your answers 🗣️ | {names[0]} {t['flag']} #Shorts" if lang == "en"
             else f"{topic}: vuestras respuestas 🗣️ | {names[1]} {t['flag']} #Shorts")
    return s[:100]


def due_reminders(notify, telegram):
    """Remind once when a Part 2 is due."""
    items, changed = load(), False
    now = datetime.now(timezone.utc)
    for t in items:
        if t["status"] in ("posted", "skipped") or t.get("notified"):
            continue
        if S.parse_iso(t["due_at"]) <= now:
            msg = (f"🗣️ Part 2 is due: {t['flag']} {t['name']} {t['lang'].upper()} — “{t['topic']}”. "
                   "Open Newsdesk → Your take to collect the answers and make it.")
            notify("YOUR TAKE: Part 2 due", msg)
            telegram(msg)
            t["notified"], changed = True, True
            if t["status"] == "waiting":
                t["status"] = "due"
    if changed:
        save(items)
