"""
desk.py — the Telegram control desk: Newsdesk's main options from the phone, run by GitHub.

A menu that stays at the bottom of the chat:
    🔴 Special edition · 📚 Explainer · 🔎 Fact check · 📊 Charts of the Week · 📰 Daily · 🗓️ Weekly · ℹ️ Status · ✖ Cancel
Each option asks only what it needs with tap buttons (channels, languages, charts), or asks you to type
(the topic). Then: draft (Gemini, Claude as fallback) → the videos come to the chat → Post / Cancel →
posted on YouTube, Instagram and Facebook with the duplicate guard and the news-metrics records.
Daily / Weekly start the existing workflows (no approval; they skip what's already posted).

alerts.py owns the bot (one reader at a time), the state (private news-metrics branch) and the helpers;
it sets desk.A = itself before calling anything here. Public-log rule: only short status lines are printed.
"""
import glob
import json
import os
import random
import re
import subprocess
import time
import uuid
from datetime import datetime, timedelta

import yaml

A = None                                          # alerts module (set by alerts.py)
MENU = [["🔴 Special edition", "📚 Explainer"], ["🔎 Fact check", "📊 Charts of the Week"],
        ["📰 Daily", "🗓️ Weekly"], ["🧪 Trials", "ℹ️ Status"], ["✖ Cancel"]]
FLOW_OF = {"🔴 Special edition": "special", "📚 Explainer": "explainer", "🔎 Fact check": "claim",
           "📊 Charts of the Week": "charts", "📰 Daily": "daily", "🗓️ Weekly": "weekly"}
CH = {"australia": "🇦🇺 Australia", "world": "🌍 World", "europe": "🇪🇺 Europe"}
NEWS_REPO = {"australia": "aus_news", "world": "world_news", "europe": "europe_news"}
NAMES = {"australia": ("Australia News", "Noticias de Australia"), "world": ("World News", "Noticias del Mundo"),
         "europe": ("Europe News", "Noticias de Europa")}
FACT_CHECKERS = {"australia": ["AAP FactCheck", "RMIT FactLab"], "world": ["AFP Fact Check", "Full Fact"],
                 "europe": ["AFP Fact Check", "Full Fact"], "es": ["Maldita", "Newtral", "EFE Verifica"]}
FLOW_TTL_H = 6


# ── small helpers ────────────────────────────────────────────────────────────
def menu(text="What would you like to do?"):
    return A.tg("sendMessage", chat_id=A.CHAT, text=text,
                reply_markup={"keyboard": [[{"text": t} for t in row] for row in MENU],
                              "resize_keyboard": True, "is_persistent": True})


def kb(rows):
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in rows]}


def ask_buttons(text, rows):
    return A.tg("sendMessage", chat_id=A.CHAT, text=text[:4000], reply_markup=kb(rows))


def edit_buttons(msg_id, rows):
    if msg_id:
        A.tg("editMessageReplyMarkup", chat_id=A.CHAT, message_id=msg_id, reply_markup=kb(rows))


def flow(st):
    f = st.get("desk")
    if f and A.now() - A.parse_iso(f["t"]) > timedelta(hours=FLOW_TTL_H):
        st.pop("desk", None)
        return None
    return f


def start_flow(st, name, **data):
    st["desk"] = {"flow": name, "step": None, "t": A.iso(), **data}
    return st["desk"]


def toggles(prefix, options, chosen, done_label="Next ➡️", allow_all=False):
    row = [((("✅ " if k in chosen else "⬜ ") + lab), f"{prefix}:{k}") for k, lab in options]
    rows = [row[i:i + 3] for i in range(0, len(row), 3)]
    if allow_all:
        rows.append([("All", f"{prefix}:*")])
    rows.append([(done_label, f"{prefix}:>"), ("✖ Cancel", "x:cancel")])
    return rows


def date_texts():
    d = datetime.now(A.SYD)
    return f"{d:%B} {d.day}, {d.year}", f"{d.day} de {A.MONTHS_ES[d.month - 1]} de {d.year}"


def strip_yaml(out):
    out = re.sub(r"</?cite\b[^>]*>", "", out)
    block = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", out, flags=re.S)
    out = block.group(1) if block else out
    i = out.find("title_en:")
    return out[i:] if i > 0 else out


def researched(system, user, topic, region="", prefer="gemini"):
    """Gemini with Google Search (+ the newest GNews articles); Claude as fallback."""
    arts = []
    try:
        for a in A.gnews("search", q=f"{topic}{region}", sortby="publishedAt"):
            arts.append(f"- {a.get('publishedAt', '')[:10]} · {(a.get('source') or {}).get('name', '')}: "
                        f"{a.get('title', '')} — {a.get('description', '')}")
    except Exception as e:
        A.log(f"GNews (topic): {type(e).__name__}")
    if arts:
        user += "\n\nLatest articles (use them, and search for more if you can):\n" + "\n".join(arts)
    out, who = A.ask(user, system, search=True, prefer=prefer)
    return out, who


# ── writers (same instructions as Newsdesk) ──────────────────────────────────
def write_explainer(ck, topic, prefer="gemini"):
    en_name, es_name = NAMES[ck]
    date_en, date_es = date_texts()
    region = {"australia": "Australia and Australians", "europe": "Europe and the EU"}.get(ck, "the world")
    system = (
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
    out, who = researched(system, f"Topic: {topic}\nToday's date: {date_en}.", topic,
                          {"australia": " Australia", "europe": " Europe"}.get(ck, ""), prefer)
    y = yaml.safe_load(strip_yaml(out)) or {}
    for k in ("title_en", "title_es", "intro_en", "intro_es", "facts_en", "facts_es"):
        if not y.get(k):
            raise ValueError(f"draft is missing {k}")
    if not 3 <= len(y["facts_en"]) <= 5 or len(y["facts_en"]) != len(y["facts_es"]):
        raise ValueError("draft needs 3–5 facts in both languages")
    return y, who


def write_claim(ck, topic, prefer="gemini"):
    en_name, es_name = NAMES[ck]
    date_en, date_es = date_texts()
    checkers = FACT_CHECKERS.get(ck, FACT_CHECKERS["world"])
    region = {"australia": "Australia", "europe": "Europe"}.get(ck, "the world")
    system = (
        f"You write 'Fact check' videos for the short-video news channel '{en_name}' / '{es_name}' ({region}). "
        "Search the web. Use ONLY a claim that a real, established fact-checker has ALREADY published a ruling on, "
        f"preferably {', '.join(checkers)} (for the Spanish side, if one of {', '.join(FACT_CHECKERS['es'])} ruled on "
        "the same claim, name it in checker_es). Never rule on a claim yourself and never invent a verdict, a "
        "checker or a link: copy the fact-checker's own rating and give the exact URL of their article. Describe "
        "the claim neutrally (\"Posts claim that…\"), without repeating insults or naming private people. Avoid "
        "claims about ongoing elections. Points: 2 or 3 short facts (max 16 words) from the fact-check that "
        "explain the verdict. Spanish from Spain, same meaning.\n\n"
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
    out, who = researched(system, user, topic or "fact check", "", prefer)
    y = yaml.safe_load(strip_yaml(out)) or {}
    for k in ("title_en", "title_es", "claim_en", "claim_es", "verdict_en", "verdict_es", "points_en", "points_es",
              "checker", "url"):
        if not y.get(k):
            raise ValueError(f"draft is missing {k}")
    if not 2 <= len(y["points_en"]) <= 3 or len(y["points_en"]) != len(y["points_es"]) or \
            not str(y["url"]).startswith("http"):
        raise ValueError("draft needs 2–3 points in both languages and the fact-check link")
    return y, who


# ── Charts of the Week: candidates from the week's day files ─────────────────
def nums(text):
    out = []
    for m in re.findall(r"-?\d[\d.,]*", str(text)):
        m = m.rstrip(".,")
        try:
            out.append(float(m.replace(",", ".")) if m.count(",") + m.count(".") <= 1 else float(re.sub(r"[.,]", "", m)))
        except ValueError:
            pass
    return out


def spec_values(sp):
    k = sp.get("chart")
    if k == "slope":
        return [float(v) for x in sp.get("series") or [] for v in (x.get("from"), x.get("to")) if v is not None]
    if k == "bignum":
        return [float(v) for v in (sp.get("value"), sp.get("previous"), sp.get("change")) if v is not None]
    return [float(v) for v in sp.get("values") or []]


def chart_candidates(ck):
    import urllib.request
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    base = f"https://raw.githubusercontent.com/{A.OWNER}/{NEWS_REPO[ck]}/main"
    cands = []
    for day in days:
        files = {}
        for l in ("EN", "ES", "PT", "ID"):
            try:
                with urllib.request.urlopen(f"{base}/{day}News{l}.yml", timeout=20) as r:
                    files[l.lower()] = yaml.safe_load(r.read().decode("utf-8")) or {}
            except Exception:
                files[l.lower()] = {}
        en = files.get("en") or {}
        if not en.get("date") or str(en["date"]) < (datetime.now(A.SYD) - timedelta(days=7)).strftime("%Y-%m-%d"):
            continue
        for i, it in enumerate(en.get("news") or []):
            v = (it or {}).get("visual") or {}
            if v.get("type") != "chart":
                continue
            spec = {k: x for k, x in v.items() if k != "fallback"}
            for l in ("es", "pt", "id"):
                items = (files.get(l) or {}).get("news") or []
                ov = ((items[i] if i < len(items) else {}) or {}).get("visual") or {}
                if ov.get("type") == "chart" and ov.get("chart") == spec.get("chart"):
                    spec.update({k: x for k, x in ov.items() if k.endswith(f"_{l}")})
                    if spec.get("series") and ov.get("series"):
                        spec["series"] = [{**a, **{k: x for k, x in b.items() if k.endswith(f"_{l}")}}
                                          for a, b in zip(spec["series"], ov["series"])]
            cands.append({"date": str(en["date"]), "title": spec.get("title_en") or it.get("title", ""),
                          "chart": spec.get("chart"), "spec": spec, "views": 0,
                          "langs": [l for l in A.LANGS if l == "en" or spec.get(f"title_{l}")]})
    try:                                          # views of that day's daily video (news-metrics, private)
        posts = json.loads(A.gh_raw("index/posts.json"))
        m = (json.loads(A.gh_raw("index/latest.json")) or {}).get("posts", {})
        for c in cands:
            c["views"] = sum((m.get(f"{p['platform']}:{p['post_id']}") or {}).get("views") or 0 for p in posts
                             if p.get("channel") == ck and p.get("kind") == "daily" and not p.get("trial")
                             and str(p.get("posted_at", ""))[:10] == c["date"])
    except Exception:
        pass
    cands.sort(key=lambda c: (c["views"], c["date"]), reverse=True)
    return cands[:10]


def write_week(picked, langs, prefer="gemini"):
    blocks = []
    for n, sp in enumerate(picked, 1):
        labels = sp.get("labels_en") or [x.get("label_en", "") for x in sp.get("series") or []]
        blocks.append(f"CHART {n}: {sp.get('title_en', '')} ({sp.get('chart')}, unit {sp.get('unit', '')}, "
                      f"source {sp.get('source', '')})\nlabels: {labels}\nvalues: {spec_values(sp)}")
    out, _ = A.ask("You write the voice-over of a 'Charts of the Week' news short. Use ONLY the numbers given for each "
                   "chart (you may round to their decimals); never add other figures, causes or forecasts. Neutral.\n"
                   "1) claim: ONE line (max 10 words) with the most striking fact of all the charts — the video opens "
                   "with it.\n2) For each chart, ONE sentence (max 18 words) with its main point.\n"
                   "Languages: English (en), Spanish from Spain (es), Brazilian Portuguese (pt), Indonesian (id); numbers "
                   "in each language's format. Return ONLY JSON: {\"claim\": {\"en\": \"...\", \"es\": \"...\", \"pt\": "
                   "\"...\", \"id\": \"...\"}, \"lines\": [{\"en\": \"...\", ...one per chart}]}\n\n" + "\n\n".join(blocks),
                   prefer=prefer, max_tokens=2000)
    j = A.json_from(out)
    y = {"type": "charts_week", **{f"claim_{l}": str((j.get("claim") or {}).get(l) or "") for l in langs}, "charts": []}
    lines = j.get("lines") or []
    for n, sp in enumerate(picked):
        sc = dict(sp)
        for l in langs:
            ln = str(((lines[n] if n < len(lines) else {}) or {}).get(l) or "")
            if ln:
                sc[f"lines_{l}"] = [ln]
        y["charts"].append(sc)
    warn = []
    allv = [v for sc in y["charts"] for v in spec_values(sc)]
    ok = lambda x, vals: any(abs(x - v) < 0.051 or round(x) == round(v) for v in vals) or 1990 <= x <= 2100
    for l in langs:
        if not y.get(f"claim_{l}"):
            warn.append(f"{l.upper()}: no claim")
        elif any(not ok(x, allv) for x in nums(y[f"claim_{l}"])):
            warn.append(f"{l.upper()} claim: a number isn't in the charts")
        for n, sc in enumerate(y["charts"], 1):
            for ln in sc.get(f"lines_{l}") or [""]:
                if not ln:
                    warn.append(f"chart {n} {l.upper()}: no sentence")
                elif any(not ok(x, spec_values(sc)) for x in nums(ln)):
                    warn.append(f"chart {n} {l.upper()}: “{ln}” has a number not in its data")
    return y, warn


# ── scene videos (explainer / fact check), built like Newsdesk ───────────────
def explainer_task(ck, lang, d):
    lines = [l.strip() for l in str(d[f"title_{lang}"]).split("\n") if l.strip()]
    topic = lines[1] if len(lines) >= 3 else (lines[-1] if lines else "")
    facts = [str(x) for x in d[f"facts_{lang}"]]
    outro = ("Follow for more explainers, and ask your questions in the comments." if lang == "en"
             else "Síguenos para más claves y deja tus preguntas en los comentarios.")
    scene = {"accent": [0, 128, 140], "ribbon_en": "EXPLAINER", "ribbon_es": "LAS CLAVES",
             "hook_en": "What you need to know", "hook_es": "Lo que debes saber",
             "title_en": d["title_en"], "title_es": d["title_es"], "labels": [str(i + 1) for i in range(len(facts))],
             "items_en": [str(x) for x in d["facts_en"]], "items_es": [str(x) for x in d["facts_es"]],
             "voice_en": [d["intro_en"]] + [str(x) for x in d["facts_en"]] + [outro if lang == "en" else ""],
             "voice_es": [d["intro_es"]] + [str(x) for x in d["facts_es"]] + [outro if lang == "es" else ""],
             "cta_en": ["Follow for more explainers", "Questions? Ask in the comments"],
             "cta_es": ["Síguenos para más claves", "¿Dudas? Pregunta en los comentarios"]}
    flag = CH[ck].split()[0]
    if lang == "en":
        cap = ([f"📚 Explainer: {topic} — what you need to know 👇", ""] + [f"{i}) {f}" for i, f in enumerate(facts, 1)]
               + ["", "Questions? Ask in the comments 👇", "", "#Explainer #News"])
        title, comment = f"{topic}: what you need to know 📚 | {NAMES[ck][0]} {flag} #Shorts", "What else would you like explained? 👇"
    else:
        cap = ([f"📚 Las claves: {topic} 👇", ""] + [f"{i}) {f}" for i, f in enumerate(facts, 1)]
               + ["", "¿Dudas? Pregunta en los comentarios 👇", "", "#LasClaves #Noticias"])
        title, comment = f"{topic}: las claves 📚 | {NAMES[ck][1]} {flag} #Shorts", "¿Qué más te gustaría que te explicáramos? 👇"
    return {"type": "scene", "kind": "explainer", "channel": ck, "lang": lang, "scene": scene, "caption": "\n".join(cap),
            "title": title[:100], "comment": comment, "meta": {"topic": topic, "format": "explainer"},
            "label": f"{flag} Explainer {lang.upper()}"}


def claim_task(ck, lang, d):
    lines = [l.strip() for l in str(d[f"title_{lang}"]).split("\n") if l.strip()]
    topic = lines[1] if len(lines) >= 3 else (lines[-1] if lines else "")
    checker = d.get("checker_es") if lang == "es" and d.get("checker_es") else d["checker"]
    url = d.get("url_es") if lang == "es" and d.get("url_es") else d["url"]
    v = str(d[f"verdict_{lang}"]).strip()
    low = v.lower()
    icon = ("✓" if low in ("true", "verdadero", "correct", "correcto", "cierto") else
            "✗" if any(w in low for w in ("false", "fals", "fake", "bulo", "fabricated", "inventad")) else "!")
    pts = [str(x) for x in d[f"points_{lang}"]]
    flag = CH[ck].split()[0]
    if lang == "en":
        items = [f"Claim: {d['claim_en']}", f"Verdict: {v} — {checker}"] + pts
        voice = [f"Fact check: {topic}.", f"The claim. {d['claim_en']}", f"The verdict from {checker}: {v}."] + pts + \
                ["Always check before you share. Follow for more fact checks."]
        cap = ([f"🔎 Fact check: {topic}", "", f"The claim: {d['claim_en']}", f"Verdict ({checker}): {v}", ""]
               + [f"{i}) {x}" for i, x in enumerate(pts, 1)]
               + ["", f"Source: {checker} — {url}", "", "Always check before you share. Seen a claim we should check? 👇",
                  "", "#FactCheck #News"])
        title, comment = f"Fact check: {topic} 🔎 | {NAMES[ck][0]} {flag} #Shorts", "Seen a claim you'd like us to check? Tell us 👇"
    else:
        items = [f"Lo que circula: {d['claim_es']}", f"Veredicto: {v} — {checker}"] + pts
        voice = [f"Verificación: {topic}.", f"Lo que circula. {d['claim_es']}", f"El veredicto de {checker}: {v}."] + pts + \
                ["Comprueba siempre antes de compartir. Síguenos para más verificaciones."]
        cap = ([f"🔎 Verificación: {topic}", "", f"Lo que circula: {d['claim_es']}", f"Veredicto ({checker}): {v}", ""]
               + [f"{i}) {x}" for i, x in enumerate(pts, 1)]
               + ["", f"Fuente: {checker} — {url}", "", "Comprueba antes de compartir. ¿Has visto un bulo que debamos verificar? 👇",
                  "", "#Verificación #Noticias"])
        title, comment = f"Verificación: {topic} 🔎 | {NAMES[ck][1]} {flag} #Shorts", "¿Has visto algo que quieras que verifiquemos? Cuéntanos 👇"
    scene = {"accent": [217, 119, 6], "ribbon_en": "FACT CHECK", "ribbon_es": "VERIFICACIÓN",
             "hook_en": "What the fact-checkers found", "hook_es": "Lo que dicen los verificadores",
             "title_en": d["title_en"], "title_es": d["title_es"],
             "labels": ["?", icon] + [str(i + 1) for i in range(len(pts))],
             "items_en": items, "items_es": items, "voice_en": voice, "voice_es": voice,
             "cta_en": ["Check before you share", "Follow for more fact checks"],
             "cta_es": ["Comprueba antes de compartir", "Síguenos para más verificaciones"]}
    return {"type": "scene", "kind": "claim", "channel": ck, "lang": lang, "scene": scene, "caption": "\n".join(cap),
            "title": title[:100], "comment": comment,
            "meta": {"topic": topic, "checker": checker, "checker_url": url, "verdict": v},
            "label": f"{flag} Fact check {lang.upper()}"}


# ── the conversation ─────────────────────────────────────────────────────────
NOT_MINE = object()


def route(st, kind, data, obj):
    """Desk part of handle(): returns NOT_MINE (alerts.py handles it), None, or ('post', job_id)."""
    msg = (obj.get("message") or {}) if kind == "tap" else {}
    if kind == "text":
        t = data.strip()
        low = t.lower()
        if low in ("/start", "menu", "/menu", "help", "/help"):
            menu("🤖 Newsdesk on Telegram — pick an option below.")
            return None
        if t in ("🧪 Trials", "/trials", "trials"):
            return trials_status()
        if t in ("ℹ️ Status", "/status", "status"):
            status(st)
            return None
        if t in ("✖ Cancel", "cancel", "/cancel"):
            st.pop("desk", None)
            menu("✖ Cancelled.")
            return None
        if t in FLOW_OF:
            return begin(st, FLOW_OF[t])
        f = flow(st)
        if f and f.get("step") == "topic":                      # the typed topic
            return got_topic(st, f, t)
        return NOT_MINE
    parts = data.split(":")
    if parts[0] == "t":                                          # trial batch details (read-only)
        return trial_details(parts[1], msg)
    if parts[0] not in ("c", "l", "w", "g", "x", "k"):
        return NOT_MINE
    f = flow(st)
    if parts[0] == "x":
        st.pop("desk", None)
        A.drop_buttons(msg)
        menu("✖ Cancelled.")
        return None
    if not f:
        A.drop_buttons(msg)
        A.say("⌛ That choice has expired — start again from the menu.")
        return None
    mid = msg.get("message_id")
    if parts[0] == "c":                                          # channels
        return pick_channel(st, f, parts[1], mid, msg)
    if parts[0] == "l":                                          # languages
        return pick_lang(st, f, parts[1], mid, msg)
    if parts[0] == "w":                                          # charts of the week: which charts
        return pick_chart(st, f, parts[1], mid, msg)
    if parts[0] == "k":                                          # fact check: find one
        if f["flow"] == "claim" and f.get("step") == "topic":
            A.drop_buttons(msg)
            return got_topic(st, f, "")
        return None
    if parts[0] == "g":                                          # go: make / rewrite / post now
        return go(st, f, parts[1], msg)
    return None


def begin(st, name):
    multi = name in ("special", "daily", "weekly")
    f = start_flow(st, name, channels=[], langs=[], step="channels")
    title = {"special": "🔴 Special edition — which channels?", "explainer": "📚 Explainer — which channel?",
             "claim": "🔎 Fact check — which channel?", "charts": "📊 Charts of the Week — which channel?",
             "daily": "📰 Daily — which channels?", "weekly": "🗓️ Weekly — which channels?"}[name]
    m = ask_buttons(title, toggles("c", list(CH.items()), [], allow_all=multi) if multi else
                    [[(lab, f"c:{k}")] for k, lab in CH.items()] + [[("✖ Cancel", "x:cancel")]])
    f["msg"] = (m or {}).get("message_id")
    return None


def lang_options(f):
    if f["flow"] in ("explainer", "claim"):
        return [("en", "EN"), ("es", "ES")]                      # scene videos: English and Spanish only
    if f["flow"] == "charts":
        ok = set(A.LANGS)
        for sp in f.get("picked") or []:
            ok &= {l for l in A.LANGS if l == "en" or sp.get(f"title_{l}")}
        return [(l, l.upper()) for l in A.LANGS if l in ok]
    return [(l, l.upper()) for l in A.LANGS]


def pick_channel(st, f, k, mid, msg):
    multi = f["flow"] in ("special", "daily", "weekly")
    if k == ">":
        if not f["channels"]:
            A.tg("sendMessage", chat_id=A.CHAT, text="Pick at least one channel.")
            return None
        A.drop_buttons(msg)
        return after_channels(st, f)
    if k == "*":
        f["channels"] = list(CH)
    elif k in CH:
        if multi:
            f["channels"] = [c for c in CH if (c in f["channels"]) != (c == k)]
        else:
            f["channels"] = [k]
            A.drop_buttons(msg)
            return after_channels(st, f)
    edit_buttons(mid, toggles("c", list(CH.items()), f["channels"], allow_all=True))
    return None


def after_channels(st, f):
    if f["flow"] == "charts":
        ck = f["channels"][0]
        A.say(f"🔎 Looking for this week's charts on {CH[ck]}…")
        cands = chart_candidates(ck)
        if len(cands) < 2:
            st.pop("desk", None)
            menu(f"Only {len(cands)} chart stor{'y' if len(cands) == 1 else 'ies'} this week on {CH[ck]} — "
                 "Charts of the Week needs at least 2.")
            return None
        f["cands"], f["order"], f["step"] = cands, [], "charts"
        text = f"📊 This week's charts on {CH[ck]} (by views of that day's video). Tap 2–5 in the order you want:\n\n" + \
               "\n".join(f"{i + 1}. {c['title']} — {c['chart']} · {c['date'][5:]} · "
                         f"{c['views']:,} views · {' '.join(l.upper() for l in c['langs'])}" for i, c in enumerate(cands))
        m = ask_buttons(text, chart_rows(f))
        f["msg"] = (m or {}).get("message_id")
        return None
    f["step"] = "langs"
    opts = lang_options(f)
    f["langs"] = [l for l, _ in opts] if f["flow"] in ("explainer", "claim") else ["en", "es"]
    m = ask_buttons("Which languages?", toggles("l", opts, f["langs"]))
    f["msg"] = (m or {}).get("message_id")
    return None


def chart_rows(f):
    row = []
    for i in range(len(f["cands"])):
        n = f["order"].index(i) + 1 if i in f["order"] else None
        row.append((f"{'①②③④⑤'[n - 1]} {i + 1}" if n else f"{i + 1}", f"w:{i}"))
    rows = [row[i:i + 5] for i in range(0, len(row), 5)]
    rows.append([("Next ➡️", "w:>"), ("✖ Cancel", "x:cancel")])
    return rows


def pick_chart(st, f, k, mid, msg):
    if k == ">":
        if not 2 <= len(f["order"]) <= 5:
            A.tg("sendMessage", chat_id=A.CHAT, text="Pick 2 to 5 charts.")
            return None
        A.drop_buttons(msg)
        f["picked"] = [f["cands"][i]["spec"] for i in f["order"]]
        f["step"] = "langs"
        opts = lang_options(f)
        f["langs"] = [l for l, _ in opts if l in ("en", "es")]
        m = ask_buttons("Which languages? (only those every chosen chart has)", toggles("l", opts, f["langs"]))
        f["msg"] = (m or {}).get("message_id")
        return None
    i = int(k)
    f["order"] = [x for x in f["order"] if x != i] if i in f["order"] else (f["order"] + [i])[:5]
    edit_buttons(mid, chart_rows(f))
    return None


def pick_lang(st, f, k, mid, msg):
    opts = lang_options(f)
    if k == ">":
        if not f["langs"]:
            A.tg("sendMessage", chat_id=A.CHAT, text="Pick at least one language.")
            return None
        A.drop_buttons(msg)
        if f["flow"] in ("daily", "weekly"):
            pairs = ", ".join(f"{CH[c]} {'/'.join(l.upper() for l in f['langs'])}" for c in f["channels"])
            ask_buttons(f"{'📰 Daily' if f['flow'] == 'daily' else '🗓️ Weekly'} now: {pairs}?\n"
                        "Anything already posted today is skipped.", [[("▶️ Post now", "g:now"), ("✖ Cancel", "x:cancel")]])
            f["step"] = "confirm"
            return None
        if f["flow"] == "charts":
            return draft_charts(st, f)
        f["step"] = "topic"
        if f["flow"] == "claim":
            ask_buttons("🔎 Type the claim or topic to check — or let it find this week's most shared claim:",
                        [[("🔍 Find one", "k:find"), ("✖ Cancel", "x:cancel")]])
        else:
            A.say("✍️ Type the topic (a few words).")
        return None
    if k in dict(opts):
        f["langs"] = [l for l, _ in opts if (l in f["langs"]) != (l == k)]
    edit_buttons(mid, toggles("l", opts, f["langs"]))
    return None


def got_topic(st, f, topic):
    f["topic"], f["step"] = topic, "drafted"
    return draft(st, f, "gemini")


def draft(st, f, prefer):
    """Write the texts (Gemini, Claude as fallback) and show them with Make / Rewrite / Cancel."""
    topic = f.get("topic", "")
    A.say(f"✍️ Writing{' with Claude' if prefer == 'claude' else ''}… (about a minute"
          f"{' per channel' if len(f['channels']) > 1 else ''})")
    f["drafts"], previews = {}, []
    for ck in f["channels"]:
        try:
            if f["flow"] == "special":
                y, who = A.write_special(ck, topic, topic, prefer)
                if any(l in f["langs"] for l in ("pt", "id")):
                    A.add_extra(ck, y, "claude" if who == "claude" else "gemini")
                f["drafts"][ck] = y
                previews.append(A.preview({"channel": ck, "yaml": y, "writer": "Claude" if who == "claude" else "Gemini"}))
            elif f["flow"] == "explainer":
                y, who = write_explainer(ck, topic, prefer)
                f["drafts"][ck] = y
                previews.append(f"📚 {CH[ck]} explainer ({who.capitalize()})\n" + "\n\n".join(
                    f"{A.FLAGS[l]} {' · '.join(str(y[f'title_{l}']).split(chr(10))[1:2])}\n" +
                    "\n".join(f"{i}. {x}" for i, x in enumerate(y[f"facts_{l}"], 1)) for l in f["langs"]))
            elif f["flow"] == "claim":
                y, who = write_claim(ck, topic, prefer)
                f["drafts"][ck] = y
                previews.append(f"🔎 {CH[ck]} fact check ({who.capitalize()})\n" + "\n\n".join(
                    f"{A.FLAGS[l]} {y[f'claim_{l}']}\nVerdict: {y[f'verdict_{l}']} — "
                    f"{(y.get('checker_es') if l == 'es' and y.get('checker_es') else y['checker'])}\n" +
                    "\n".join(f"• {x}" for x in y[f"points_{l}"]) for l in f["langs"]) +
                    f"\n\n🔗 {y['url']}" + (f"\n🔗 {y['url_es']}" if y.get("url_es") and y.get("url_es") != y["url"] else "") +
                    "\nOpen the link and confirm the verdict before posting.")
        except Exception as e:
            A.log(f"Draft failed ({type(e).__name__})")
            previews.append(f"⚠️ {CH[ck]}: couldn't write it ({type(e).__name__}) — try Rewrite, or another topic.")
    for p in previews:
        A.say(p)
    if not f["drafts"]:
        f["step"] = "topic"
        A.say("Type another topic, or ✖ Cancel.")
        return None
    ask_buttons("Check the facts, then:", [[("🎬 Make the videos", "g:make")],
                                           [("✍️ Rewrite with Claude", "g:rewrite"), ("✖ Cancel", "x:cancel")]])
    return None


def draft_charts(st, f):
    A.say("✍️ Writing the claim and the lines…")
    try:
        y, warn = write_week(f["picked"], f["langs"])
    except Exception as e:
        A.say(f"⚠️ Couldn't write it ({type(e).__name__}). Try again from the menu.")
        st.pop("desk", None)
        return None
    f["drafts"] = {f["channels"][0]: y}
    txt = "📊 Charts of the Week\n\n" + "\n\n".join(
        f"{A.FLAGS[l]} {y.get(f'claim_{l}', '')}\n" + "\n".join(f"{n}. {(sc.get(f'lines_{l}') or [''])[0]}"
                                                                for n, sc in enumerate(y['charts'], 1)) for l in f["langs"])
    if warn:
        txt += "\n\n⚠️ " + "\n⚠️ ".join(warn[:6])
    A.say(txt)
    ask_buttons("Then:", [[("🎬 Make the videos", "g:make")], [("✖ Cancel", "x:cancel")]])
    f["step"] = "drafted"
    return None


def go(st, f, k, msg):
    A.drop_buttons(msg)
    if k == "rewrite" and f["flow"] in ("special", "explainer", "claim"):
        return draft(st, f, "claude")
    if k == "now" and f["flow"] in ("daily", "weekly"):
        res = dispatch(f["flow"], f["channels"], f["langs"])
        st.pop("desk", None)
        menu("▶️ Started:\n" + "\n".join(res) + "\n\nGitHub posts them in the next minutes (already-posted ones are skipped).")
        return None
    if k != "make" or not f.get("drafts"):
        return None
    tasks = []
    for ck, y in f["drafts"].items():
        for lang in f["langs"]:
            if f["flow"] == "special" and y.get(f"news_{lang}"):
                tasks.append({"type": "special", "kind": "special", "channel": ck, "lang": lang, "yaml": y,
                              "label": f"{CH[ck].split()[0]} Special {lang.upper()}"})
            elif f["flow"] == "explainer":
                tasks.append(explainer_task(ck, lang, y))
            elif f["flow"] == "claim":
                tasks.append(claim_task(ck, lang, y))
            elif f["flow"] == "charts":
                tasks.append({"type": "charts", "kind": "charts_week", "channel": ck, "lang": lang, "yaml": y,
                              "title": f"{y.get(f'claim_{lang}', '')} 📊 | {NAMES[ck][0 if lang == 'en' else 1]} "
                                       f"{CH[ck].split()[0]} #Shorts"[:100],
                              "label": f"{CH[ck].split()[0]} Charts of the Week {lang.upper()}"})
    if not tasks:
        A.say("Nothing to make (no language has a text).")
        return None
    jid = new_job(st, tasks, f["flow"])
    st.pop("desk", None)
    mins = 4 + 2 * len(tasks)
    menu(f"🎬 Making {len(tasks)} video{'s' if len(tasks) > 1 else ''} — they'll come here to approve (about {mins}–{mins + 5} min).")
    return ("post", jid)


def new_job(st, tasks, label):
    jid = uuid.uuid4().hex[:6]
    st.setdefault("jobs", {})[jid] = {"t": A.iso(), "label": label, "tasks": tasks, "status": "queued"}
    return jid


def job_from_breaking(st, did):
    d = st["drafts"][did]
    tasks = [{"type": "special", "kind": "special", "channel": d["channel"], "lang": l, "yaml": d["yaml"],
              "label": f"{CH[d['channel']].split()[0]} Special {l.upper()}"} for l in d["langs"]]
    return new_job(st, tasks, "breaking")


# ── daily / weekly: start the existing workflows ─────────────────────────────
def dispatch(kind, channels, langs):
    reels = {"australia": "australia-news-reels", "world": "world-news-reels", "europe": "europe-news-reels"}
    out = []
    for ck in channels:
        for l in langs:
            repo = reels[ck] if l in ("en", "es") else NEWS_REPO[ck]
            wf = f"{kind}_{l}.yml"
            try:
                A.gh("POST", f"/repos/{A.OWNER}/{repo}/actions/workflows/{wf}/dispatches", {"ref": "main"})
                out.append(f"✅ {CH[ck]} {l.upper()}")
            except Exception as e:
                code = getattr(e, "code", "")
                out.append(f"⚠️ {CH[ck]} {l.upper()} not started ({code or type(e).__name__})")
    return out


def status(st):
    last = st.get("last_check")
    last_txt = f"{A.parse_iso(last).astimezone(A.SYD):%a %H:%M}" if last else "none yet"
    jobs = sorted((st.get("jobs") or {}).items(), key=lambda kv: kv[1]["t"], reverse=True)[:5]
    f = flow(st)
    menu("ℹ️ Status\n"
         f"• Breaking checks at {', '.join(f'{h}:00' for h in A.ALERT_HOURS)} Sydney · last {last_txt}\n"
         f"• I read this chat every ~10 min, and within seconds for {A.LISTEN_MIN} min after you use it\n"
         + (f"• In progress: {f['flow']} ({f.get('step')})\n" if f else "")
         + trials_summary_line()
         + ("• Recent jobs:\n" + "\n".join(f"   {A.parse_iso(j['t']).astimezone(A.SYD):%a %H:%M} {j['label']}: {j['status']} "
                                         f"({len(j['tasks'])} video{'s' if len(j['tasks']) > 1 else ''})" for _, j in jobs)
            if jobs else "• No jobs yet"))


# ── job 2: make the videos, approve, post ────────────────────────────────────
def reels_dir(ck):
    return f"reels_{ck}"


def render(task, logf):
    """Make one video. Returns (video, caption) paths, or None."""
    ck, lang = task["channel"], task["lang"]
    cwd = reels_dir(ck)
    c = A.CHANNELS[ck]
    v, m = A.pick(c["video"], "mp4", cwd), A.pick(c["music"], "mp3", cwd)
    if not (v and m):
        return None
    out = f"tg_{task['kind']}_{lang}"
    subprocess.run(["rm", "-rf", os.path.join(cwd, out)])
    if task["type"] == "special":
        with open(os.path.join(cwd, "tg_special.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(task["yaml"], f, allow_unicode=True, sort_keys=False, width=1000)
        ok = A.sh(["python", "make_reels_es_en.py", "--yml", "tg_special.yml", "--video", v, "--music", m, "--lang", lang,
                   "--output-dir", out, "--no-week-label", "--title-color", "red"], logf, cwd=cwd)
        vids, caps = glob.glob(f"{cwd}/{out}/reel_{lang}_*.mp4"), glob.glob(f"{cwd}/{out}/caption_{lang}_*.txt")
        return (vids[0], caps[0]) if ok and vids and caps else None
    if task["type"] == "charts":
        with open(os.path.join(cwd, "tg_charts.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(task["yaml"], f, allow_unicode=True, sort_keys=False, width=1000)
        ok = A.sh(["python", "chart_reel.py", "--yml", "tg_charts.yml", "--video", v, "--music", m, "--lang", lang,
                   "--output-dir", out], logf, cwd=cwd)
        vid, cap = f"{cwd}/{out}/reel_{lang}_charts.mp4", f"{cwd}/{out}/caption_{lang}_charts.txt"
        return (vid, cap) if ok and os.path.exists(vid) else None
    # scene (explainer / fact check): one scene with voice, then the music under it — like Newsdesk
    os.makedirs(os.path.join(cwd, out), exist_ok=True)
    with open(os.path.join(cwd, out, "scene0.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(task["scene"], f, allow_unicode=True, sort_keys=False, width=1000)
    ok = A.sh(["python", "make_reels_es_en.py", "--yml", f"{out}/scene0.yml", "--video", v, "--lang", lang,
               "--output-dir", f"{out}/s0", "--scene"], logf, cwd=cwd)
    srcs = sorted(glob.glob(f"{cwd}/{out}/s0/reel_{lang}_*.mp4"))
    if not (ok and srcs):
        return None
    final = f"{cwd}/{out}/reel_{lang}_{task['kind']}.mp4"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", srcs[0]],
                               capture_output=True, text=True).stdout.strip() or 0)
    ok = A.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", srcs[0], "-stream_loop", "-1", "-i", os.path.join(cwd, m),
               "-filter_complex", "[0:a]aresample=44100,aformat=channel_layouts=stereo[v];"
               "[1:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.04[m];"
               f"[v][m]amix=inputs=2:duration=first:normalize=0,afade=t=out:st={max(0, dur - 2):.2f}:d=2[a]",
               "-map", "0:v", "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
               "-movflags", "+faststart", final], logf, cwd=".")
    if not ok:
        return None
    cover = None
    for cp in sorted(glob.glob(f"{cwd}/{out}/s0/cover_*.jpg"))[:1]:
        cover = f"cover_{lang}_{task['kind']}.jpg"
        with open(cp, "rb") as a_, open(f"{cwd}/{out}/{cover}", "wb") as b_:
            b_.write(a_.read())
    cap = f"{cwd}/{out}/caption_{lang}_{task['kind']}.txt"
    with open(cap, "w", encoding="utf-8") as f:
        f.write(task["caption"] + "\n")
    with open(f"{cwd}/{out}/meta_{lang}_{task['kind']}.json", "w", encoding="utf-8") as f:
        json.dump({"channel": ck, "lang": lang, "kind": task["kind"], "trial": False, "voice": True,
                   "duration_s": round(dur, 2), "background": v, "music": m, "comment": task["comment"], "cover": cover,
                   "source": "telegram", **task.get("meta", {}), "created_at": A.iso()}, f, ensure_ascii=False)
    return final, cap


def post_env(ck, lang):
    """The posting logins of one channel + language (the workflow passes all of them, named per channel)."""
    C, L = ck.upper(), lang.upper()
    return {"YOUTUBE_REFRESH_TOKEN": os.environ.get(f"YT_{C}_{L}", ""), "GITHUB_TOKEN": A.PAT,
            f"INSTAGRAM_{L}_TOKEN": os.environ.get("INSTAGRAM_TOKEN", ""),
            f"INSTAGRAM_{L}_USER_ID": os.environ.get(f"IG_{C}_{L}", ""),
            f"FACEBOOK_{L}_PAGE_ID": os.environ.get(f"FB_{C}_{L}", "")}


def run_job(st, jid):
    os.environ["GITHUB_TOKEN"] = A.PAT
    job = (st.get("jobs") or {}).get(jid)
    if not job or job["status"] != "queued":
        A.log("Nothing queued")
        return
    job["status"] = "making"
    A.save_state(st)
    os.makedirs("logs", exist_ok=True)
    made, failed, skipped = [], [], []
    for n, task in enumerate(job["tasks"]):
        logf = os.path.abspath(f"logs/{n}_{task['channel']}_{task['lang']}.log")
        if task["type"] == "special" and A.already_posted(task["channel"], task["lang"], task["yaml"]):
            skipped.append(task["label"])
            A.log(f"{n + 1}: already posted — skipped")
            continue
        try:
            r = render(task, logf)
        except Exception as e:
            r = None
            with open(logf, "a") as f:
                f.write(f"\n{type(e).__name__}: {e}\n")
        if r:
            made.append((task, r[0], r[1], logf))
            A.log(f"{n + 1}: video made")
        else:
            failed.append(task["label"])
            A.log(f"{n + 1}: video failed")
            A.tg_file("sendDocument", "document", logf, chat_id=A.CHAT, caption=f"⚠️ {task['label']}: video failed — log")
    if not made:
        job["status"] = "failed" if failed else "skipped"
        A.save_state(st)
        menu("⚠️ Nothing to post." + (f" Already posted: {', '.join(skipped)}." if skipped else ""))
        return
    if A.APPROVE_VIDEO:
        for task, vid, cap, _ in made:
            A.tg_file("sendVideo", "video", vid, chat_id=A.CHAT, supports_streaming="true", caption=f"🎬 {task['label']}")
        ask = A.say(f"Post {len(made)} video{'s' if len(made) > 1 else ''} now? (YouTube, Instagram, Facebook)",
                    [[("✅ Post", f"a:{jid}:ok"), ("✖️ Cancel", f"a:{jid}:no")]])
        A.save_state(st)
        if not A.wait_approval(st, jid, [ask]):
            job["status"] = "cancelled"
            A.save_state(st)
            menu("✖️ Not posted.")
            return
    job["status"] = "posting"
    A.save_state(st)
    done = []
    for task, vid, cap, logf in made:
        cwd = reels_dir(task["channel"])
        env = post_env(task["channel"], task["lang"])
        if task.get("title"):
            env["YT_TITLE"] = task["title"]
        rv, rc = os.path.relpath(vid, cwd), os.path.relpath(cap, cwd)
        yt = A.sh(["python", "upload_youtube.py", rv, rc, task["lang"]], logf, env, cwd=cwd)
        ig_file = os.path.join(os.path.dirname(rv), f"reel_{task['lang']}_instagram.mp4")
        enc = A.sh(["ffmpeg", "-y", "-loglevel", "error", "-i", rv, "-vf",
                    "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30", "-c:v", "libx264",
                    "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p", "-b:v", "3500k", "-maxrate", "3500k",
                    "-bufsize", "7000k", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart",
                    "-t", "90", ig_file], logf, cwd=cwd)
        ig = enc and A.sh(["python", "upload_instagram.py", ig_file, rc, task["lang"]], logf, env, cwd=cwd)
        if yt and ig:
            done.append(task["label"])
        else:
            failed.append(task["label"] + (" (YouTube)" if not yt else "") + (" (Instagram/Facebook)" if not ig else ""))
            A.tg_file("sendDocument", "document", logf, chat_id=A.CHAT, caption=f"⚠️ {task['label']}: posting problem — log")
        A.log(f"{task['channel']} {task['lang']}: YouTube {'ok' if yt else 'failed'} · Instagram/Facebook {'ok' if ig else 'failed'}")
    job["status"] = "posted" if done else "failed"
    A.save_state(st)
    menu(("✅ Posted: " + ", ".join(done) if done else "⚠️ Nothing posted")
         + (f"\n⚠️ Failed: {', '.join(failed)}" if failed else "") + (f"\n⏭️ Already posted: {', '.join(skipped)}" if skipped else ""))


# ── 🧪 Trial batches (stage 1d): read-only status from news-metrics/batches/ (written by Newsdesk) ──
TRIAL_ICON = {"rendering": "🛠️", "review": "👀", "posting": "📤", "waiting": "⏳", "results": "🏆", "done": "✅",
              "failed": "✖", "discarded": "🗑️"}
TRIAL_WORD = {"rendering": "making the trials", "review": "waiting for your review in Newsdesk", "posting": "posting",
              "waiting": "posted, waiting for results", "results": "results in — make the winner in Newsdesk",
              "done": "winner made", "failed": "failed", "discarded": "discarded"}
STYLE_WORD = {"new": "new style", "current-voice": "current + voice", "current-silent": "current, silent"}


def trial_batches(limit=6):
    """The newest batches (this month and last), newest first. [] if none or unreadable."""
    out, today = [], A.now()
    months = {today.strftime("%Y-%m"), (today.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")}
    names = []
    for m in sorted(months, reverse=True):
        try:
            names += [x["name"] for x in A.gh("GET", f"/repos/{A.STATE_REPO}/contents/batches/{m}") if x["name"].endswith(".json")]
        except Exception:
            pass
    for n in sorted(set(names), reverse=True)[:limit]:
        try:
            out.append(json.loads(A.gh_raw(f"batches/{_month_of(n)}/{n}")))
        except Exception:
            pass
    return out


def _month_of(name):
    """b-20261113-002345-wo.json → 2026-11"""
    d = name.split("-")[1]
    return f"{d[:4]}-{d[4:6]}"


def _syd(iso):
    return f"{A.parse_iso(iso).astimezone(A.SYD):%a %H:%M}" if iso else ""


def _who(b):
    return f"{CH.get(b.get('channel'), b.get('channel'))} · {b.get('format_label', b.get('format'))}" + \
           (f": {b['topic']}" if b.get("topic") else "")


def _winner_line(b, lang):
    w = (b.get("winners") or {}).get(lang) or {}
    s = next((x for x in b.get("slots", []) if x.get("key") == w.get("key")), None)
    if not s:
        return ""
    r = s.get("results") or {}
    t = f"{lang.upper()}: trial {s['slot']} 🥇" + (f" ({r.get('views')} views)" if r.get("views") is not None else "")
    if w.get("close"):
        t += " ⚖️ close call"
    if w.get("done") == "recipe":
        t += " → used for the next daily" + (" ✓" if w.get("recipe_used_at") else "")
    elif w.get("done") == "posted":
        t += " → on YouTube/Facebook; promote it in the Instagram app"
    return t


def trials_status():
    bs = trial_batches()
    if not bs:
        menu("🧪 No trial batches yet. Start one in Newsdesk: Trial batch… on a channel row, or in the special edition, "
             "explainer, fact check or Charts of the Week windows.")
        return None
    lines = ["🧪 Trial batches (newest first)"]
    for b in bs:
        st = b.get("status", "")
        extra = ""
        if st == "waiting" and b.get("results_due"):
            extra = f" · results {_syd(b['results_due'])} Sydney"
        if st in ("results", "done"):
            wl = [_winner_line(b, l) for l in b.get("langs", [])]
            extra = (" · " + " · ".join(x for x in wl if x)) if any(wl) else ""
        n_posted = sum(1 for s in b.get("slots", []) if s.get("status") == "posted")
        lines.append(f"{TRIAL_ICON.get(st, '•')} {_who(b)} · {'+'.join(l.upper() for l in b.get('langs', []))} · "
                     f"{n_posted}/{len(b.get('slots', []))} posted — {TRIAL_WORD.get(st, st)}{extra}")
    rows = [[(f"{TRIAL_ICON.get(b.get('status'), '•')} {(CH.get(b.get('channel'), '') or '?').split()[0]} {b.get('format_label', '')} "
              f"{b.get('created_at', '')[5:10]}", f"t:{b['id']}")] for b in bs]
    ask_buttons("\n".join(lines) + "\n\nTap one for the details.", rows)
    return None


def trial_details(bid, msg):
    try:
        b = json.loads(A.gh_raw(f"batches/{_month_of(bid + '.json')}/{bid}.json"))
    except Exception:
        A.say("⚠️ I couldn't read that batch from news-metrics.")
        return None
    st = b.get("status", "")
    lines = [f"{TRIAL_ICON.get(st, '•')} {_who(b)}", f"{TRIAL_WORD.get(st, st)} · every {b.get('spacing_min')} min · "
             f"results after {b.get('wait_h')} h" + (f" (due {_syd(b['results_due'])} Sydney)" if b.get("results_due") else "")]
    for s in sorted(b.get("slots", []), key=lambda x: (x.get("lang"), x.get("slot"))):
        se, r = s.get("settings") or {}, s.get("results") or {}
        bits = [STYLE_WORD.get(se.get("style"), "")] if se.get("style") else []
        if se.get("order"):
            bits.append(f"#{se['order'][0]} leads")
        if se.get("voice"):
            bits.append(se["voice"].split("-")[-1].replace("Neural", ""))
        nums = (f" · {r.get('views', '–')} views · {r.get('avg_watch_s', '–')} s · skip {r.get('skip_rate', '–')}%"
                if r else "")
        win = " 🥇" if ((b.get("winners") or {}).get(s.get("lang")) or {}).get("key") == s.get("key") else ""
        lines.append(f"• {s['lang'].upper()} {s['slot']}{win}: {', '.join(x for x in bits if x) or 'settings'} — "
                     f"{s.get('status')}{nums}")
    wl = [_winner_line(b, l) for l in b.get("langs", [])]
    if any(wl):
        lines.append("")
        lines += [x for x in wl if x]
    if st == "results":
        lines.append("\n👉 Make the winner in Newsdesk → Trials.")
    if b.get("results_at"):
        lines.append(f"(numbers read {_syd(b['results_at'])} Sydney)")
    A.say("\n".join(lines)[:4000])
    return None


def trials_summary_line():
    """One line for ℹ️ Status ('' if nothing to say)."""
    try:
        bs = trial_batches(10)
    except Exception:
        return ""
    w = sum(1 for b in bs if b.get("status") == "waiting")
    r = sum(1 for b in bs if b.get("status") == "results")
    rv = sum(1 for b in bs if b.get("status") == "review")
    parts = [f"{w} waiting for results"] if w else []
    parts += [f"{r} with results in"] if r else []
    parts += [f"{rv} waiting for your review"] if rv else []
    return ("• Trial batches: " + ", ".join(parts) + "\n") if parts else ""
