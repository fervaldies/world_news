import anthropic
import sys
import json
import re
import time
from datetime import datetime


def extract_json(text):
    """Extract the first complete JSON object from text, ignoring any preamble."""
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {repr(text)}")
    return text[start:end + 1]


def get_news(day_name):
    import anthropic
    from ai_failover import generate_with_search

    # --- Fetch 5 world news headlines in English (with retry) ---
    en_data = None
    for attempt in range(3):
        en_text = generate_with_search(
            "Search for the 5 most important world news stories today. "
            "Choose stories from different regions and topics — avoid picking 2 stories "
            "about the same country or subject. Prioritise stories with broad global relevance. "
            "You MUST return ONLY a JSON object with no other text whatsoever. "
            "No prose, no explanation, no apology — even if uncertain, pick the 5 best "
            "headlines you found and return them in this exact format:\n"
            '{"news": [{"title": "headline under 20 words"}, {"title": "headline under 20 words"}, {"title": "headline under 20 words"}, {"title": "headline under 20 words"}, {"title": "headline under 20 words"}]}'
        )
        try:
            en_text = extract_json(en_text)
            print(f"Cleaned EN text: {repr(en_text)}")
            en_data = json.loads(en_text)
            break
        except (ValueError, json.JSONDecodeError) as e:
            print(f"⚠️ Attempt {attempt + 1}/3 failed to get valid JSON: {e}")
            if attempt == 2:
                raise

    # --- Translate to Spanish (Spain) ---
    time.sleep(10)
    headlines = "\n".join(f"- {n['title']}" for n in en_data["news"])
    client = anthropic.Anthropic()
    es_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Translate these headlines to Spanish from Spain. "
                "Return ONLY raw JSON, no markdown, no backticks, no explanation:\n"
                '{"news": [{"title": "translated"}, {"title": "translated"}, {"title": "translated"}, {"title": "translated"}, {"title": "translated"}]}\n\n'
                f"Headlines:\n{headlines}"
            )
        }]
    )
    es_text = extract_json("".join(b.text for b in es_response.content if hasattr(b, "text")))
    print(f"Cleaned ES text: {repr(es_text)}")
    es_data = json.loads(es_text)

    # --- Write YML files ---
    date_str = datetime.now().strftime("%Y-%m-%d")

    def build_yml(data):
        lines = [f"date: {date_str}", f"day: {day_name}", "news:"]
        for item in data["news"]:
            lines.append(f'  - title: "{item["title"]}"')
        return "\n".join(lines) + "\n"

    with open(f"{day_name}NewsEN.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(en_data))
    with open(f"{day_name}NewsES.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(es_data))

    print(f"✅ Created {day_name}NewsEN.yml and {day_name}NewsES.yml")


if __name__ == "__main__":
    day_name = sys.argv[1]
    get_news(day_name)
