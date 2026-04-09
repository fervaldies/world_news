import anthropic
import sys
import json
import re
import time
from datetime import datetime


def extract_json(text):
    """Extract the first complete JSON object from text, ignoring any preamble."""
    text = text.strip()
    # Remove markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    # Find the first { and last } to isolate the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {repr(text)}")
    return text[start:end + 1]


def call_with_search(client, prompt):
    """Call API with web search and return final text."""
    messages = [{"role": "user", "content": prompt}]
    for _ in range(5):  # max 5 iterations
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages
        )
        print(f"stop_reason: {response.stop_reason}")
        print(f"content blocks: {[b.type for b in response.content]}")
        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            print(f"Raw response: {repr(text)}")
            return extract_json(text)
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Search completed."
                })
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
    raise Exception("Max iterations reached without end_turn")


def get_news(day_name):
    client = anthropic.Anthropic()

    # --- Fetch 5 world news headlines in English ---
    en_text = call_with_search(client, (
        "Search for the 5 most important world news stories today. "
        "Choose stories from different regions and topics — avoid picking 2 stories about the same country or subject. "
        "Prioritise stories with broad global relevance. "
        "Return ONLY raw JSON, no markdown, no backticks, no explanation:\n"
        '{"news": [{"title": "headline under 20 words"}, {"title": "headline under 20 words"}, {"title": "headline under 20 words"}, {"title": "headline under 20 words"}, {"title": "headline under 20 words"}]}'
    ))
    print(f"Cleaned EN text: {repr(en_text)}")
    en_data = json.loads(en_text)

    # --- Translate to Spanish (Spain) ---
    time.sleep(10)  # wait for rate limit window to reset
    headlines = "\n".join(f"- {n['title']}" for n in en_data["news"])
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
