"""Stand-in for Newsdesk's runner.py: only the paths the copied modules need (GitHub Actions: a temp folder)."""
import os

DATA_DIR = os.environ.get("NEWSDESK_DATA_DIR", "/tmp/newsdesk_desk")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_PATH = os.path.join(DATA_DIR, "history.jsonl")
