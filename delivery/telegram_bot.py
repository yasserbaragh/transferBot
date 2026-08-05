"""Sends the rumor digest to a Telegram chat via the Bot API (pipeline
step 7).

Plain HTTP POST to Telegram's sendMessage endpoint - no bot-framework
dependency needed since this only pushes a digest out, it never needs to
receive messages/commands from users. Same shape as the originally
planned Discord webhook: one HTTP call per digest, no long-running
process of its own.
"""

import os

import requests
from dotenv import load_dotenv

from delivery.chunking import chunk_text

# Loads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env, same pattern as
# ingestion/llm_filter.py.
load_dotenv()

TELEGRAM_API_BASE = "https://api.telegram.org"

# Telegram hard-caps a single sendMessage call at 4096 characters.
MAX_MESSAGE_LENGTH = 4096


def send_digest(text: str, token: str | None = None, chat_id: str | None = None) -> bool:
    """Send a digest to Telegram, splitting into multiple messages if it
    exceeds Telegram's per-message length limit.

    Returns True only if every chunk sends successfully - a partial send
    (e.g. chunk 2 of 3 fails) counts as a failure so callers don't mark
    the report as delivered when it wasn't fully delivered.
    """
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"

    for chunk in chunk_text(text, MAX_MESSAGE_LENGTH):
        response = requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=10)
        if not response.ok:
            print(f"[error] telegram send failed: {response.status_code} {response.text}")
            return False
    return True


if __name__ == "__main__":
    ok = send_digest("Test message from the transfer-rumor bot.")
    print("sent OK" if ok else "send FAILED")
