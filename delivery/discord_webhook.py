"""Sends the rumor digest to a Discord channel via an incoming webhook
(pipeline step 7).

Fallback delivery channel while the Telegram bot is blocked on account
restrictions - same send_digest(text) -> bool contract as
delivery/telegram_bot.py, so scheduler.py only needs a one-line import
change to switch between them (or later, to send to both).
"""

import os

import requests
from dotenv import load_dotenv

from delivery.chunking import chunk_text

# Loads DISCORD_WEBHOOK_URL from .env, same pattern as the other delivery
# modules.
load_dotenv()

# Discord hard-caps a single message's content at 2000 characters.
MAX_MESSAGE_LENGTH = 2000


def send_digest(text: str, webhook_url: str | None = None) -> bool:
    """Send a digest to Discord, splitting into multiple messages if it
    exceeds Discord's per-message length limit.

    Returns True only if every chunk sends successfully - a partial send
    (e.g. chunk 2 of 3 fails) counts as a failure so callers don't mark
    the report as delivered when it wasn't fully delivered.
    """
    webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL must be set")

    for chunk in chunk_text(text, MAX_MESSAGE_LENGTH):
        response = requests.post(webhook_url, json={"content": chunk}, timeout=10)
        if not response.ok:
            print(f"[error] discord send failed: {response.status_code} {response.text}")
            return False
    return True


if __name__ == "__main__":
    ok = send_digest("Test message from the transfer-rumor bot.")
    print("sent OK" if ok else "send FAILED")
