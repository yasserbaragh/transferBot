"""Shared message-splitting logic for delivery channels.

Both Telegram and Discord cap a single message's length (4096 and 2000
chars respectively), so any digest longer than that has to be split
before sending. Shared here since the splitting rule itself - break on
cluster boundaries, never mid-entry - doesn't depend on which channel
it's headed to.
"""


def chunk_text(text: str, max_length: int) -> list[str]:
    """Split on cluster boundaries (the blank line build_digest puts
    between entries) so a chunk break never lands mid-entry."""
    if len(text) <= max_length:
        return [text]

    blocks = text.split("\n\n")
    chunks = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > max_length:
            if current:
                chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
