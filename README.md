# Transfer Market Rumor Bot

A bot that watches football transfer news, figures out when different outlets are talking about the same rumor, scores how believable it looks based on who's reporting it, and posts a digest to Discord (and Telegram, if you set it up).

It doesn't wait for you to ask it anything. It runs on a timer, checks the news, and reports what's new.

## Why this exists

Transfer window news is chaos. The same rumor gets reported by ten outlets, in five different languages, with five different transfer fees attached, and it's genuinely hard to tell "this is new information" apart from "someone just rewrote yesterday's story" by skimming headlines. This bot sorts that out for you: it reads the news, pulls out the useful bits (who, from where, to where, for how much), and groups reports that are about the same transfer so you can judge how credible a rumor is by how many separate sources agree on it — not by how confident any one headline sounds.

## How it works

Roughly, in order:

1. **Read the news** — pull in articles from a list of football news sources.
2. **Skip what's already been seen** — so the same story doesn't get processed twice.
3. **Quick relevance check** — toss out anything that's obviously not a transfer story (match reports, injury updates, etc).
4. **Understand what's left** — an AI model reads each article and pulls out the player, the clubs involved, the fee, the wage, and how confident it is that this is a real rumor.
5. **Group duplicates together** — if two articles are clearly about the same rumor, they get merged into one story instead of showing up twice.
6. **Decide how believable it is** — the more separate, trustworthy sources reporting the same thing, the higher the confidence.
7. **Remember it** — everything gets saved so future runs know what's already been reported.
8. **Send the report** — on a schedule, whatever's new or has changed gets written up and sent out.

## What it does, simply put

Think of it as a person whose whole job is reading every football transfer article as it comes out, remembering what's already been said, noticing when five sites are reporting the exact same thing, and only tapping you on the shoulder when there's something genuinely new or newly confirmed — instead of you having to read all of it yourself.

## How to run it

1. Install Python, then install the project's dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # on Linux/Mac: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your own keys:

   ```bash
   copy .env.example .env        # on Linux/Mac: cp .env.example .env
   ```

   You'll need:
   - A Gemini API key, for the AI extraction step.
   - A Discord webhook URL, for where the reports get sent.
   - A Telegram bot token and chat ID, only if you also want reports sent to Telegram.

3. Open `config/sources.yaml` and check the list of news sources — add, remove, or adjust any you want.

4. Run it:

   ```bash
   python scheduler.py
   ```

   It runs once immediately, then keeps checking for updates on a timer.

## What a report looks like

```
**Transfer rumor digest - 2 update(s)**

__New rumors__

**Alex Scott** - Chelsea -> Arsenal
50/50 | Sources: fabrizio_romano_daily_briefing
https://...

__Updated rumors__

**Bruno Guimaraes** - Newcastle -> Arsenal
Confident | Sources: caughtoffside, bbc_sport_football
Reported fee: £80m
https://...
```

New rumors and updates to ones you've already seen are shown separately. Confidence is a simple label, not a fake-precise percentage — it reflects how many independent sources agree, not how sure the AI felt about one single article.

## License

MIT. See the `LICENSE` file.
