# Newspeak: Daily AI/ML News Digest & Newsletter

Newspeak is a modular, functional Python application designed to automatically curate, verify, rank, and summarize the top AI and Machine Learning news stories (7 by default, via `NEWSLETTER_SIZE`) each run. It aggregates data from multiple feed sources, uses Google Gemini to filter and rank stories, and sends a highly polished HTML email to a list of subscribers.

The project runs as a serverless cron job via GitHub Actions, completely free of charge.

## Features

- **Automated Ingestion**: Concurrently aggregates news from standard RSS/Atom feeds (arXiv, Hugging Face, TechCrunch AI, Apple ML) and Hacker News.
- **Jaccard Deduplication**: Compares and deduplicates overlapping articles from different sources based on title similarity.
- **Cross-run history**: Remembers previously sent stories in `sent_history.json` (committed back by the workflow) and skips them, so the same news doesn't reappear across newsletters. Tune via `HISTORY_RETENTION_DAYS` (default 30).
- **AI Ranking & Curated Summaries**: Uses Gemini (`gemini-2.5-flash`) to rank relevance and write dense 2-sentence impact summaries, revealed via a per-article **Summary** toggle in the email.
- **Polished HTML Styling**: Generates a visually stunning email newsletter utilizing a premium slate/indigo responsive card design.
- **Flexible Delivery**: Supports sending via **Resend API** or standard **SMTP** (e.g. Gmail, SendGrid SMTP).
- **Free Deployment**: Pre-configured GitHub Actions workflow runs the job every morning.

---

## Getting Started

### Prerequisites

You need `uv` installed on your machine. If you don't have it, install it using:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Local Setup & Installation

Clone the repository and install the project dependencies:

```bash
# Navigate to project directory
cd Newspeak

# Install dependencies and sync virtual env
uv sync
```

### Local Configuration

Create a `.env` file in the root of the project to set up your keys and configurations:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# RECIPIENTS (Comma-separated list of emails)
RECIPIENTS=user1@example.com,user2@example.com

# Email Provider Options (Choose one)

# Option A: Resend API
RESEND_API_KEY=your_resend_api_key_here
SMTP_FROM=newspeak@yourverifieddomain.com

# Option B: Standard SMTP Fallback (e.g., Gmail SMTP)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587          # 465 (implicit SSL) and 587/25 (STARTTLS) are both supported
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=your_email@gmail.com

# Optional: ranking backend & free-tier tuning (see "Ranking Backends" below)
# LLM_BACKEND=gemini            # gemini (default) | ollama | heuristic
# LLM_MAX_CANDIDATES=150        # lower to reduce tokens if you hit free-tier limits
# OLLAMA_MODEL=llama3.2         # only used when LLM_BACKEND=ollama
# OLLAMA_HOST=http://localhost:11434
```

---

## Ranking Backends & Free-Tier Resilience

Newspeak curates stories with a pluggable ranking backend, chosen via `LLM_BACKEND`:

| `LLM_BACKEND` | Behavior | Cost | Notes |
| --- | --- | --- | --- |
| `gemini` *(default)* | Gemini ranks/summarizes, **with an automatic heuristic backup** | Free-tier Gemini | If Gemini is rate-limited or quota-exhausted, the newsletter still ships via the heuristic ranker instead of failing. Recommended for GitHub Actions. |
| `heuristic` | Pure-Python scoring, no LLM, no API key | Free | Ranks by AI/ML keyword relevance, content depth, and source authority; summaries are drawn from the article text. Always available. |
| `ollama` | A **local** model via [Ollama](https://ollama.com) (with heuristic backup) | Free (your hardware) | For running/previewing on your own machine without spending Gemini quota. **Not suitable for GitHub Actions** (CPU-only runners are too slow). |

**Staying within the Gemini free tier:** the pipeline makes exactly one Gemini call per run. To keep it comfortably under per-minute token limits it heuristically pre-ranks candidates and sends Gemini only the best `LLM_MAX_CANDIDATES` (default 150, with truncated descriptions), and retries transient `429`/`503` responses with backoff. Lower `LLM_MAX_CANDIDATES` if you still hit limits.

**Using Ollama locally:**
```bash
# One-time: install Ollama and pull a small model
ollama pull llama3.2

# Preview the newsletter using the local model (no Gemini quota spent)
LLM_BACKEND=ollama PYTHONPATH=src uv run python src/newspeak/main.py --dry-run
```

---

## Running Locally & Verifying

Newspeak provides CLI flags to easily test the pipeline without spending API tokens or dispatching emails.

### 1. Dry Run with Mock LLM (Safe Preview)
Runs the entire ingestion, filters, compiles the HTML template using mock summaries, and saves the output locally without calling Gemini or sending emails.
```bash
PYTHONPATH=src uv run python src/newspeak/main.py --dry-run --mock-llm
```
*This creates a `last_newsletter.html` file in the root directory. Double-click it to preview the design.*

### 2. Dry Run with Live Gemini
Calls the live Gemini API to rank and summarize real news, but writes the output to `last_newsletter.html` instead of sending emails.
```bash
PYTHONPATH=src uv run python src/newspeak/main.py --dry-run
```

### 3. Full Local Production Run
Runs the entire pipeline (ingestion, live Gemini, live email delivery).
```bash
PYTHONPATH=src uv run python src/newspeak/main.py
```

### Running Tests
To run the automated test suite verifying parser and deduplication logic:
```bash
PYTHONPATH=src uv run pytest
```

---

## GitHub Actions Deployment

The workflow is located in `.github/workflows/newsletter.yml` and is configured to run automatically once a day at 6:00 AM UTC (11:30 AM IST).

### Setup Secrets:
1. Go to your GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions**.
2. Add the following repository secrets:
   - `GEMINI_API_KEY` (Required)
   - `RECIPIENTS` (Required, comma-separated list)
   - `RESEND_API_KEY` (If using Resend) or `SMTP_SERVER` / `SMTP_PORT` / `SMTP_USERNAME` / `SMTP_PASSWORD`
   - `SMTP_FROM` (Email address of the sender)
3. Under the **Actions** tab of your repository, you can manually trigger a run at any time using **Run workflow**.
