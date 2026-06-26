# Bold CV Analyzer

A tiny web app that lives in your browser and tells you how well your CV fits a job.

You upload a CV (PDF) and paste a job description. It parses your CV, compares it with the job’s requirements, and gives you:
- A match score (0–100)
- What’s matched vs. missing
- Simple recommendations to strengthen your CV


Install
Option 1 — Python (Windows/macOS/Linux)
1) Install Python 3.11 or newer
   - Windows: https://www.python.org/downloads/
   - During install on Windows, tick “Add Python to PATH”

2) Open a terminal in this project folder
   - Windows: Shift+Right-click in the folder → “Open PowerShell window here”
   - macOS/Linux: open Terminal and cd into the project directory

3) Create a virtual environment and install dependencies
   - Windows:
     python -m venv .venv
     .venv\Scripts\activate
     python -m pip install --upgrade pip
     pip install -r requirements.txt
   - macOS/Linux:
     python3 -m venv .venv
     source .venv/bin/activate
     python -m pip install --upgrade pip
     pip install -r requirements.txt

4) Start the app
   - flask --app app run -p 8000

5) Open your browser
   - http://localhost:8000
   - If the PDF preview shows no text (scanned PDF), paste your CV text into the fallback box and click Analyze

What it does
- Upload a CV (PDF). The app extracts text (pdfminer.six first, PyPDF if needed). If the PDF is scanned (image-only), paste text manually.
- Paste job requirements. The app splits both texts into short sentences/bullets.
- Matching happens in two ways at once:
  - Text similarity (TF‑IDF with word and character n‑grams)
  - Evidence signals (literal keyword hits, simple aliases, project paragraphs, optional web snippets)
- Score is mostly how many requirements you matched, with a small boost from similarity.

How to use it
1) Go to the app in your browser.
2) Upload your CV as a PDF.
3) Paste the job description on the right.
4) Click Analyze.
5) Review your score, matched and missing items, and the recommendations.

Commands (API endpoints you can call)
- GET /
  Serves the single‑page UI
- GET /api/health
  Returns “ok” (for monitoring/health checks)
- POST /api/parse_pdf (multipart/form-data, field: file)
  Returns { ok, text, chars, warning? }
- POST /api/analyze (application/json)
  Body: { "cv_text": "...", "job_text": "..." }
  Returns: { ok, match_score, matched_keywords[], missing_keywords[], recommendations[] }

How scoring works (plain English)
- Split job and CV into short chunks (sentences, bullets).
- Check for:
  - Similar language between each job chunk and your CV chunks (word + character TF‑IDF).
  - Literal phrase matches and simple alias keywords (e.g., “leadership” ≈ “led a team”).
  - Mentions inside “project” or “experience” paragraphs.
  - Optional web snippets (small public search snippets) for uncommon terms.
- A requirement is “matched” if any of these signals look good.
- Final score = 80% match rate + 20% average similarity, with guardrails so obvious matches don’t get tiny scores.

Reading the score
- 75–100: Strong fit. Most requirements appear covered.
- 45–74: Partial fit. You meet some requirements—add evidence for the missing ones.
- 0–44: Low fit. Tailor your CV with clearer, concrete phrasing and examples.

Troubleshooting
- PDF preview is empty
  The PDF is likely scanned (images only). Paste the CV text in the fallback box and click Analyze. OCR is not included by default.
- Score feels too low
  Paste the full job description (not just the title). Use plain, concrete language in your CV: tools, metrics, outcomes. Optionally enable web snippets.
- Port already in use
  Use a different port, for example:
  flask --app app run -p 8001
- “Module not found” or build errors on scikit‑learn
  Ensure you ran pip install -r requirements.txt in an activated virtual environment. On Linux, install build tools (sudo apt install -y build-essential).

For devs
- Minimal layout:
  - app.py           Flask app + UI + analyzer
  - requirements.txt Python dependencies
  - vercel.json      Optional (serverless deploys)
  - Dockerfile       Optional (container deploys)
  - README.md        This file
- Tech stack:
  - Flask (backend + routes + inline UI)
  - scikit‑learn (TF‑IDF + cosine similarity)
  - pdfminer.six / PyPDF (PDF text extraction)
  - requests (optional web snippets)
- Design notes:
  - No database; everything is in memory.
  - Temporary files stored under /tmp and cleaned up.
  - Optional enrichment cache lives in /tmp/enrich_cache.json (ephemeral on serverless).
  - Keep dependencies lean for fast cold starts and smaller footprints.

Built with
- Flask
- scikit‑learn
- pdfminer.six and PyPDF
- requests

History
This started as an ambitious project of mine, I wanted to create this massive project that I know would take time, just to make my summer more iconic and also because I want to take coding to the next level, so I pured basically days and nights to this, and I can produly say that this has been a massive success of mine which I am highly proud of, I hope you enjoy the project.

License
MIT

About
Tiny app. Useful feedback.
