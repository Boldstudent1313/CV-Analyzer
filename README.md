# Strategy B: Lightweight Semantic Evaluation Engine on Flask (Vercel-ready)

This project implements a production-grade, Vercel-compatible Flask app with a custom, dependency-light semantic evaluation model. It avoids heavy ML artifacts and instead uses from-scratch text normalization, subword/n-gram tokenization, and pure NumPy implementations of TF-IDF, BM25, and cosine similarity. A resilient web-search layer (DuckDuckGo) enriches candidate-job alignment with external signals for university prestige, company tier, and project saturation.

Key features:
- Custom normalization and tokenization (character, subword, and word n-grams)
- Vectorizers: TF, IDF, TF-IDF, BM25 with robust numerical safeguards
- Scoring: cosine similarity, Jaccard, soft merges of multi-representation vectors
- Resilient DuckDuckGo search with progressive backoff and HTML parsing
- Strategy B weighting layer adjusts base semantic scores from external signals
- PDF and text ingestion, deterministic processing for reproducibility
- Clean Tailwind-based front-end with reactive state panels

Deployment:
- Designed to run on Vercel Python builder; top-level `app = Flask(__name__)` is guaranteed
- Small dependency footprint to respect 250MB runtime constraints

## Deployment Notes

- Configure Vercel Python builder.
- Provide environment if needed.

## Finalization

Prepared for Vercel deployment.
