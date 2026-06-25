# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Optional timeouts for web lookups
    GCS_TIMEOUT=1.2

# System deps for PDFs and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    libffi-dev \
    libjpeg62-turbo \
    zlib1g \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Install Python dependencies first for better layer caching
# If you have a requirements.txt in repo root, this will pick it up
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source
# Ensure your app.py is at repo root or adjust the path accordingly
COPY . /app

# Expose the port your app will listen on
EXPOSE 8000

# Environment variables (runtime secrets should be provided by your platform)
# GOOGLE_API_KEY, GOOGLE_CX, GCS_ENABLED can be injected by your host
# Example:
# ENV GCS_ENABLED=1
# ENV GOOGLE_API_KEY=YOUR_KEY
# ENV GOOGLE_CX=YOUR_CX

# Default command:
# If your app is the Vercel-style (no __main__), run with Gunicorn WSGI server
# Flask app variable must be named "app" in app.py (it is in your code)
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:8000", "app:app"]
