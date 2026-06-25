
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Optional small timeout for external snippet lookups
    GCS_TIMEOUT=1.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    libffi-dev \
    libjpeg62-turbo \
    zlib1g \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt


COPY . /app


EXPOSE 8080


CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:8080", "app:app"]
