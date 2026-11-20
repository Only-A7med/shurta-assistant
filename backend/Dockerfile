# The API. Ollama is not containerised: it stays on the host, where it can use
# the machine's memory and stay resident between requests, and is reached at
# OLLAMA_BASE_URL (http://host.docker.internal:11434 from inside a container).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so a change to the source does not reinstall them.
COPY pyproject.toml README.md ./
COPY backend/shurta/__init__.py backend/shurta/__init__.py
RUN python -m pip install --no-cache-dir .

COPY backend/shurta backend/shurta

# The catalogue and the index are built on the host and mounted in, because the
# catalogue is collected from the official public website by whoever runs this
# and is not part of the image.
VOLUME ["/app/data"]

EXPOSE 8000

# No shell form, so signals reach the process and the container stops promptly.
CMD ["python", "-m", "uvicorn", "shurta.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
