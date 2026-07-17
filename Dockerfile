FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies first — separate layer so it's cached on code-only changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev

# Copy source after deps so the expensive layer above is reused
COPY . .

EXPOSE 8000

# Render injects $PORT at runtime; fall back to 8000 for local `docker run`.
# Use the shell form so ${PORT} is expanded (exec form would pass it literally).
CMD ["sh", "-c", "fastapi run main.py --host 0.0.0.0 --port ${PORT:-8000}"]
