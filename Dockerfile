# The Nimbus API image. Two stages: build (uv installs EXACT versions from uv.lock, compiled to bytecode) and runtime
# (only the virtual environment and the code: no compilers, no uv), running as a non-root user.
#   docker build -t nimbus-api .          docker run -p 8000:8000 -e MISTRAL_API_KEY nimbus-api

FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: a code change doesn't reinstall them (layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

FROM python:3.14-slim AS runtime
RUN useradd --create-home --uid 10001 nimbus
WORKDIR /app
COPY --from=build --chown=nimbus:nimbus /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 NIMBUS_LOG_FORMAT=json
USER nimbus
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "nimbus.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
