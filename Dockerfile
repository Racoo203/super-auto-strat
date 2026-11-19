# Training image: Python 3.13 + the exact dependency set in uv.lock.
# The torch wheel bundles CUDA 12.8, so no CUDA base image is needed: the host only needs the
# NVIDIA driver (>= 570) and the NVIDIA Container Toolkit (see deploy/README.md).
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer: only rebuilt when the lockfile changes).
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY sap ./sap
COPY agents ./agents
COPY tests ./tests
COPY main.py ./

# Unprivileged user; UID/GID match the host user so files in the mounted runs/ stay editable.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" app && useradd -m -u "$UID" -g "$GID" app && mkdir -p /app/runs && chown app:app /app/runs
USER app

# Recorded with every run (Optuna user attr git_commit) since the image has no .git.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT
LABEL org.opencontainers.image.revision=$GIT_COMMIT

CMD ["python", "-m", "agents.ppo.run", "--help"]
