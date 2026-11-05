FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# the operator lib and some of its dependencies are installed from their git repositories
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# the same base, so the interpreter the environment links to is there
FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/SENERGY-Platform/open_window_detection
WORKDIR /usr/src/app
COPY --from=build /usr/src/app/.venv .venv
ENV PATH="/usr/src/app/.venv/bin:$PATH"
# printed by the operator lib at start
ARG GIT_COMMIT=unknown
RUN echo "commit=$GIT_COMMIT" > git_commit
COPY main.py ./
COPY algo algo
CMD ["python", "-u", "main.py"]
