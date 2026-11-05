FROM python:3.12
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/
LABEL org.opencontainers.image.source https://github.com/SENERGY-Platform/open_window_v2
WORKDIR /usr/src/app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY . .
RUN apt-get update && apt-get install -y git
RUN git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > git_commit 
RUN uv sync --locked --no-dev
RUN apt-get purge -y git && apt-get auto-remove -y && apt-get clean && rm -rf .git
ENV PATH="/usr/src/app/.venv/bin:$PATH"
CMD [ "python3", "-u", "main.py"]
