# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Mem0's telemetry opens a second, undeclared local Qdrant under ~/.mem0 and
    # then locks its own directory. Off by default here; it is not the deployer's
    # job to know that.
    MEM0_TELEMETRY=False

WORKDIR /app

# Dependencies before source, so a code change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[all]"

# Nothing here needs root.
RUN useradd --create-home --uid 10001 memgw && chown -R memgw:memgw /app
USER memgw

EXPOSE 8080

# `serve` runs doctor first and refuses to start on a configuration that is
# already provably broken -- a container that exits with the reason beats one
# that stays up and 500s.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["memgw"]
CMD ["serve"]
