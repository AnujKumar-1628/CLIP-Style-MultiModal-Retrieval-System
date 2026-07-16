# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13.7

FROM python:${PYTHON_VERSION}-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/home/app/.cache/huggingface \
    API_HOST=0.0.0.0 \
    API_PORT=8001

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY api ./api
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY docker/entrypoint.sh /usr/local/bin/clip-entrypoint

RUN chmod +x /usr/local/bin/clip-entrypoint scripts/*.sh \
    && groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p data artifacts experiments "${HF_HOME}" \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8001

ENTRYPOINT ["clip-entrypoint"]
CMD ["api"]
