FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    GRADIO_SERVER_NAME=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py SPACE_README.md ./

RUN pip install --upgrade pip \
    && pip install -e ".[demo]"

EXPOSE 7860

CMD ["python", "app.py"]
