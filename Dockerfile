FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir .

RUN mkdir -p /app/storage /models

ENV HF_HOME=/models

EXPOSE 8765

CMD ["uvicorn", "caption_worker.main:app", "--host", "0.0.0.0", "--port", "8765"]
