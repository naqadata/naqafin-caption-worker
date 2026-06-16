# Naqafin Caption Worker

Optional CUDA transcription worker for the Jellyfin auto-generated captions plugin.

The Jellyfin plugin can upload extracted audio here, let this service run a larger Whisper model on a beefier GPU, then pull back VTT/SRT/text captions. If the worker is unavailable, the plugin can fall back to local generation.

## API

Auth is optional. If `CAPTION_WORKER_API_KEY` is set, requests must include:

```http
Authorization: Bearer <key>
```

Endpoints:

```text
GET  /health
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/captions?format=vtt
DELETE /v1/jobs/{job_id}
```

`POST /v1/jobs` accepts multipart form data:

```text
audio: uploaded audio/video file
model: optional model override
language: optional language hint, e.g. en
output_format: vtt, srt, txt, or json
```

## Docker

Install the NVIDIA container toolkit on the host, then:

```bash
cp .env.example .env
docker compose up --build
```

Health check:

```bash
curl http://localhost:8765/health
```

Submit a job:

```bash
curl -F 'audio=@sample.flac' http://localhost:8765/v1/jobs
```

Poll the returned `job_id`:

```bash
curl http://localhost:8765/v1/jobs/<job_id>
curl http://localhost:8765/v1/jobs/<job_id>/captions?format=vtt
```

## Local Dev

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
uvicorn caption_worker.main:app --reload --port 8765
```

The default runtime assumes CUDA. For CPU smoke tests, set:

```bash
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```
