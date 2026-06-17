# Naqafin Caption Worker

Optional CUDA transcription worker for the Jellyfin auto-generated captions plugin.

The Jellyfin plugin can upload extracted audio here, let this service run a larger Whisper model on a beefier GPU, then pull back VTT/SRT/text captions. If the worker is unavailable, the plugin can fall back to local generation.

## Related Projects

- [Jellyfin Plugin Auto Generate Captions](https://github.com/naqadata/jellyfin-plugin-auto-generate-captions): Jellyfin server plugin that extracts audio chunks and optionally submits them to this worker.
- [Naqafin for Roku](https://github.com/naqadata/naqafin-roku): Roku client that starts generated-caption sessions and displays the live WebVTT stream from the Jellyfin plugin.
- [Jellyfin Plugin Playlist Up Next](https://github.com/naqadata/jellyfin-plugin-playlist-up-next): separate companion plugin used by Naqafin for playlist-aware resume rows.

This worker is optional. The Jellyfin plugin can still run local Whisper transcription without it, but this service is the preferred path when a separate host has a stronger CUDA GPU.

The runtime stack is intentionally narrow: this service exposes a small HTTP API, runs `faster-whisper`, and stores temporary job files locally. It does not need Jellyfin credentials, Jellyfin media paths, or direct access to the Roku client.

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

## Jellyfin Plugin Configuration

In the Auto Generate Captions plugin settings, enable the remote worker and set:

```text
Remote worker URL: http://<worker-host>:8765
Remote worker API key: value from CAPTION_WORKER_API_KEY, if configured
Remote worker model: large-v3
Fallback to local when unavailable: enabled
```

The plugin uploads already-extracted audio chunks, so this worker does not need access to Jellyfin media paths or Jellyfin authentication.

Naqafin Roku never calls this service directly. The request flow is:

```text
Naqafin Roku -> Jellyfin Auto Generate Captions plugin -> Naqafin Caption Worker
```

The worker returns caption text to the plugin, and the plugin serves live WebVTT back to Naqafin.

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

## Systemd

For a non-Docker host install, use the bundled helper and service template:

```bash
sudo ./scripts/install-systemd.sh /opt/naqafin-caption-worker
sudo systemctl enable --now naqafin-caption-worker.service
```

Review `systemd/naqafin-caption-worker.service` and the environment variables in `.env.example` before enabling it on a production host.
