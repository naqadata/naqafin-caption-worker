from contextlib import asynccontextmanager
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import PlainTextResponse

from caption_worker.auth import require_api_key
from caption_worker.config import Settings, get_settings
from caption_worker.formatting import format_srt, format_txt, format_vtt
from caption_worker.jobs import JobStore
from caption_worker.schemas import (
    HealthResponse,
    JobResponse,
    JobState,
    OutputFormat,
    TranscriptionOptions,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

settings = get_settings()
store = JobStore(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.start()
    yield
    await store.stop()


app = FastAPI(
    title="Naqafin Caption Worker",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    queued, running = await store.counts()
    return HealthResponse(
        model=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        auth_required=bool(settings.caption_worker_api_key.strip()),
        queued_jobs=queued,
        running_jobs=running,
    )


@app.post(
    "/v1/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
async def create_job(
    audio: Annotated[UploadFile, File()],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.vtt,
    vad_threshold: Annotated[float, Form()] = 0.35,
    enable_regrouping: Annotated[bool, Form()] = True,
    regroup_split_gap_seconds: Annotated[float, Form()] = 0.35,
    max_cue_characters: Annotated[int, Form()] = 84,
    max_cue_words: Annotated[int, Form()] = 14,
    max_cue_duration_seconds: Annotated[float, Form()] = 6.0,
    settings: Settings = Depends(get_settings),
) -> JobResponse:
    try:
        options = TranscriptionOptions(
            vad_threshold=vad_threshold,
            enable_regrouping=enable_regrouping,
            regroup_split_gap_seconds=regroup_split_gap_seconds,
            max_cue_characters=max_cue_characters,
            max_cue_words=max_cue_words,
            max_cue_duration_seconds=max_cue_duration_seconds,
        )
        job = await store.create_job(
            audio,
            model=(model or settings.whisper_model).strip(),
            language=(language or settings.whisper_language or "").strip() or None,
            output_format=output_format,
            options=options,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
    return job.to_response()


@app.get("/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str) -> JobResponse:
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job.to_response()


@app.get("/v1/jobs/{job_id}/captions", dependencies=[Depends(require_api_key)])
async def get_captions(
    job_id: str,
    output_format: Annotated[OutputFormat, Query(alias="format")] = OutputFormat.vtt,
):
    job = await store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if job.state != JobState.succeeded or not job.result:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Job is {job.state}.")

    if output_format == OutputFormat.json:
        return job.result
    if output_format == OutputFormat.srt:
        return PlainTextResponse(format_srt(job.result), media_type="application/x-subrip")
    if output_format == OutputFormat.txt:
        return PlainTextResponse(format_txt(job.result), media_type="text/plain")
    return PlainTextResponse(format_vtt(job.result), media_type="text/vtt")


@app.delete(
    "/v1/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
)
async def delete_job(job_id: str) -> None:
    deleted = await store.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
