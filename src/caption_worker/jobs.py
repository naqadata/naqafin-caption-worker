import asyncio
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from caption_worker.config import Settings
from caption_worker.schemas import JobResponse, JobState, OutputFormat, TranscriptResult
from caption_worker.transcribe import transcribe_audio


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass
class CaptionJob:
    job_id: str
    audio_path: Path
    model: str
    language: str | None
    output_format: OutputFormat
    state: JobState = JobState.queued
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: TranscriptResult | None = None

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            state=self.state,
            model=self.model,
            language=self.language,
            output_format=self.output_format,
            created_at=iso(self.created_at) or "",
            updated_at=iso(self.updated_at) or "",
            started_at=iso(self.started_at),
            completed_at=iso(self.completed_at),
            error=self.error,
            result_url=f"/v1/jobs/{self.job_id}/captions" if self.result else None,
        )


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, CaptionJob] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.lock = asyncio.Lock()
        self.workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self.workers:
            return
        for index in range(self.settings.max_concurrent_jobs):
            self.workers.append(asyncio.create_task(self._worker(index)))

    async def stop(self) -> None:
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def create_job(
        self,
        upload: UploadFile,
        *,
        model: str,
        language: str | None,
        output_format: OutputFormat,
    ) -> CaptionJob:
        job_id = uuid4().hex
        job_dir = self.settings.job_storage_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        suffix = Path(upload.filename or "audio").suffix or ".audio"
        audio_path = job_dir / f"source{suffix}"

        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        written = 0
        with audio_path.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise ValueError(f"Upload exceeds {self.settings.max_upload_mb} MB limit.")
                output.write(chunk)

        job = CaptionJob(
            job_id=job_id,
            audio_path=audio_path,
            model=model,
            language=language,
            output_format=output_format,
        )
        async with self.lock:
            self.jobs[job_id] = job
        await self.queue.put(job_id)
        return job

    async def get_job(self, job_id: str) -> CaptionJob | None:
        async with self.lock:
            return self.jobs.get(job_id)

    async def delete_job(self, job_id: str) -> bool:
        async with self.lock:
            job = self.jobs.pop(job_id, None)
        if not job:
            return False
        shutil.rmtree(job.audio_path.parent, ignore_errors=True)
        return True

    async def counts(self) -> tuple[int, int]:
        async with self.lock:
            queued = sum(1 for job in self.jobs.values() if job.state == JobState.queued)
            running = sum(1 for job in self.jobs.values() if job.state == JobState.running)
        return queued, running

    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                await self._run_job(job_id)
            finally:
                self.queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        async with self.lock:
            job = self.jobs.get(job_id)
            if not job or job.state == JobState.cancelled:
                return
            job.state = JobState.running
            job.started_at = utc_now()
            job.updated_at = job.started_at

        try:
            result = await asyncio.to_thread(
                transcribe_audio,
                job.audio_path,
                model_name=job.model,
                language=job.language,
                settings=self.settings,
            )
        except Exception as exc:
            async with self.lock:
                job.state = JobState.failed
                job.error = str(exc)
                job.completed_at = utc_now()
                job.updated_at = job.completed_at
            return

        async with self.lock:
            job.result = result
            job.state = JobState.succeeded
            job.completed_at = utc_now()
            job.updated_at = job.completed_at
