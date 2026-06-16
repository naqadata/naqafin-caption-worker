from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class OutputFormat(str, Enum):
    vtt = "vtt"
    srt = "srt"
    txt = "txt"
    json = "json"


class JobResponse(BaseModel):
    job_id: str
    state: JobState
    model: str
    language: str | None = None
    output_format: OutputFormat
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_url: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    device: str
    compute_type: str
    auth_required: bool
    queued_jobs: int
    running_jobs: int


class Segment(BaseModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class TranscriptResult(BaseModel):
    language: str | None = None
    duration: float | None = None
    segments: list[Segment]
    metadata: dict[str, Any] = Field(default_factory=dict)
