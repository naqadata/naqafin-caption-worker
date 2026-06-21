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


class Word(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Segment(BaseModel):
    id: int
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    words: list[Word] = Field(default_factory=list)


class TranscriptionOptions(BaseModel):
    vad_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    enable_regrouping: bool = True
    regroup_split_gap_seconds: float = Field(default=0.35, ge=0.1, le=2.0)
    max_cue_characters: int = Field(default=84, ge=20, le=180)
    max_cue_words: int = Field(default=14, ge=3, le=40)
    max_cue_duration_seconds: float = Field(default=6.0, ge=1.0, le=15.0)


class TranscriptResult(BaseModel):
    language: str | None = None
    duration: float | None = None
    segments: list[Segment]
    metadata: dict[str, Any] = Field(default_factory=dict)
