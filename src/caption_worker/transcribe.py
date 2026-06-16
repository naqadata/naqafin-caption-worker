from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from caption_worker.config import Settings
from caption_worker.formatting import normalize_segments
from caption_worker.schemas import Segment, TranscriptResult


@lru_cache(maxsize=4)
def get_model(model_name: str, device: str, compute_type: str, cache_dir: str) -> WhisperModel:
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=cache_dir,
    )


def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str,
    language: str | None,
    settings: Settings,
) -> TranscriptResult:
    model = get_model(
        model_name,
        settings.whisper_device,
        settings.whisper_compute_type,
        str(settings.model_cache_dir),
    )
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
    )

    segments = [
        Segment(id=index, start=segment.start, end=segment.end, text=segment.text)
        for index, segment in enumerate(segments_iter)
    ]

    detected_language = getattr(info, "language", None)
    duration = getattr(info, "duration", None)
    return TranscriptResult(
        language=detected_language,
        duration=duration,
        segments=normalize_segments(segments),
        metadata={
            "model": model_name,
            "device": settings.whisper_device,
            "compute_type": settings.whisper_compute_type,
        },
    )
