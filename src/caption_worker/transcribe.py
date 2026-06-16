from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from caption_worker.config import Settings
from caption_worker.formatting import normalize_segments
from caption_worker.schemas import Segment, TranscriptionOptions, TranscriptResult


SENTENCE_PUNCTUATION = (".", "?", "!")
SOFT_PUNCTUATION = (",", ";", ":")


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
    options: TranscriptionOptions | None = None,
) -> TranscriptResult:
    options = options or TranscriptionOptions()
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
        vad_parameters={"threshold": options.vad_threshold},
        word_timestamps=options.enable_regrouping,
    )

    raw_segments = list(segments_iter)
    if options.enable_regrouping:
        segments = regroup_segments(raw_segments, options)
    else:
        segments = [
            Segment(id=index, start=segment.start, end=segment.end, text=segment.text)
            for index, segment in enumerate(raw_segments)
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
            "vad_threshold": options.vad_threshold,
            "enable_regrouping": options.enable_regrouping,
            "regroup_split_gap_seconds": options.regroup_split_gap_seconds,
            "max_cue_characters": options.max_cue_characters,
            "max_cue_words": options.max_cue_words,
            "max_cue_duration_seconds": options.max_cue_duration_seconds,
        },
    )


def regroup_segments(raw_segments: list[object], options: TranscriptionOptions) -> list[Segment]:
    output: list[Segment] = []
    words: list[object] = []
    cue_start: float | None = None
    cue_end = 0.0
    previous_end: float | None = None

    def flush() -> None:
        nonlocal words, cue_start, cue_end
        if cue_start is None or not words:
            return

        text = " ".join(str(getattr(word, "word", "")).strip() for word in words).strip()
        if text:
            output.append(Segment(id=len(output), start=cue_start, end=max(cue_start, cue_end), text=text))

        words = []
        cue_start = None
        cue_end = 0.0

    for raw_segment in raw_segments:
        segment_words = list(getattr(raw_segment, "words", None) or [])
        if not segment_words:
            flush()
            text = str(getattr(raw_segment, "text", "")).strip()
            if text:
                output.append(
                    Segment(
                        id=len(output),
                        start=max(0.0, float(getattr(raw_segment, "start", 0.0))),
                        end=max(0.0, float(getattr(raw_segment, "end", 0.0))),
                        text=text,
                    )
                )
            previous_end = None
            continue

        for word in segment_words:
            word_start = max(0.0, float(getattr(word, "start", getattr(raw_segment, "start", 0.0))))
            word_end = max(word_start, float(getattr(word, "end", word_start)))
            if (
                previous_end is not None
                and words
                and word_start - previous_end >= options.regroup_split_gap_seconds
            ):
                flush()

            if cue_start is None:
                cue_start = word_start

            words.append(word)
            cue_end = word_end
            previous_end = word_end

            text = " ".join(str(getattr(item, "word", "")).strip() for item in words).strip()
            word_count = len(words)
            duration = cue_end - cue_start
            ends_sentence = text.endswith(SENTENCE_PUNCTUATION) and word_count >= 4
            ends_soft = text.endswith(SOFT_PUNCTUATION) and word_count >= 7
            too_long = len(text) >= options.max_cue_characters or word_count >= options.max_cue_words
            too_slow = duration >= options.max_cue_duration_seconds
            if ends_sentence or ends_soft or too_long or too_slow:
                flush()

    flush()
    return output
