from functools import lru_cache
import logging
from pathlib import Path

from faster_whisper import WhisperModel

from caption_worker.config import Settings
from caption_worker.formatting import normalize_segments
from caption_worker.schemas import Segment, TranscriptionOptions, TranscriptResult, Word


SENTENCE_PUNCTUATION = (".", "?", "!")
SOFT_PUNCTUATION = (",", ";", ":")
LOGGER = logging.getLogger(__name__)


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
    LOGGER.info(
        "Whisper transcription complete: audio_path=%s model=%s language_hint=%s detected_language=%s "
        "raw_segments=%d duration=%s enable_regrouping=%s",
        audio_path,
        model_name,
        language,
        getattr(info, "language", None),
        len(raw_segments),
        getattr(info, "duration", None),
        options.enable_regrouping,
    )
    if options.enable_regrouping:
        segments = regroup_segments(raw_segments, options)
    else:
        segments = [
            Segment(
                id=index,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=extract_words(segment),
            )
            for index, segment in enumerate(raw_segments)
        ]

    segments = normalize_segments(segments)

    detected_language = getattr(info, "language", None)
    duration = getattr(info, "duration", None)
    return TranscriptResult(
        language=detected_language,
        duration=duration,
        segments=segments,
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
    words: list[Word] = []
    cue_start: float | None = None
    cue_end = 0.0
    previous_end: float | None = None

    def flush() -> None:
        nonlocal words, cue_start, cue_end
        if cue_start is None or not words:
            return

        text = " ".join(word.text.strip() for word in words).strip()
        if text:
            output.append(
                Segment(
                    id=len(output),
                    start=cue_start,
                    end=max(cue_start, cue_end),
                    text=text,
                    words=words.copy(),
                )
            )

        words = []
        cue_start = None
        cue_end = 0.0

    for raw_segment in raw_segments:
        segment_words = extract_words(raw_segment)
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
                        words=[],
                    )
                )
            previous_end = None
            continue

        for word in segment_words:
            word_start = word.start
            word_end = word.end
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

            text = " ".join(item.text.strip() for item in words).strip()
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


def extract_words(raw_segment: object) -> list[Word]:
    words: list[Word] = []
    for raw_word in list(getattr(raw_segment, "words", None) or []):
        text = str(getattr(raw_word, "word", "")).strip()
        if not text:
            continue

        start = max(0.0, float(getattr(raw_word, "start", getattr(raw_segment, "start", 0.0))))
        end = max(start, float(getattr(raw_word, "end", start)))
        words.append(Word(start=start, end=end, text=text))

    return words
