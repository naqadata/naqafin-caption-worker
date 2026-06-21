from caption_worker.formatting import (
    format_srt,
    format_timestamp,
    format_txt,
    format_vtt,
    normalize_caption_text,
    normalize_segments,
)
from caption_worker.schemas import Segment, TranscriptResult, Word


def test_format_timestamp() -> None:
    assert format_timestamp(3723.456, separator=".") == "01:02:03.456"
    assert format_timestamp(3.4, separator=",") == "00:00:03,400"


def test_caption_formats() -> None:
    result = TranscriptResult(
        language="en",
        duration=2.0,
        segments=[
            Segment(id=0, start=0.0, end=1.25, text=" Hello "),
            Segment(id=1, start=1.25, end=2.0, text="world"),
        ],
    )

    assert format_vtt(result).startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.250" in format_vtt(result)
    assert "00:00:00,000 --> 00:00:01,250" in format_srt(result)
    assert format_txt(result) == "Hello\nworld"


def test_normalize_caption_text_only_cleans_whitespace() -> None:
    assert normalize_caption_text(" i  think i'm ready ") == "i think i'm ready"


def test_normalize_segments_preserves_word_timestamps() -> None:
    segments = normalize_segments(
        [
            Segment(
                id=4,
                start=1.0,
                end=4.0,
                text=" gotta run you ",
                words=[
                    Word(start=1.0, end=1.3, text="gotta"),
                    Word(start=1.35, end=1.6, text="run"),
                    Word(start=4.4, end=4.6, text="you"),
                ],
            )
        ]
    )

    assert len(segments) == 1
    assert segments[0].id == 0
    assert segments[0].text == "gotta run you"
    assert [(word.start, word.end, word.text) for word in segments[0].words] == [
        (1.0, 1.3, "gotta"),
        (1.35, 1.6, "run"),
        (4.4, 4.6, "you"),
    ]
