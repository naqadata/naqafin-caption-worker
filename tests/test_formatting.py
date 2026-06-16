from caption_worker.formatting import (
    format_srt,
    format_timestamp,
    format_txt,
    format_vtt,
    normalize_caption_text,
)
from caption_worker.schemas import Segment, TranscriptResult


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


def test_normalize_caption_text_capitalizes_first_person_pronouns() -> None:
    assert normalize_caption_text("i think i'm ready and i've got it") == "I think I'm ready and I've got it"
    assert normalize_caption_text("maybe i'll know when i'd seen it") == "maybe I'll know when I'd seen it"
