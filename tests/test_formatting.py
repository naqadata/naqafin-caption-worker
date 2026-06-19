from caption_worker.formatting import (
    format_srt,
    format_timestamp,
    format_txt,
    format_vtt,
    normalize_caption_text,
    normalize_segments,
)
from caption_worker.punctuation import restore_punctuation_with_model
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


def test_restore_punctuation_updates_segment_text_and_words() -> None:
    segments = [
        Segment(
            id=0,
            start=1.0,
            end=3.0,
            text="hello there are you okay",
            words=[
                Word(start=1.0, end=1.2, text="hello"),
                Word(start=1.3, end=1.5, text="there"),
                Word(start=1.6, end=1.8, text="are"),
                Word(start=1.9, end=2.1, text="you"),
                Word(start=2.2, end=2.4, text="okay"),
            ],
        )
    ]

    restored = restore_punctuation_with_model(segments, FakePunctuationModel())

    assert restored[0].text == "Hello there, are you okay?"
    assert [word.text for word in restored[0].words] == [
        "Hello",
        "there,",
        "are",
        "you",
        "okay?",
    ]


class FakePunctuationModel:
    def preprocess(self, text: str) -> str:
        return text

    def predict(self, text: str) -> list[tuple[str, str, float]]:
        return [
            ("Hello", "0", 0.9),
            ("there", ",", 0.9),
            ("are", "0", 0.9),
            ("you", "0", 0.9),
            ("okay", "?", 0.9),
        ]

    def restore_punctuation(self, text: str) -> str:
        return text
