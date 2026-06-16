import re

from caption_worker.schemas import Segment, TranscriptResult


FIRST_PERSON_PATTERN = re.compile(r"\bi(?:('|\u2019)(m|ve|ll|d))?\b", re.IGNORECASE)


def normalize_caption_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        suffix = match.group(2)
        if not suffix:
            return "I"
        return f"I'{suffix}"

    return FIRST_PERSON_PATTERN.sub(replace, text)


def format_timestamp(seconds: float, *, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def format_vtt(result: TranscriptResult) -> str:
    cues = ["WEBVTT", ""]
    for segment in result.segments:
        cues.append(
            f"{format_timestamp(segment.start, separator='.')} --> "
            f"{format_timestamp(segment.end, separator='.')}"
        )
        cues.append(segment.text.strip())
        cues.append("")
    return "\n".join(cues)


def format_srt(result: TranscriptResult) -> str:
    cues: list[str] = []
    for index, segment in enumerate(result.segments, start=1):
        cues.append(str(index))
        cues.append(
            f"{format_timestamp(segment.start, separator=',')} --> "
            f"{format_timestamp(segment.end, separator=',')}"
        )
        cues.append(segment.text.strip())
        cues.append("")
    return "\n".join(cues)


def format_txt(result: TranscriptResult) -> str:
    return "\n".join(segment.text.strip() for segment in result.segments if segment.text.strip())


def normalize_segments(raw_segments: list[Segment]) -> list[Segment]:
    segments: list[Segment] = []
    for index, segment in enumerate(raw_segments):
        start = max(0.0, segment.start)
        end = max(start, segment.end)
        text = normalize_caption_text(" ".join(segment.text.split()))
        if text:
            segments.append(Segment(id=index, start=start, end=end, text=text))
    return segments
