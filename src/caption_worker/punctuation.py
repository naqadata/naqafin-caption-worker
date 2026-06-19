from functools import lru_cache
import logging

from caption_worker.schemas import Segment, Word


LOGGER = logging.getLogger(__name__)
DEFAULT_PUNCTUATION_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"


@lru_cache(maxsize=2)
def get_punctuation_model(model_name: str):
    from deepmultilingualpunctuation import PunctuationModel

    return PunctuationModel(model=model_name)


def restore_punctuation(segments: list[Segment], model_name: str) -> list[Segment]:
    if not segments:
        return segments

    try:
        model = get_punctuation_model(model_name or DEFAULT_PUNCTUATION_MODEL)
        return restore_punctuation_with_model(segments, model)
    except Exception:
        LOGGER.exception("Punctuation restoration failed; using unmodified transcript.")
        return segments


def restore_punctuation_with_model(segments: list[Segment], model: object) -> list[Segment]:
    token_groups = [segment_tokens(segment) for segment in segments]
    flat_tokens = [token for group in token_groups for token in group]
    if not flat_tokens:
        return segments

    predicted_tokens = predict_tokens(model, " ".join(flat_tokens))
    if len(predicted_tokens) != len(flat_tokens):
        return [
            segment.model_copy(update={"text": restore_segment_text(model, segment.text)})
            for segment in segments
        ]

    restored_segments: list[Segment] = []
    offset = 0
    for segment, tokens in zip(segments, token_groups, strict=True):
        restored = predicted_tokens[offset : offset + len(tokens)]
        offset += len(tokens)
        text = " ".join(restored).strip()
        if segment.words and len(segment.words) == len(restored):
            words = [
                Word(start=word.start, end=word.end, text=token)
                for word, token in zip(segment.words, restored, strict=True)
            ]
        else:
            words = segment.words

        restored_segments.append(segment.model_copy(update={"text": text or segment.text, "words": words}))

    return restored_segments


def segment_tokens(segment: Segment) -> list[str]:
    if segment.words:
        return [word.text.strip() for word in segment.words if word.text.strip()]
    return [token for token in segment.text.split() if token]


def predict_tokens(model: object, text: str) -> list[str]:
    clean_text = model.preprocess(text)
    labels = model.predict(clean_text)
    tokens: list[str] = []
    for item in labels:
        if len(item) < 2:
            continue

        word = str(item[0]).strip()
        label = str(item[1]).strip()
        if not word:
            continue

        punctuation = "" if label in {"0", "O"} else label
        tokens.append(word + punctuation)

    return tokens


def restore_segment_text(model: object, text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    restored = str(model.restore_punctuation(stripped)).strip()
    return restored or stripped
