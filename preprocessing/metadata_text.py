import html
import re


METADATA_SENTENCE_FIELDS = ["title", "price", "brand", "feature", "categories", "description"]


def list_to_str(value) -> str:
    return " ".join(str(item) for item in value) if isinstance(value, list) else str(value)


def metadata_clean_text(raw_text) -> str:
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if not text.endswith((".", "!", "?")):
        text += "."

    return text


def metadata_sent_process(raw) -> str:
    sentence = ""
    if isinstance(raw, float):
        sentence = f"{raw}."
    elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        for values in raw:
            for value in values:
                sentence += metadata_clean_text(value)[:-1]
                sentence += ", "
        sentence = sentence[:-2]
        sentence += "."
    elif isinstance(raw, list):
        for value in raw:
            sentence += metadata_clean_text(value)
    else:
        sentence = metadata_clean_text(str(raw))
    return sentence + " "


def build_metadata_sentence(meta_data: dict) -> str:
    sentence = ""
    keys = set(meta_data.keys())
    for field in METADATA_SENTENCE_FIELDS:
        if field in keys:
            sentence += metadata_sent_process(meta_data[field])
    return sentence
