"""Masking of personal data in free text.

Used before training and before anything is logged, so that email addresses,
phone numbers, URLs and long account or card numbers never reach a model
artifact or a log file.
"""

from __future__ import annotations

import re

EMAIL_TOKEN = "[EMAIL]"
URL_TOKEN = "[URL]"
PHONE_TOKEN = "[PHONE]"
NUMBER_TOKEN = "[NUMBER]"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
# Candidate phone numbers: digits with common separators. Only runs with at
# least nine digits are masked, so dates and short quantities are kept.
_PHONE_CANDIDATE = re.compile(r"(?<![\w-])\+?\d[\d\s().-]{6,}\d(?![\w-])")
_LONG_DIGITS = re.compile(r"\b\d{6,}\b")
_MIN_PHONE_DIGITS = 9


def _mask_phone(match: re.Match[str]) -> str:
    text = match.group(0)
    digits = sum(ch.isdigit() for ch in text)
    return PHONE_TOKEN if digits >= _MIN_PHONE_DIGITS else text


def mask_pii(text: str) -> str:
    """Replace emails, URLs, phone numbers and long digit strings with tokens."""
    text = _EMAIL.sub(EMAIL_TOKEN, text)
    text = _URL.sub(URL_TOKEN, text)
    text = _PHONE_CANDIDATE.sub(_mask_phone, text)
    return _LONG_DIGITS.sub(NUMBER_TOKEN, text)
