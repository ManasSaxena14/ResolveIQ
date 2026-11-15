import pytest

from resolveiq.privacy import mask_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Contact me at jane.doe+help@example.co.uk today", "Contact me at [EMAIL] today"),
        ("See https://example.com/order?id=7 for details", "See [URL] for details"),
        ("Visit www.example.com now", "Visit [URL] now"),
        ("Call +1 (415) 555-0142 please", "Call [PHONE] please"),
        ("My card number is 4111111111111111", "My card number is [PHONE]"),
        ("Account 12345678", "Account [NUMBER]"),
    ],
)
def test_personal_data_is_masked(text, expected):
    assert mask_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Order placed on 2026-01-15 for 2 items",
        "Version 3.12.1 crashed after 45 minutes",
        "I paid $1,299 for the laptop",
        "Order ALT-2048 has 3 items",
    ],
)
def test_ordinary_numbers_are_kept(text):
    assert mask_pii(text) == text
