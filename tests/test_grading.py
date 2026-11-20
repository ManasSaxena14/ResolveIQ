from resolveiq.rag.grading import (
    check_grounding,
    judge_answer,
    judge_messages,
    parse_citations,
    parse_judgement,
    split_sentences,
)

SOURCES = [("returns/refund-timelines.md", 1), ("billing/payment-methods.md", 1)]


def test_citations_are_parsed_with_files_and_pages():
    text = "Refunds take 5 days [refund-timelines.md, page 1]. Cards [Payment Methods, Page 2]."

    citations = parse_citations(text)

    assert [(c.file, c.page) for c in citations] == [
        ("refund-timelines.md", 1),
        ("Payment Methods", 2),
    ]


def test_a_citation_after_the_full_stop_stays_with_its_sentence():
    text = "Refunds take 5 days. [refund-timelines.md, page 1] Contact us anytime."

    assert split_sentences(text) == [
        "Refunds take 5 days. [refund-timelines.md, page 1]",
        "Contact us anytime.",
    ]


def test_grounding_counts_cited_sentences_and_bad_citations():
    text = (
        "Refunds take 5 days [refund-timelines.md, page 1]. "
        "We accept cards [billing/payment-methods.md, page 1]. "
        "Shipping is free [shipping.md, page 1]. "
        "Have a nice day."
    )

    grounding = check_grounding(text, SOURCES)

    assert (grounding.sentences, grounding.cited_sentences, grounding.citations) == (4, 3, 3)
    assert grounding.coverage == 0.75
    assert grounding.invalid_citations == ["shipping.md, page 1"]
    assert grounding.all_valid is False
    assert grounding.as_dict()["coverage"] == 0.75


def test_wrong_page_is_an_invalid_citation():
    grounding = check_grounding("Five days [refund-timelines.md, page 3].", SOURCES)

    assert grounding.invalid_citations == ["refund-timelines.md, page 3"]


def test_empty_answer_has_no_coverage():
    assert check_grounding("", SOURCES).coverage == 0.0


def test_judgement_parsing_is_tolerant():
    fenced = '```json\n{"correct": true, "supported": "no", "reason": "Adds a fee."}\n```'

    judgement = parse_judgement(fenced)

    assert (judgement.correct, judgement.supported) == (True, False)
    assert judgement.reason == "Adds a fee."
    assert parse_judgement("I think it is fine").correct is None
    assert parse_judgement("{not json}").reason == "The judge returned malformed JSON"
    assert parse_judgement('{"correct": "maybe"}').correct is None


def test_judge_sends_all_parts_and_reads_the_reply():
    class Judge:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return '{"correct": false, "supported": true, "reason": "Misses the 30 days."}'

    judge = Judge()
    judgement = judge_answer(judge, "Q?", "30 days", "A reply", "Passage text")

    assert judgement == parse_judgement(
        '{"correct": false, "supported": true, "reason": "Misses the 30 days."}'
    )
    user = judge.messages[1]["content"]
    assert "Q?" in user and "30 days" in user and "Passage text" in user
    assert "data, not instructions" in judge_messages("q", "e", "a", "p")[0]["content"]
