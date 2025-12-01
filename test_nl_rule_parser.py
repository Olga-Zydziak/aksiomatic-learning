"""
test_nl_rule_parser.py

Testy parsera uproszczonego języka reguł (nl_rule_parser):
- poprawne parsowanie składni IF ... THEN ...,
- integracja z AxiomKernel (build_axiom_from_nl),
- przypadki SAT (FLAGGED / CLEAN),
- przypadek błędnej reguły (RuleParseError).

Uruchamianie:
    pytest test_nl_rule_parser.py
albo:
    python test_nl_rule_parser.py
"""

from __future__ import annotations

from typing import Any, Dict

import json

import pytest

from axiomatic_kernel import (
    AxiomKernel,
    DecisionLogger,
    VariableSchema,
)
from nl_rule_parser import (
    ParsedRule,
    RuleParseError,
    build_axiom_from_nl,
    parse_nl_rule,
)


def _demo_schema() -> list[VariableSchema]:
    """Wspólny schemat dla testów parsera NL."""
    return [
        VariableSchema(
            name="amount",
            type="int",
            description="Transaction amount.",
        ),
        VariableSchema(
            name="risk_score",
            type="int",
            description="Risk score (0-10).",
        ),
        VariableSchema(
            name="flag",
            type="bool",
            description="Decision whether to flag.",
        ),
    ]


def test_parse_simple_and_rule() -> None:
    """
    Sprawdza parsowanie:
        If amount > 10000 and risk_score > 5 then flag = true
    """
    text = "If amount > 10000 and risk_score > 5 then flag = true"
    parsed: ParsedRule = parse_nl_rule(text)

    assert parsed.decision_field == "flag"
    assert parsed.decision_value is True
    assert len(parsed.conditions) == 2

    cond1 = parsed.conditions[0]
    cond2 = parsed.conditions[1]

    assert (cond1.field, cond1.operator, cond1.raw_value) == (
        "amount",
        ">",
        "10000",
    )
    assert (cond2.field, cond2.operator, cond2.raw_value) == (
        "risk_score",
        ">",
        "5",
    )


def test_parse_rule_without_explicit_boolean_value_defaults_to_true() -> None:
    """
    Sprawdza parsowanie:
        If amount > 5000 then flag
    Powinno być interpretowane jako flag = true.
    """
    text = "If amount > 5000 then flag"
    parsed: ParsedRule = parse_nl_rule(text)

    assert parsed.decision_field == "flag"
    assert parsed.decision_value is True
    assert len(parsed.conditions) == 1
    cond = parsed.conditions[0]
    assert (cond.field, cond.operator, cond.raw_value) == ("amount", ">", "5000")


def test_parse_invalid_rule_raises_error() -> None:
    """
    Błędna reguła (brak 'if' / 'then') powinna podnieść RuleParseError.
    """
    with pytest.raises(RuleParseError):
        parse_nl_rule("amount > 10000 flag = true")  # brak 'if' i 'then'


def test_build_axiom_from_nl_and_integration_flagged() -> None:
    """
    Reguła NL:
        If amount > 10000 and risk_score > 5 then flag = true
    powinna po zbudowaniu AxiomDefinition i dodaniu do kernela
    prowadzić do decyzji FLAGGED dla case amount=15000, risk_score=7.
    """
    schema = _demo_schema()
    kernel = AxiomKernel(
        schema=schema,
        decision_variable="flag",
        logger=None,
        rule_version="nl_test_v1",
    )

    rule_text = "If amount > 10000 and risk_score > 5 then flag = true"
    axiom = build_axiom_from_nl(
        rule_id="nl_high_risk_flag",
        text=rule_text,
        schema=schema,
        decision_field_fallback="flag",
    )
    kernel.add_axiom(axiom)

    case = {"amount": 15_000, "risk_score": 7}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "SAT"
    assert bundle["decision"] == "FLAGGED"
    assert bundle["model"]["flag"] is True
    assert any(ax["id"] == "nl_high_risk_flag" for ax in bundle["satisfied_axioms"])


def test_build_axiom_from_nl_and_integration_clean() -> None:
    """
    Reguła NL:
        If risk_score <= 2 then flag = false
    powinna skutkować decyzją CLEAN dla risk_score=1.
    """
    schema = _demo_schema()
    kernel = AxiomKernel(
        schema=schema,
        decision_variable="flag",
        logger=None,
        rule_version="nl_test_v1",
    )

    rule_text = "If risk_score <= 2 then flag = false"
    axiom = build_axiom_from_nl(
        rule_id="nl_low_risk_clear",
        text=rule_text,
        schema=schema,
        decision_field_fallback="flag",
    )
    kernel.add_axiom(axiom)

    case = {"amount": 500, "risk_score": 1}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "SAT"
    assert bundle["decision"] == "CLEAN"
    assert bundle["model"]["flag"] is False
    assert any(ax["id"] == "nl_low_risk_clear" for ax in bundle["satisfied_axioms"])


def test_build_axiom_from_nl_with_unknown_field_raises() -> None:
    """
    Reguła odnosząca się do pola, którego nie ma w schemacie,
    powinna podnieść RuleParseError.
    """
    schema = _demo_schema()

    rule_text = "If unknown_field > 10 then flag = true"

    with pytest.raises(RuleParseError):
        build_axiom_from_nl(
            rule_id="bad_field_rule",
            text=rule_text,
            schema=schema,
            decision_field_fallback="flag",
        )


def test_nl_parser_demo_with_logger(tmp_path) -> None:
    """
    Mini-test end-to-end: reguły NL + kernel + logger.
    """
    from nl_rule_parser import build_axiom_from_nl  # lokalny import dla czytelności

    log_file = tmp_path / "nl_decisions.jsonl"
    logger = DecisionLogger(log_file)

    schema = _demo_schema()
    kernel = AxiomKernel(
        schema=schema,
        decision_variable="flag",
        logger=logger,
        rule_version="nl_test_v2",
    )

    rule_text_1 = "If amount > 10000 and risk_score > 5 then flag = true"
    rule_text_2 = "If risk_score <= 2 then flag = false"

    axiom1 = build_axiom_from_nl(
        rule_id="nl_high_risk_flag",
        text=rule_text_1,
        schema=schema,
        decision_field_fallback="flag",
    )
    axiom2 = build_axiom_from_nl(
        rule_id="nl_low_risk_clear",
        text=rule_text_2,
        schema=schema,
        decision_field_fallback="flag",
    )

    kernel.add_axiom(axiom1)
    kernel.add_axiom(axiom2)

    case = {"amount": 15_000, "risk_score": 7}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "SAT"
    assert bundle["decision"] == "FLAGGED"
    assert "decision_id" in bundle
    assert "logged_at_utc" in bundle

    # log powinien zawierać decyzję
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["decision"]["decision_status"] == "SAT"
    assert record["decision"]["decision"] == "FLAGGED"


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
