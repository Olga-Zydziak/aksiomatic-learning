"""
test_axiomatic_kernel.py

Testy jądra AxiomKernel:
- przypadki SAT (FLAGGED / CLEAN),
- przypadek UNSAT (sprzeczne reguły),
- obsługa błędnych danych wejściowych.

Uruchamianie:
    pytest test_axiomatic_kernel.py
albo:
    python test_axiomatic_kernel.py
"""

from __future__ import annotations

from typing import Any, Dict

import json

import pytest

from axiomatic_kernel import (
    AxiomDefinition,
    AxiomKernel,
    DecisionLogger,
    VariableSchema,
)


def _build_test_kernel(logger: DecisionLogger | None = None) -> AxiomKernel:
    """Tworzy minimalny kernel testowy dla reguł AML-like."""
    schema = [
        VariableSchema(
            name="amount",
            type="int",
            description="Transaction amount in minor units.",
        ),
        VariableSchema(
            name="risk_score",
            type="int",
            description="Risk score from 0 to 10.",
        ),
        VariableSchema(
            name="flag",
            type="bool",
            description="Decision whether to flag the transaction.",
        ),
    ]
    return AxiomKernel(
        schema=schema,
        decision_variable="flag",
        logger=logger,
        rule_version="test_rules_v1.0.0",
    )


def test_high_risk_transaction_is_flagged() -> None:
    """
    Jeśli amount > 10000 i risk_score > 5 -> transakcja powinna być oflagowana.
    """
    kernel = _build_test_kernel()

    def rule_high_risk(vars_z3: Dict[str, Any]) -> Any:
        amount = vars_z3["amount"]
        risk = vars_z3["risk_score"]
        flag = vars_z3["flag"]
        from z3 import And, Implies  # type: ignore

        return Implies(And(amount > 10_000, risk > 5), flag)

    kernel.add_axiom(
        AxiomDefinition(
            id="high_risk_flag",
            description="High-risk transactions must be flagged.",
            build_constraint=rule_high_risk,
            priority=10,
        )
    )

    case = {"amount": 15_000, "risk_score": 7}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "SAT"
    assert bundle["decision"] == "FLAGGED"
    assert bundle["model"]["flag"] is True
    assert any(ax["id"] == "high_risk_flag" for ax in bundle["satisfied_axioms"])
    assert bundle["violated_axioms"] == []


def test_low_risk_transaction_is_clean() -> None:
    """
    Jeśli risk_score <= 2 -> transakcja powinna być CLEAN.
    """
    kernel = _build_test_kernel()

    def rule_low_risk(vars_z3: Dict[str, Any]) -> Any:
        risk = vars_z3["risk_score"]
        flag = vars_z3["flag"]
        from z3 import Implies  # type: ignore

        return Implies(risk <= 2, flag == False)

    kernel.add_axiom(
        AxiomDefinition(
            id="low_risk_clear",
            description="Very low-risk transactions must not be flagged.",
            build_constraint=rule_low_risk,
            priority=5,
        )
    )

    case = {"amount": 1_000, "risk_score": 1}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "SAT"
    assert bundle["decision"] == "CLEAN"
    assert bundle["model"]["flag"] is False
    assert any(ax["id"] == "low_risk_clear" for ax in bundle["satisfied_axioms"])
    assert bundle["violated_axioms"] == []


def test_conflicting_rules_produce_unsat_and_conflicting_axioms() -> None:
    """
    Dwie sprzeczne reguły dla tego samego zakresu -> UNSAT + wskazanie konfliktu.

    Rule1: jeśli amount > 10000 -> flag = True
    Rule2: jeśli amount > 10000 -> flag = False

    Dla amount = 15000 system musi zgłosić UNSAT i wskazać obie reguły jako konfliktujące.
    """
    kernel = _build_test_kernel()

    from z3 import Implies  # type: ignore

    def rule_flag_true(vars_z3: Dict[str, Any]) -> Any:
        amount = vars_z3["amount"]
        flag = vars_z3["flag"]
        return Implies(amount > 10_000, flag == True)

    def rule_flag_false(vars_z3: Dict[str, Any]) -> Any:
        amount = vars_z3["amount"]
        flag = vars_z3["flag"]
        return Implies(amount > 10_000, flag == False)

    kernel.add_axiom(
        AxiomDefinition(
            id="amount_flag_true",
            description="If amount > 10000 then flag must be True.",
            build_constraint=rule_flag_true,
            priority=10,
        )
    )
    kernel.add_axiom(
        AxiomDefinition(
            id="amount_flag_false",
            description="If amount > 10000 then flag must be False.",
            build_constraint=rule_flag_false,
            priority=10,
        )
    )

    case = {"amount": 15_000, "risk_score": 5}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "UNSAT"
    assert bundle["decision"] == "ERROR"
    # obie reguły powinny pojawić się jako konfliktujące
    assert set(bundle["conflicting_axioms"]) == {
        "amount_flag_true",
        "amount_flag_false",
    }


def test_invalid_input_type_results_in_error_bundle() -> None:
    """
    Jeśli case ma zły typ (np. amount='abc'), kernel powinien zwrócić status ERROR, a nie crashować.
    """
    kernel = _build_test_kernel()

    def rule_dummy(vars_z3: Dict[str, Any]) -> Any:
        # jakakolwiek poprawna reguła
        amount = vars_z3["amount"]
        flag = vars_z3["flag"]
        from z3 import Implies  # type: ignore

        return Implies(amount > 0, flag == False)

    kernel.add_axiom(
        AxiomDefinition(
            id="dummy_rule",
            description="Dummy rule for testing invalid input.",
            build_constraint=rule_dummy,
            priority=0,
        )
    )

    # amount ma zły typ (string zamiast int)
    case = {"amount": "not_a_number", "risk_score": 3}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "ERROR"
    assert bundle["decision"] == "ERROR"
    # oczekujemy komunikatu o błędnym typie
    assert "expected int" in bundle["error"].lower()


def test_logger_persists_decision_id_and_timestamp(tmp_path) -> None:
    """
    Sprawdza, że DecisionLogger zapisuje rekord i kernel uzupełnia decision_id oraz logged_at_utc.
    """
    log_file = tmp_path / "decisions.jsonl"
    logger = DecisionLogger(log_file)

    kernel = _build_test_kernel(logger=logger)

    def rule_always_flag(vars_z3: Dict[str, Any]) -> Any:
        flag = vars_z3["flag"]
        from z3 import Implies  # type: ignore

        return Implies(True, flag == True)

    kernel.add_axiom(
        AxiomDefinition(
            id="always_flag",
            description="Always flag (for logging test).",
            build_constraint=rule_always_flag,
            priority=0,
        )
    )

    case = {"amount": 100, "risk_score": 1}
    bundle = kernel.evaluate(case)

    assert bundle["decision_status"] == "SAT"
    assert "decision_id" in bundle
    assert "logged_at_utc" in bundle

    # plik logu istnieje i ma co najmniej jeden wiersz
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 1
    record = json.loads(content[-1])
    assert record["decision_id"] == bundle["decision_id"]


if __name__ == "__main__":  # pozwala odpalić testy także przez `python ...`
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))
