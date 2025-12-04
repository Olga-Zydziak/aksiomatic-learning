"""
nl_rule_parser.py

Parser uproszczonego języka reguł (restricted natural language)
na AxiomDefinition kompatybilny z AxiomKernel.

Przykłady wspieranych reguł:

    If amount > 10000 and risk_score > 5 then flag = true
    If risk_score <= 2 then flag = false
    If amount > 5000 then flag

Ograniczenia (świadome, dla bezpieczeństwa):

- Słowo kluczowe: "if ... then ..."
- Operatory porównania: >, >=, <, <=, ==, !=
- Łączenie warunków: "and" (bez "or" w tej wersji)
- Część "then" może mieć:
    - flag = true
    - flag = false
    - flag          (skrót dla flag = true)

Kod jest deterministyczny, bez użycia LLM, z jasnymi błędami gdy
reguła nie pasuje do wspieranej składni.

Wymagania:
    - axiomatic_kernel.py (AxiomDefinition, VariableSchema)
    - z3-solver

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import re

from z3 import (  # type: ignore
    And,
    BoolVal,
    ExprRef,
    IntVal,
    RealVal,
    Implies,
)

from axiomatic_kernel import AxiomDefinition, VariableSchema


# -----------------------------------------------------------------------------
# Prosty AST dla reguł
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AtomicCondition:
    """Pojedynczy warunek w postaci: pole operator wartość."""

    field: str
    operator: str
    raw_value: str  # tekstowa wartość, interpretowana wg schematu


@dataclass(frozen=True)
class ParsedRule:
    """
    Zparsowana reguła:
    - lista warunków po stronie "if" (konjunkcja),
    - konkluzja po stronie "then" (bool dla zmiennej decyzyjnej).
    """

    conditions: List[AtomicCondition]
    decision_field: str
    decision_value: bool


# -----------------------------------------------------------------------------
# Wyjątki specyficzne dla parsera
# -----------------------------------------------------------------------------

class RuleParseError(ValueError):
    """Błąd parsowania reguły w języku naturalnym."""


# -----------------------------------------------------------------------------
# Parsowanie tekstu reguły do AST (ParsedRule)
# -----------------------------------------------------------------------------

_IF_THEN_SPLIT_RE = re.compile(r"^\s*if\s+(?P<cond>.+?)\s+then\s+(?P<then>.+)\s*$",
                               flags=re.IGNORECASE)
_AND_SPLIT_RE = re.compile(r"\s+and\s+", flags=re.IGNORECASE)

_ATOMIC_COND_RE = re.compile(
    r"^\s*(?P<field>[a-zA-Z_][a-zA-Z0-9_]*)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<value>.+?)\s*$"
)

_DECISION_RE = re.compile(
    r"^\s*(?P<field>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\s*(?:=|is)\s*(?P<value>true|false))?\s*$",
    flags=re.IGNORECASE,
)


def parse_nl_rule(text: str) -> ParsedRule:
    """
    Parsuje uproszczoną regułę w stylu natural language do ParsedRule.

    Obsługiwane formy:

        If amount > 10000 and risk_score > 5 then flag = true
        If risk_score <= 2 then flag = false
        If amount > 5000 then flag

    Zwraca:
        ParsedRule

    Podnosi:
        RuleParseError – jeśli tekst nie pasuje do wspieranej składni.
    """
    match = _IF_THEN_SPLIT_RE.match(text)
    if not match:
        raise RuleParseError(
            "Rule must be of the form: 'If <conditions> then <decision>'. "
            f"Got: {text!r}"
        )

    cond_text = match.group("cond").strip()
    then_text = match.group("then").strip()

    # 1. Parsowanie części warunkowej (IF ...)
    condition_parts = _AND_SPLIT_RE.split(cond_text)
    conditions: List[AtomicCondition] = []

    for part in condition_parts:
        part = part.strip()
        if not part:
            continue

        m_cond = _ATOMIC_COND_RE.match(part)
        if not m_cond:
            raise RuleParseError(
                "Each condition must look like 'field op value', e.g. "
                "'amount > 10000'. "
                f"Problematic segment: {part!r}"
            )

        field = m_cond.group("field")
        op = m_cond.group("op")
        value = m_cond.group("value").strip()

        conditions.append(AtomicCondition(field=field, operator=op, raw_value=value))

    if not conditions:
        raise RuleParseError("At least one condition is required in the IF part.")

    # 2. Parsowanie części decyzyjnej (THEN ...)
    m_decision = _DECISION_RE.match(then_text)
    if not m_decision:
        raise RuleParseError(
            "Decision must be of the form 'flag', 'flag = true' "
            "or 'flag = false'. "
            f"Got: {then_text!r}"
        )

    decision_field = m_decision.group("field")
    raw_decision_val = m_decision.group("value")

    if raw_decision_val is None:
        # 'flag' -> domyślnie True
        decision_value = True
    else:
        lowered = raw_decision_val.lower()
        decision_value = lowered == "true"

    return ParsedRule(
        conditions=conditions,
        decision_field=decision_field,
        decision_value=decision_value,
    )


# -----------------------------------------------------------------------------
# Interpretacja AST + schema → Z3 constraint (Implies(...))
# -----------------------------------------------------------------------------

def _to_z3_literal(
    raw_value: str,
    field_name: str,
    field_type: str,
) -> ExprRef:
    """
    Konwertuje tekstową wartość na literę Z3
    zgodnie z typem zmiennej ze schematu.
    """
    if field_type == "int":
        try:
            return IntVal(int(raw_value))
        except ValueError as exc:
            raise RuleParseError(
                f"Value {raw_value!r} for int field '{field_name}' "
                "is not a valid integer."
            ) from exc

    if field_type == "real":
        try:
            return RealVal(float(raw_value))
        except ValueError as exc:
            raise RuleParseError(
                f"Value {raw_value!r} for real field '{field_name}' "
                "is not a valid real (float)."
            ) from exc

    if field_type == "bool":
        lowered = raw_value.lower()
        if lowered in {"true", "1", "yes"}:
            return BoolVal(True)
        if lowered in {"false", "0", "no"}:
            return BoolVal(False)
        raise RuleParseError(
            f"Value {raw_value!r} for bool field '{field_name}' is not "
            "a valid boolean literal (true/false)."
        )

    raise RuleParseError(f"Unsupported field type: {field_type!r} for '{field_name}'.")


def _build_condition_expr(
    parsed: ParsedRule,
    schema_by_name: Dict[str, VariableSchema],
    vars_z3: Dict[str, ExprRef],
) -> ExprRef:
    """
    Buduje koniunkcję warunków (AND) na podstawie ParsedRule
    i schematu zmiennych.
    """
    z3_conditions: List[ExprRef] = []

    for cond in parsed.conditions:
        if cond.field not in schema_by_name:
            raise RuleParseError(
                f"Field {cond.field!r} used in condition is not present "
                "in the schema."
            )

        var_spec = schema_by_name[cond.field]
        var_z3 = vars_z3.get(cond.field)
        if var_z3 is None:
            raise RuleParseError(
                f"Internal error: Z3 variable for field '{cond.field}' "
                "not found in vars_z3."
            )

        lit = _to_z3_literal(cond.raw_value, cond.field, var_spec.type)

        op = cond.operator
        if op == ">":
            z3_expr = var_z3 > lit
        elif op == ">=":
            z3_expr = var_z3 >= lit
        elif op == "<":
            z3_expr = var_z3 < lit
        elif op == "<=":
            z3_expr = var_z3 <= lit
        elif op == "==":
            z3_expr = var_z3 == lit
        elif op == "!=":
            z3_expr = var_z3 != lit
        else:
            raise RuleParseError(
                f"Unsupported operator {op!r} in condition on field "
                f"'{cond.field}'."
            )

        z3_conditions.append(z3_expr)

    if not z3_conditions:
        raise RuleParseError("No valid conditions parsed.")

    if len(z3_conditions) == 1:
        return z3_conditions[0]

    return And(*z3_conditions)


def build_axiom_from_nl(
    rule_id: str,
    text: str,
    schema: Iterable[VariableSchema],
    decision_field_fallback: Optional[str] = None,
    description: Optional[str] = None,
    priority: int = 0,
) -> AxiomDefinition:
    """
    Tworzy AxiomDefinition z tekstu reguły w stylu natural language.

    Parametry:
        rule_id:
            Identyfikator reguły (unikalny w AxiomKernel).
        text:
            Reguła w stylu:
                "If amount > 10000 and risk_score > 5 then flag = true"
        schema:
            Schemat zmiennych (z AxiomKernel).
        decision_field_fallback:
            Jeśli nie chcesz, żeby użytkownik podawał nazwę zmiennej
            decyzyjnej (np. zawsze "flag"), możesz tu podać nazwę i
            parser zweryfikuje zgodność.
        description:
            Opcjonalny opis reguły; jeśli None, zostanie użyty `text`.
        priority:
            Priorytet reguły (przekazywany do AxiomDefinition).

    Zwraca:
        AxiomDefinition – gotowy do dodania do AxiomKernel.

    Podnosi:
        RuleParseError – jeśli tekst nie jest poprawny względem gramatyki
        lub schematu.
    """
    schema_by_name: Dict[str, VariableSchema] = {v.name: v for v in schema}
    parsed = parse_nl_rule(text)

    # Weryfikacja pola decyzyjnego
    if parsed.decision_field not in schema_by_name:
        raise RuleParseError(
            f"Decision field {parsed.decision_field!r} is not defined "
            "in the schema."
        )

    if decision_field_fallback is not None and parsed.decision_field != decision_field_fallback:
        raise RuleParseError(
            f"Decision field in rule ({parsed.decision_field!r}) "
            f"does not match expected decision field "
            f"{decision_field_fallback!r}."
        )

    decision_spec = schema_by_name[parsed.decision_field]
    if decision_spec.type != "bool":
        raise RuleParseError(
            f"Decision field {parsed.decision_field!r} must be of type 'bool', "
            f"but schema defines it as {decision_spec.type!r}."
        )

    # build_constraint będzie closurą korzystającą z parsed + schema_by_name
    def _build_constraint(vars_z3: Dict[str, ExprRef]) -> ExprRef:
        condition_expr = _build_condition_expr(parsed, schema_by_name, vars_z3)

        dec_var = vars_z3.get(parsed.decision_field)
        if dec_var is None:
            raise RuleParseError(
                f"Internal error: decision variable '{parsed.decision_field}' "
                "not present in vars_z3."
            )

        dec_literal = BoolVal(parsed.decision_value)
        decision_expr = dec_var == dec_literal

        return Implies(condition_expr, decision_expr)

    ax_description = description or text

    return AxiomDefinition(
        id=rule_id,
        description=ax_description,
        build_constraint=_build_constraint,
        priority=priority,
    )


# -----------------------------------------------------------------------------
# Demo: jak tego użyć razem z AxiomKernel
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Prosty demo-use-case, żebyś mogła od razu odpalić:
    # python nl_rule_parser.py
    from axiomatic_kernel import (
        AxiomKernel,
        VariableSchema,
        DecisionLogger,
    )

    # 1. Definiujemy schemat taki jak w kernelu AML
    demo_schema = [
        VariableSchema(
            name="amount",
            type="int",
            description="Transaction amount (in minor units).",
        ),
        VariableSchema(
            name="risk_score",
            type="int",
            description="Risk score from ML model (0-10).",
        ),
        VariableSchema(
            name="flag",
            type="bool",
            description="Decision whether to flag the transaction.",
        ),
    ]

    # 2. Budujemy kernel
    kernel = AxiomKernel(
        schema=demo_schema,
        decision_variable="flag",
        logger=DecisionLogger("logs/nl_rules_demo.jsonl"),
        rule_version="aml_rules_v1.1.0_nl",
    )

    # 3. Natural-language-style reguły (w naszej prostszej składni)
    rule_text_1 = "If amount > 10000 and risk_score > 5 then flag = true"
    rule_text_2 = "If risk_score <= 2 then flag = false"

    axiom1 = build_axiom_from_nl(
        rule_id="high_risk_flag_nl",
        text=rule_text_1,
        schema=demo_schema,
        decision_field_fallback="flag",
    )

    axiom2 = build_axiom_from_nl(
        rule_id="low_risk_clear_nl",
        text=rule_text_2,
        schema=demo_schema,
        decision_field_fallback="flag",
    )

    kernel.add_axiom(axiom1)
    kernel.add_axiom(axiom2)

    # 4. Przykładowy case
    case = {
        "amount": 15_000,
        "risk_score": 7,
    }

    bundle = kernel.evaluate(case)

    print("\n=== Proof-Carrying Decision (NL rule parser demo) ===")
    print(json.dumps(bundle, indent=2, ensure_ascii=False))
