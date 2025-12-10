"""
nl_rule_parser.py

Parser uproszczonego języka reguł (restricted natural language)
na AxiomDefinition kompatybilny z AxiomKernel.

Przykłady wspieranych reguł:

    If amount > 10000 and risk_score > 5 then flag = true
    If risk_score <= 2 then flag = false
    If amount > 5000 then flag

Rozszerzenia względem minimalnej wersji:

- obsługa aliasów pól (np. "kwota" → "amount", "ryzyko" → "risk_score",
  "flaga" → "flag"),
- normalizacja wartości liczbowych z sufiksami k/m (np. "10k" → 10000),
- prosty preprocessing tekstu "prawie naturalnego", np.:
    "Flag transactions over 10k with high risk"
    → "If amount > 10k and risk_score > 7 then flag = true"

Ograniczenia (świadome, dla bezpieczeństwa):

- słowo kluczowe: "if ... then ...",
- operatory porównania: >, >=, <, <=, ==, !=,
- łączenie warunków: tylko "and",
- część "then" może mieć:
    * flag = true
    * flag = false
    * flag          (skrót dla flag = true)

Kod jest deterministyczny, bez użycia LLM, z jasnymi błędami gdy
reguła nie pasuje do wspieranej składni.

Wymagania:
    - axiomatic_kernel.py (AxiomDefinition, VariableSchema)
    - z3-solver
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

import re

from z3 import (  # type: ignore[import]
    And,
    BoolVal,
    ExprRef,
    IntVal,
    RealVal,
    Implies,
)

from axiomatic_kernel import AxiomDefinition, VariableSchema


# =============================================================================
# Prosty AST dla reguł
# =============================================================================


@dataclass(frozen=True)
class AtomicCondition:
    """Pojedynczy warunek w postaci: pole operator wartość."""

    field: str
    operator: str
    raw_value: str  # tekstowa wartość, interpretowana później wg schematu


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


# =============================================================================
# Wyjątki specyficzne dla parsera
# =============================================================================


class RuleParseError(ValueError):
    """Błąd parsowania lub interpretacji reguły w języku naturalnym."""


# =============================================================================
# Funkcje pomocnicze: aliasy pól, normalizacja liczb, preprocessing tekstu
# =============================================================================

_ALIAS_MAP: Dict[str, str] = {
    # amount
    "kwota": "amount",
    "suma": "amount",
    "wartosc": "amount",
    "wartość": "amount",
    "amount": "amount",
    # risk_score
    "ryzyko": "risk_score",
    "risk": "risk_score",
    "risk_score": "risk_score",
    # flag / decision
    "flaga": "flag",
    "flag": "flag",
}


def normalize_field_name(field: str, schema_fields: Iterable[str]) -> str:
    """
    Zwraca kanoniczną nazwę pola na podstawie aliasów i schematu.

    Przykłady:
        normalize_field_name("kwota", ["amount", "risk_score", "flag"]) -> "amount"
        normalize_field_name("FLAGA", ["amount", "risk_score", "flag"]) -> "flag"

    Jeśli po zastosowaniu aliasów pole nie występuje w schemacie,
    podnosi RuleParseError.
    """
    canonical_fields = {name for name in schema_fields}
    key = field.strip().lower()

    # mapowanie aliasów
    mapped = _ALIAS_MAP.get(key, field.strip())
    if mapped not in canonical_fields:
        raise RuleParseError(
            f"Field {field!r} is not defined in the schema "
            f"(after alias resolution got {mapped!r})."
        )
    return mapped


def normalize_numeric_value(raw_value: str, field_name: str) -> str:
    """
    Normalizuje wartości liczbowe z prostymi sufiksami i separatorami.

    Obsługiwane:
        - sufiksy k/K (tysiące), np. "10k" -> "10000",
        - sufiksy m/M (miliony), np. "1.5m" -> "1500000",
        - separator przecinkowy zamiast kropki, np. "1,5m" -> "1500000",
        - spacje na brzegach są ignorowane.

    Zwraca tekstową reprezentację liczby całkowitej, gotową do parsowania
    przez int(...). Jeśli nie wykryje sufiksu, zwraca oczyszczony tekst
    bez spacji na brzegach.
    """
    text = raw_value.strip()
    if not text:
        raise RuleParseError(
            f"Empty numeric value for field {field_name!r} is not allowed."
        )

    # Rozdziel sufiks k/m (case-insensitive)
    match = re.fullmatch(r"([0-9][0-9_.,]*)\s*([kKmM])?", text)
    if not match:
        # Nic sprytnego – zwracamy oryginał bez spacji i pozwalamy
        # standardowej konwersji zgłosić błąd, jeśli to nie liczba.
        return text

    number_part, suffix = match.groups()
    # Zamiana przecinka na kropkę, usunięcie podkreśleń
    number_part = number_part.replace("_", "").replace(",", ".")
    if suffix is None:
        # Bez sufiksu – zostawiamy jak jest
        return number_part

    try:
        base = float(number_part)
    except ValueError as exc:  # pragma: no cover - defensywne
        raise RuleParseError(
            f"Value {raw_value!r} for field {field_name!r} "
            "is not a valid numeric literal."
        ) from exc

    multiplier = 1_000 if suffix.lower() == "k" else 1_000_000
    value = int(base * multiplier)
    return str(value)


def preprocess_natural(text: str) -> str:
    """
    Przetwarza uproszczony, prawie naturalny tekst na kanoniczny DSL
    w stylu "If ... then ...".

    Jeśli tekst już zaczyna się od "if" (case-insensitive), jest zwracany
    bez zmian.

    Przykłady wspierane scenariusze (bez użycia LLM, tylko reguły):

        "Flag transactions over 10k with high risk"
            → "If amount > 10k and risk_score > 7 then flag = true"

        "Clear all low-risk transactions"
            → "If risk_score <= 3 then flag = false"

    Jeśli tekst nie pasuje do żadnego wspieranego wzorca, zostaje zwrócony
    bez zmian – dalsze etapy (parse_nl_rule) zadecydują, czy to poprawna reguła.
    """
    stripped = text.strip()
    if not stripped:
        raise RuleParseError("Rule text is empty.")

    # Jeśli to już wygląda na nasz DSL, nic nie robimy.
    if stripped.lower().startswith("if "):
        return stripped

    lower = stripped.lower()

    # Wzorzec: "Flag transactions over 10k with high risk"
    if "flag" in lower and "transactions" in lower and "over" in lower:
        # Szukamy progu kwoty po słowie "over"
        amount_match = re.search(
            r"over\s+([0-9][0-9_.,kKmM]*)",
            stripped,
            flags=re.IGNORECASE,
        )
        if amount_match:
            amount_token = amount_match.group(1).strip()
            # zachowujemy dokładny token (np. "10k"), bez wymuszania przeliczenia,
            # bo DSL i tak obsłuży sufiksy k/m na dalszym etapie
            return (
                f"If amount > {amount_token} and risk_score > 7 "
                "then flag = true"
            )

    # Wzorzec: "Clear all low-risk transactions"
    if "clear" in lower and "low-risk" in lower and "transactions" in lower:
        return "If risk_score <= 3 then flag = false"

    # Brak dopasowania – zwracamy oryginał
    return stripped


# =============================================================================
# Parsowanie tekstu reguły do AST (ParsedRule)
# =============================================================================

_IF_THEN_SPLIT_RE = re.compile(
    r"^\s*if\s+(?P<cond>.+?)\s+then\s+(?P<then>.+)\s*$",
    flags=re.IGNORECASE,
)
_AND_SPLIT_RE = re.compile(r"\s+and\s+", flags=re.IGNORECASE)

_ATOMIC_COND_RE = re.compile(
    r"^\s*(?P<field>[a-zA-Z_][a-zA-Z0-9_]*)\s*"
    r"(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<value>.+?)\s*$",
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
        segment = part.strip()
        if not segment:
            continue

        cond_match = _ATOMIC_COND_RE.match(segment)
        if not cond_match:
            raise RuleParseError(
                "Each condition must look like 'field op value', e.g. "
                "'amount > 10000'. "
                f"Problematic segment: {segment!r}"
            )

        field = cond_match.group("field")
        op = cond_match.group("op")
        value = cond_match.group("value").strip()

        conditions.append(
            AtomicCondition(field=field, operator=op, raw_value=value)
        )

    if not conditions:
        raise RuleParseError("At least one condition is required in the IF part.")

    # 2. Parsowanie części decyzyjnej (THEN ...)
    decision_match = _DECISION_RE.match(then_text)
    if not decision_match:
        raise RuleParseError(
            "Decision must be of the form 'flag', 'flag = true' "
            "or 'flag = false'. "
            f"Got: {then_text!r}"
        )

    decision_field = decision_match.group("field")
    raw_decision_val = decision_match.group("value")

    if raw_decision_val is None:
        # 'flag' -> domyślnie True
        decision_value = True
    else:
        decision_value = raw_decision_val.lower() == "true"

    return ParsedRule(
        conditions=conditions,
        decision_field=decision_field,
        decision_value=decision_value,
    )


# =============================================================================
# Interpretacja AST + schema → Z3 constraint (Implies(...))
# =============================================================================


def _to_z3_literal(
    raw_value: str,
    field_name: str,
    field_type: str,
) -> ExprRef:
    """
    Konwertuje tekstową wartość na literę Z3
    zgodnie z typem zmiennej ze schematu.

    Dla pól całkowitych i rzeczywistych stosowana jest dodatkowa
    normalizacja (normalize_numeric_value), dzięki czemu wspieramy
    sufiksy k/m oraz separator przecinkowy.
    """
    if field_type == "int":
        normalized = normalize_numeric_value(raw_value, field_name)
        try:
            return IntVal(int(normalized))
        except ValueError as exc:
            raise RuleParseError(
                f"Value {raw_value!r} for int field '{field_name}' "
                "is not a valid integer."
            ) from exc

    if field_type == "real":
        text = raw_value.strip().replace("_", "").replace(",", ".")
        try:
            return RealVal(float(text))
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

    raise RuleParseError(
        f"Unsupported field type: {field_type!r} for '{field_name}'."
    )


def _build_condition_expr(
    parsed: ParsedRule,
    schema_by_name: Dict[str, VariableSchema],
    vars_z3: Dict[str, ExprRef],
) -> ExprRef:
    """
    Buduje koniunkcję warunków (AND) na podstawie ParsedRule
    i schematu zmiennych.

    W tym miejscu stosujemy normalize_field_name, aby obsłużyć aliasy.
    """
    z3_conditions: List[ExprRef] = []
    schema_fields = list(schema_by_name.keys())

    for cond in parsed.conditions:
        canonical_field = normalize_field_name(cond.field, schema_fields)

        var_spec = schema_by_name[canonical_field]
        var_z3 = vars_z3.get(canonical_field)
        if var_z3 is None:
            raise RuleParseError(
                "Internal error: Z3 variable for field "
                f"{canonical_field!r} not found in vars_z3."
            )

        lit = _to_z3_literal(cond.raw_value, canonical_field, var_spec.type)

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
        else:  # pragma: no cover - powinno być niemożliwe przy aktualnym regexie
            raise RuleParseError(
                f"Unsupported operator {op!r} in condition on field "
                f"{cond.field!r}."
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
            lub wariant "prawie naturalny", np.:
                "Flag transactions over 10k with high risk".
        schema:
            Schemat zmiennych (z AxiomKernel).
        decision_field_fallback:
            Jeśli nie chcesz, żeby użytkownik podawał nazwę zmiennej
            decyzyjnej (np. zawsze "flag"), możesz tu podać nazwę (także
            z aliasami, np. "flaga") – parser zweryfikuje zgodność.
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
    schema_list = list(schema)
    schema_by_name: Dict[str, VariableSchema] = {v.name: v for v in schema_list}
    schema_fields = list(schema_by_name.keys())

    # Najpierw próbujemy przerobić "prawie naturalny" tekst na kanoniczny DSL.
    canonical_text = preprocess_natural(text)

    parsed = parse_nl_rule(canonical_text)

    # Weryfikacja pola decyzyjnego (z obsługą aliasów).
    decision_field_canonical = normalize_field_name(
        parsed.decision_field,
        schema_fields,
    )

    if decision_field_fallback is not None:
        fallback_canonical = normalize_field_name(
            decision_field_fallback,
            schema_fields,
        )
        if decision_field_canonical != fallback_canonical:
            raise RuleParseError(
                "Decision field in rule "
                f"({parsed.decision_field!r} → {decision_field_canonical!r}) "
                "does not match expected decision field "
                f"{decision_field_fallback!r} "
                f"(→ {fallback_canonical!r})."
            )

    decision_spec = schema_by_name[decision_field_canonical]
    if decision_spec.type != "bool":
        raise RuleParseError(
            "Decision field "
            f"{decision_field_canonical!r} must be of type 'bool', "
            f"but schema defines it as {decision_spec.type!r}."
        )

    def _build_constraint(vars_z3: Dict[str, ExprRef]) -> ExprRef:
        condition_expr = _build_condition_expr(parsed, schema_by_name, vars_z3)

        dec_var = vars_z3.get(decision_field_canonical)
        if dec_var is None:
            raise RuleParseError(
                "Internal error: decision variable "
                f"{decision_field_canonical!r} not present in vars_z3."
            )

        dec_literal = BoolVal(parsed.decision_value)
        decision_expr = dec_var == dec_literal

        return Implies(condition_expr, decision_expr)

    ax_description = description or canonical_text

    return AxiomDefinition(
        id=rule_id,
        description=ax_description,
        build_constraint=_build_constraint,
        priority=priority,
    )


# =============================================================================
# Demo: jak tego użyć razem z AxiomKernel
# =============================================================================

if __name__ == "__main__":
    # Prosty demo-use-case, żeby można było od razu odpalić:
    # python nl_rule_parser.py
    import json

    from axiomatic_kernel import (  # type: ignore[import]
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
    #    w tym przykład z preprocess_natural
    rule_text_1 = "If amount > 10000 and risk_score > 5 then flag = true"
    rule_text_2 = "If risk_score <= 2 then flag = false"
    rule_text_3 = "Flag transactions over 10k with high risk"

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

    axiom3 = build_axiom_from_nl(
        rule_id="natural_high_risk_flag_nl",
        text=rule_text_3,
        schema=demo_schema,
        decision_field_fallback="flag",
    )

    kernel.add_axiom(axiom1)
    kernel.add_axiom(axiom2)
    kernel.add_axiom(axiom3)

    # 4. Przykładowy case
    case = {
        "amount": 15_000,
        "risk_score": 7,
    }

    bundle = kernel.evaluate(case)

    print("\n=== Proof-Carrying Decision (NL rule parser demo) ===")
    print(json.dumps(bundle, indent=2, ensure_ascii=False))
