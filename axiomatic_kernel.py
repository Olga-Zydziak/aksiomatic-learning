"""
axiomatic_kernel.py

Proof-Carrying Decision Kernel oparty na Z3, z dynamicznym schematem
zmiennych, obsługą konfliktów reguł (UNSAT / add_axiom_safe),
rozróżnieniem reguł aktywnych / nieaktywnych oraz audytem (logger).

Zastosowania:
- AML / fraud rules
- reguły compliance / policy
- limity kredytowe
- dowolne decyzje oparte na twardych regułach

Wymagania:
    pip install z3-solver

Autor: Ty + ChatGPT (axiomatic learning PoC)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import json
import logging
import uuid

from z3 import (  # type: ignore
    And,
    Bool,
    BoolRef,
    BoolVal,
    ExprRef,
    Implies,
    Int,
    IntNumRef,
    RatNumRef,
    Real,
    Solver,
    is_true,
    sat,
    unsat,
    unknown,
)

# -----------------------------------------------------------------------------
# Typy i modele danych
# -----------------------------------------------------------------------------

VariableType = Literal["int", "bool", "real"]


@dataclass(frozen=True)
class VariableSchema:
    """Opis pojedynczej zmiennej w modelu decyzyjnym."""

    name: str
    type: VariableType
    description: Optional[str] = None


@dataclass(frozen=True)
class AxiomDefinition:
    """
    Definicja aksjomatu/reguły.

    build_constraint:
        Funkcja przyjmująca mapę nazw zmiennych -> z3.ExprRef
        i zwracająca pojedyncze wyrażenie z3 reprezentujące regułę.

        Najwygodniej pisać reguły w formie implikacji:
            Implies(And(...warunki...), ...konkluzja...)
    """

    id: str
    description: str
    build_constraint: Callable[[Dict[str, ExprRef]], ExprRef]
    priority: int = 0  # do przyszłej strategii rozwiązywania konfliktów


@dataclass(frozen=True)
class AxiomConflict:
    """
    Opis konfliktu nowej reguły z istniejącą.

    existing_axiom_id:
        Identyfikator istniejącej reguły, z którą zachodzi konflikt.
    existing_description:
        Opis istniejącej reguły.
    overlap_example:
        Przykładowy model (przypisania zmiennych), w którym antecedenty
        obu reguł są prawdziwe (czyli "obszar konfliktu").
    """

    existing_axiom_id: str
    existing_description: str
    overlap_example: Optional[Dict[str, Any]] = None


class AxiomConflictError(ValueError):
    """
    Wyjątek rzucany przez add_axiom_safe, gdy:
    - nowa reguła jest wewnętrznie niespójna (self_inconsistent), lub
    - nowa reguła wchodzi w konflikt z istniejącymi.

    conflicts:
        Lista konfliktów z istniejącymi regułami.
    self_inconsistent:
        True, jeśli cond ∧ cons nowej reguły jest UNSAT (reguła nigdy
        nie może się aktywować bez sprzeczności).
    """

    def __init__(
        self,
        axiom_id: str,
        conflicts: List[AxiomConflict],
        self_inconsistent: bool = False,
    ) -> None:
        messages: List[str] = []
        if self_inconsistent:
            messages.append(
                f"New axiom '{axiom_id}' is self-inconsistent "
                "(its condition and consequence cannot hold together)."
            )
        if conflicts:
            ids = ", ".join(c.existing_axiom_id for c in conflicts)
            messages.append(
                f"New axiom '{axiom_id}' conflicts with existing axioms: {ids}."
            )
        message = " ".join(messages) if messages else f"New axiom '{axiom_id}' invalid."
        super().__init__(message)
        self.axiom_id: str = axiom_id
        self.conflicts: List[AxiomConflict] = conflicts
        self.self_inconsistent: bool = self_inconsistent


DecisionStatus = Literal["SAT", "UNSAT", "UNKNOWN", "ERROR"]
DecisionOutcome = Literal["FLAGGED", "CLEAN", "UNKNOWN", "ERROR"]


# -----------------------------------------------------------------------------
# Pomocnicze: konwersje typów
# -----------------------------------------------------------------------------

def _z3_value_to_python(value: Any) -> Any:
    """Bezpieczna konwersja wartości z3 do typu Python."""
    if isinstance(value, BoolRef):
        return bool(is_true(value))
    if isinstance(value, IntNumRef):
        return int(value.as_long())
    if isinstance(value, RatNumRef):
        return float(value.as_fraction())
    # Fallback – np. symbole, nieoczekiwane typy
    return str(value)


# -----------------------------------------------------------------------------
# Logger decyzji (audit trail)
# -----------------------------------------------------------------------------

class DecisionLogger:
    """
    Prosty logger decyzji do pliku JSONL (jeden rekord na linię).

    Format rekordu:
        {
          "decision_id": "<uuid>",
          "logged_at_utc": "<timestamp>",
          "decision": { ... proof bundle ... }
        }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, decision_bundle: Dict[str, Any]) -> str:
        """Zapisz decyzję i zwróć jej ID."""
        decision_id = decision_bundle.get("decision_id") or str(uuid.uuid4())
        decision_bundle["decision_id"] = decision_id

        record = {
            "decision_id": decision_id,
            "logged_at_utc": datetime.utcnow().isoformat(),
            "decision": decision_bundle,
        }

        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return decision_id


# -----------------------------------------------------------------------------
# Jądro aksjomatyczne
# -----------------------------------------------------------------------------

class AxiomKernel:
    """
    Jądro decyzyjne oparte na aksjomatach.

    - Przyjmuje dynamiczny schemat zmiennych (VariableSchema)
    - Przyjmuje zestaw aksjomatów (AxiomDefinition)
    - Dla podanego case (dict) generuje:
        - decyzję,
        - model wartości zmiennych,
        - informacje o spełnionych/niespełnionych regułach,
        - rozróżnienie reguł aktywnych / nieaktywnych,
        - w razie konfliktów: UNSAT core z listą kolidujących reguł,
        - opcjonalny zapis do logów (audit trail).
    """

    def __init__(
        self,
        schema: Iterable[VariableSchema],
        decision_variable: str = "flag",
        logger: Optional[DecisionLogger] = None,
        rule_version: str = "v1.0.0",
    ) -> None:
        self._schema: Dict[str, VariableSchema] = {v.name: v for v in schema}
        self._decision_variable = decision_variable
        self._logger = logger
        self._rule_version = rule_version

        self._axioms: List[AxiomDefinition] = []
        self._variables: Dict[str, ExprRef] = self._build_variables()

        self._logger_internal = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------ #
    # Konfiguracja
    # ------------------------------------------------------------------ #

    def add_axiom(self, axiom: AxiomDefinition) -> None:
        """
        Dodaj nowy aksjomat do kernela BEZ walidacji konfliktów.

        Używaj świadomie – do zastosowań produkcyjnych preferuj
        add_axiom_safe().
        """
        if any(a.id == axiom.id for a in self._axioms):
            raise ValueError(f"Axiom with id '{axiom.id}' already exists.")
        self._axioms.append(axiom)

    def add_axiom_safe(
        self,
        axiom: AxiomDefinition,
        raise_on_conflict: bool = True,
    ) -> List[AxiomConflict]:
        """
        Dodaj nowy aksjomat z walidacją konfliktów w Z3.

        Sprawdza:
        - unikalność id,
        - spójność samej reguły (czy istnieje model z TRUE antecedent
          i TRUE konsekwencją),
        - konflikty z istniejącymi regułami:
            * czy ich warunki mogą się nałożyć,
            * czy w regionie nakładania konsekwencje są logicznie sprzeczne.

        Zwraca:
            List[AxiomConflict] – lista konfliktów z istniejącymi regułami.
            Może być pusta, jeśli konfliktów nie ma.

        Parametry:
            raise_on_conflict:
                - True  -> w przypadku self-inconsistency lub konfliktów
                           podnosi AxiomConflictError i NIE dodaje reguły.
                - False -> dodaje regułę mimo konfliktów, ale zwraca ich listę.

        To jest preferowana metoda dodawania reguł w PoC/produkcji.
        """
        if any(a.id == axiom.id for a in self._axioms):
            raise ValueError(f"Axiom with id '{axiom.id}' already exists.")

        # Zbuduj constraint nowej reguły i rozdziel na (cond, cons).
        new_expr = axiom.build_constraint(self._variables)
        new_cond, new_cons = self._split_implication(new_expr)

        # 1. Spójność własna: czy istnieje model z cond ∧ cons?
        self_solver = Solver()
        self_solver.add(new_cond, new_cons)
        self_result = self_solver.check()
        self_inconsistent = self_result == unsat

        # 2. Konflikty z istniejącymi regułami
        conflicts: List[AxiomConflict] = []

        for existing in self._axioms:
            existing_expr = existing.build_constraint(self._variables)
            existing_cond, existing_cons = self._split_implication(existing_expr)

            # 2.1. Czy warunki mogą się nałożyć?
            overlap_solver = Solver()
            overlap_solver.add(new_cond, existing_cond)
            overlap_result = overlap_solver.check()

            if overlap_result == unsat:
                # Warunki nigdy nie są jednocześnie prawdziwe – brak konfliktu.
                continue

            overlap_example: Optional[Dict[str, Any]] = None
            if overlap_result == sat:
                overlap_model = overlap_solver.model()
                overlap_example = self._extract_model(overlap_model)

            # 2.2. Czy istnieje model z cond_new ∧ cond_E ∧ cons_new ∧ cons_E?
            conflict_solver = Solver()
            conflict_solver.add(new_cond, existing_cond, new_cons, existing_cons)
            conflict_result = conflict_solver.check()

            if conflict_result == unsat:
                # W regionie overlap nie można spełnić obu konsekwencji jednocześnie.
                conflicts.append(
                    AxiomConflict(
                        existing_axiom_id=existing.id,
                        existing_description=existing.description,
                        overlap_example=overlap_example,
                    )
                )

        if (self_inconsistent or conflicts) and raise_on_conflict:
            raise AxiomConflictError(
                axiom_id=axiom.id,
                conflicts=conflicts,
                self_inconsistent=self_inconsistent,
            )

        # Dodajemy regułę (nawet jeśli są konflikty, gdy raise_on_conflict=False).
        self._axioms.append(axiom)
        return conflicts

    # ------------------------------------------------------------------ #
    # Główna funkcja: ewaluacja case
    # ------------------------------------------------------------------ #

    def evaluate(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ewaluuj pojedynczy case (np. transakcję) przeciwko zestawowi aksjomatów.

        Zwraca "proof bundle" m.in.:

        - decision_status: SAT/UNSAT/UNKNOWN/ERROR
        - decision: FLAGGED/CLEAN/UNKNOWN/ERROR
        - facts: oryginalny case po walidacji typów
        - model: wartości wszystkich zmiennych w modelu Z3
        - satisfied_axioms: lista reguł logicznie spełnionych w modelu
        - violated_axioms: lista reguł logicznie niespełnionych
        - active_axioms: reguły, których antecedent był TRUE w modelu
        - inactive_actions: reguły, których antecedent był FALSE,
          ale cała implikacja była TRUE (vacuous truth)
        - conflicting_axioms: lista id reguł w unsat_core (gdy UNSAT)
        - rule_version: wersja zestawu reguł
        - decision_id / logged_at_utc: uzupełniane po logowaniu (jeśli logger)
        """
        try:
            solver = Solver()
            solver.set(unsat_core=True)

            # 1. Fakty wejściowe
            facts, fact_constraints = self._build_fact_constraints(case)
            solver.add(*fact_constraints)

            # 2. Aksjomaty + assumption literaly (do unsat_core)
            assumption_literals: Dict[str, BoolRef] = {}
            axiom_constraints: Dict[str, ExprRef] = {}

            for axiom in self._axioms:
                literal = Bool(f"assump__{axiom.id}")
                assumption_literals[axiom.id] = literal

                constraint = axiom.build_constraint(self._variables)
                axiom_constraints[axiom.id] = constraint

                # assump__id ⇒ constraint
                solver.add(Implies(literal, constraint))

            assumptions_list = list(assumption_literals.values())

            # 3. SAT / UNSAT / UNKNOWN
            result = solver.check(assumptions_list)

            if result == sat:
                model = solver.model()
                model_dict = self._extract_model(model)
                decision, decision_status = self._derive_decision(model_dict)

                (
                    satisfied,
                    violated,
                    active,
                    inactive,
                ) = self._classify_axioms_in_model(axiom_constraints, model)

                bundle: Dict[str, Any] = {
                    "decision_status": decision_status,
                    "decision": decision,
                    "facts": facts,
                    "model": model_dict,
                    "satisfied_axioms": satisfied,
                    "violated_axioms": violated,
                    "active_axioms": active,
                    "inactive_actions": inactive,
                    "conflicting_axioms": [],
                    "rule_version": self._rule_version,
                }

            elif result == unsat:
                core = solver.unsat_core()
                core_ids = {
                    ax_id
                    for ax_id, lit in assumption_literals.items()
                    if lit in core
                }
                bundle = {
                    "decision_status": "UNSAT",
                    "decision": "ERROR",
                    "facts": facts,
                    "model": {},
                    "satisfied_axioms": [],
                    "violated_axioms": [],
                    "active_axioms": [],
                    "inactive_actions": [],
                    "conflicting_axioms": sorted(core_ids),
                    "rule_version": self._rule_version,
                    "error": "Constraints are unsatisfiable for given case.",
                }

            elif result == unknown:
                bundle = {
                    "decision_status": "UNKNOWN",
                    "decision": "UNKNOWN",
                    "facts": facts,
                    "model": {},
                    "satisfied_axioms": [],
                    "violated_axioms": [],
                    "active_axioms": [],
                    "inactive_actions": [],
                    "conflicting_axioms": [],
                    "rule_version": self._rule_version,
                    "error": "Solver returned UNKNOWN (timeout or unsupported).",
                }

            else:
                bundle = {
                    "decision_status": "ERROR",
                    "decision": "ERROR",
                    "facts": facts,
                    "model": {},
                    "satisfied_axioms": [],
                    "violated_axioms": [],
                    "active_axioms": [],
                    "inactive_actions": [],
                    "conflicting_axioms": [],
                    "rule_version": self._rule_version,
                    "error": f"Unexpected solver result: {result}",
                }

        except Exception as exc:  # szeroka siatka na poziomie kernela
            self._logger_internal.exception("Kernel evaluation error")
            bundle = {
                "decision_status": "ERROR",
                "decision": "ERROR",
                "facts": case,
                "model": {},
                "satisfied_axioms": [],
                "violated_axioms": [],
                "active_axioms": [],
                "inactive_actions": [],
                "conflicting_axioms": [],
                "rule_version": self._rule_version,
                "error": str(exc),
            }

        # 4. Audit trail (opcjonalnie)
        if self._logger is not None:
            decision_id = self._logger.log(bundle)
            bundle["decision_id"] = decision_id
            bundle["logged_at_utc"] = datetime.utcnow().isoformat()

        return bundle

    # ------------------------------------------------------------------ #
    # Prywatne: budowa zmiennych, faktów, modelu
    # ------------------------------------------------------------------ #

    def _build_variables(self) -> Dict[str, ExprRef]:
        """Tworzy zmienne Z3 na podstawie schematu."""
        variables: Dict[str, ExprRef] = {}
        for name, spec in self._schema.items():
            if spec.type == "int":
                variables[name] = Int(name)
            elif spec.type == "bool":
                variables[name] = Bool(name)
            elif spec.type == "real":
                variables[name] = Real(name)
            else:
                raise ValueError(f"Unsupported variable type: {spec.type}")
        return variables

    def _build_fact_constraints(
        self, case: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[ExprRef]]:
        """
        Z walidacją typu konwertuje "case" na:
        - facts: dopasowane wartości Python,
        - constraints: lista równości do dodania do solvera.
        """
        facts: Dict[str, Any] = {}
        constraints: List[ExprRef] = []

        for name, spec in self._schema.items():
            if name not in case:
                # Zmienna może zostać nieskonkretyzowana – solver znajdzie wartość.
                continue

            raw_value = case[name]

            if spec.type == "int":
                try:
                    value = int(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Field '{name}' expected int, got {raw_value!r}"
                    ) from exc
            elif spec.type == "bool":
                if isinstance(raw_value, bool):
                    value = raw_value
                elif isinstance(raw_value, str):
                    lowered = raw_value.lower()
                    if lowered in {"true", "1", "yes"}:
                        value = True
                    elif lowered in {"false", "0", "no"}:
                        value = False
                    else:
                        raise ValueError(
                            f"Field '{name}' expected bool, "
                            f"got string {raw_value!r}"
                        )
                else:
                    raise ValueError(
                        f"Field '{name}' expected bool, got {raw_value!r}"
                    )
            elif spec.type == "real":
                try:
                    value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Field '{name}' expected real, got {raw_value!r}"
                    ) from exc
            else:
                raise ValueError(f"Unsupported variable type: {spec.type}")

            facts[name] = value

            var = self._variables[name]
            if spec.type == "int":
                constraints.append(var == int(value))
            elif spec.type == "bool":
                constraints.append(var == bool(value))
            elif spec.type == "real":
                constraints.append(var == float(value))

        return facts, constraints

    def _extract_model(self, model: Any) -> Dict[str, Any]:
        """Wyciąga wartości zmiennych z modelu Z3 do słownika Python."""
        result: Dict[str, Any] = {}
        for name, var in self._variables.items():
            val = model.eval(var, model_completion=True)
            result[name] = _z3_value_to_python(val)
        return result

    def _derive_decision(
        self, model_dict: Dict[str, Any]
    ) -> Tuple[DecisionOutcome, DecisionStatus]:
        """
        Wyprowadza decyzję biznesową na podstawie wartości
        kluczowej zmiennej decyzyjnej, np. "flag".
        """
        if self._decision_variable not in model_dict:
            return "UNKNOWN", "SAT"

        value = model_dict[self._decision_variable]

        if isinstance(value, bool):
            return ("FLAGGED" if value else "CLEAN", "SAT")

        # Jeśli zmienna decyzyjna istnieje, ale nie jest bool.
        return "UNKNOWN", "SAT"

    # ------------------------------------------------------------------ #
    # Prywatne: analiza reguł w modelu
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_implication(expr: ExprRef) -> Tuple[ExprRef, ExprRef]:
        """
        Dzieli wyrażenie na (antecedent, consequent).

        Jeśli expr jest implikacją (=>), zwraca:
            (expr.arg(0), expr.arg(1))
        W przeciwnym razie:
            (True, expr)
        """
        try:
            if expr.decl().name() == "=>":
                return expr.arg(0), expr.arg(1)
        except Exception:
            # Na wszelki wypadek – jeśli expr nie ma decl()/arg()
            pass
        return BoolVal(True), expr

    def _classify_axioms_in_model(
        self, constraints: Dict[str, ExprRef], model: Any
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Sprawdza dla każdej reguły:

        - czy constraint (cała formuła) jest spełniony w modelu (holds),
        - czy antecedent (warunek) jest TRUE / FALSE w modelu.

        Heurystyka dla antecedentu:
        - jeśli constraint jest Implies(A, B), to antecedent = A,
        - w przeciwnym razie antecedent = cały constraint.

        Zwraca:
            (satisfied_axioms, violated_axioms, active_axioms, inactive_actions)
        """
        satisfied: List[Dict[str, Any]] = []
        violated: List[Dict[str, Any]] = []
        active: List[Dict[str, Any]] = []
        inactive: List[Dict[str, Any]] = []

        for axiom in self._axioms:
            expr = constraints[axiom.id]

            antecedent_expr, _ = self._split_implication(expr)

            expr_val = model.eval(expr, model_completion=True)
            antecedent_val = model.eval(antecedent_expr, model_completion=True)

            holds = bool(is_true(expr_val))
            antecedent_true = bool(is_true(antecedent_val))

            entry = {
                "id": axiom.id,
                "description": axiom.description,
                "holds": holds,
                "antecedent_true": antecedent_true,
            }

            target = satisfied if holds else violated
            target.append(entry)

            # Aktywna: antecedent TRUE (reguła "odpaliła").
            # Nieaktywna: antecedent FALSE, ale cała implikacja TRUE (vacuous truth).
            if antecedent_true:
                active.append(entry)
            elif holds:
                inactive.append(entry)

        return satisfied, violated, active, inactive


# -----------------------------------------------------------------------------
# DEMO / PoC
# -----------------------------------------------------------------------------

def _build_demo_kernel() -> AxiomKernel:
    """
    Tworzy przykładowy kernel:
    - amount: kwota transakcji (int)
    - risk_score: wynik ryzyka (int)
    - flag: decyzja o naznaczeniu transakcji (bool)
    """
    schema = [
        VariableSchema(
            name="amount",
            type="int",
            description="Transaction amount (in minor units, e.g. cents).",
        ),
        VariableSchema(
            name="risk_score",
            type="int",
            description="Precomputed risk score from ML model (0-10).",
        ),
        VariableSchema(
            name="flag",
            type="bool",
            description="Decision whether to flag the transaction.",
        ),
    ]

    kernel = AxiomKernel(
        schema=schema,
        decision_variable="flag",
        logger=DecisionLogger("logs/axiom_kernel_decisions.jsonl"),
        rule_version="aml_rules_v1.0.0",
    )

    # Reguła wysokiego ryzyka
    def rule_high_risk(vars_z3: Dict[str, ExprRef]) -> ExprRef:
        amount = vars_z3["amount"]
        risk_score = vars_z3["risk_score"]
        flag = vars_z3["flag"]
        return Implies(And(amount > 10_000, risk_score > 5), flag)

    # Reguła bardzo niskiego ryzyka
    def rule_low_risk(vars_z3: Dict[str, ExprRef]) -> ExprRef:
        risk_score = vars_z3["risk_score"]
        flag = vars_z3["flag"]
        return Implies(risk_score <= 2, flag == False)

    kernel.add_axiom(
        AxiomDefinition(
            id="high_risk_flag",
            description=(
                "If amount > 10000 and risk_score > 5 then "
                "transaction must be flagged."
            ),
            build_constraint=rule_high_risk,
            priority=10,
        )
    )

    kernel.add_axiom(
        AxiomDefinition(
            id="low_risk_clear",
            description="If risk_score <= 2 then transaction must be clean.",
            build_constraint=rule_low_risk,
            priority=5,
        )
    )

    return kernel


def _demo() -> None:
    """Prosty demo-case do uruchomienia z linii komend."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    kernel = _build_demo_kernel()

    case = {
        "amount": 15_000,
        "risk_score": 7,
    }

    proof_bundle = kernel.evaluate(case)

    print("\n=== Proof-Carrying Decision ===")
    print(json.dumps(proof_bundle, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()
