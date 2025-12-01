"""
explanation_engine.py

Warstwa wyjaśniająca (human-readable explanations) dla decyzji
z AxiomKernel. Przyjmuje proof bundle z evaluate() i generuje
zwięzły opis: co się stało, dlaczego, które reguły zadziałały,
które były nieaktywne, jakie są konflikty / błędy.

Zaprojektowane jako cienka warstwa nad:
    - axiomatic_kernel.AxiomKernel
    - nl_rule_parser.build_axiom_from_nl

Nie używa LLM – opiera się wyłącznie na danych z bundla.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from axiomatic_kernel import DecisionStatus, DecisionOutcome  # type: ignore[import]


LanguageCode = Literal["pl", "en"]


@dataclass
class ExplanationConfig:
    """
    Konfiguracja generatora wyjaśnień.

    Attributes:
        language:
            'pl' lub 'en' – język generowanego tekstu.
        include_inactive:
            Czy dołączać listę reguł, które były true tylko
            w sensie "vacuous truth" (antecedent=False, implikacja=True).
        include_conflicts:
            Czy dołączać informację o conflicting_axioms (UNSAT).
        max_active_rules:
            Maksymalna liczba aktywnych reguł w sekcji "powody".
        max_inactive_rules:
            Maksymalna liczba nieaktywnych reguł w sekcji "reguły, które nie zadziałały".
    """

    language: LanguageCode = "pl"
    include_inactive: bool = True
    include_conflicts: bool = True
    max_active_rules: int = 5
    max_inactive_rules: int = 5


@dataclass
class DecisionExplanation:
    """
    Struktura na wyjaśnienie jednej decyzji.

    Umożliwia zarówno programistyczny dostęp do poszczególnych sekcji,
    jak i generowanie kompletnego tekstu (metoda to_text()).
    """

    decision: DecisionOutcome
    status: DecisionStatus
    headline: str
    reasons: List[str] = field(default_factory=list)
    inactive_reasons: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_text(self, language: LanguageCode = "pl") -> str:
        """
        Zwraca pełne wyjaśnienie jako tekst w danym języku.

        Parametry:
            language: 'pl' lub 'en'. Jeśli różni się od języka użytego
                      przy budowaniu headline/reasons, tekst nadal będzie
                      sensowny, ale nagłówki sekcji będą w wybranym języku.
        """
        lines: List[str] = [self.headline]

        if self.reasons:
            lines.append("")
            if language == "pl":
                lines.append("Powody (aktywne reguły):")
            else:
                lines.append("Reasons (active rules):")
            for r in self.reasons:
                lines.append(f"- {r}")

        if self.inactive_reasons:
            lines.append("")
            if language == "pl":
                lines.append("Reguły, które nie zadziałały w tym przypadku:")
            else:
                lines.append("Rules that did not fire in this case:")
            for r in self.inactive_reasons:
                lines.append(f"- {r}")

        if self.conflicts:
            lines.append("")
            if language == "pl":
                lines.append("Konflikty reguł:")
            else:
                lines.append("Rule conflicts:")
            for c in self.conflicts:
                lines.append(f"- {c}")

        if self.error:
            lines.append("")
            if language == "pl":
                lines.append(f"Błąd techniczny: {self.error}")
            else:
                lines.append(f"Technical error: {self.error}")

        return "\n".join(lines)


class DecisionExplainer:
    """
    Generator wyjaśnień dla proof bundles z AxiomKernel.

    Typowy użytek:

        kernel = AxiomKernel(...)
        bundle = kernel.evaluate(case)
        explainer = DecisionExplainer(ExplanationConfig(language="pl"))
        explanation = explainer.explain(bundle)
        print(explanation.to_text())
    """

    def __init__(self, config: Optional[ExplanationConfig] = None) -> None:
        self._config: ExplanationConfig = config or ExplanationConfig()

    # ------------------------------------------------------------------ #
    # API główne
    # ------------------------------------------------------------------ #

    def explain(self, bundle: Dict[str, Any]) -> DecisionExplanation:
        """
        Buduje strukturalne wyjaśnienie dla bundla z evaluate().

        Parametry:
            bundle:
                Słownik zwrócony przez AxiomKernel.evaluate(...).

        Zwraca:
            DecisionExplanation – gotowe do użycia wyjaśnienie.
        """
        status = bundle.get("decision_status", "UNKNOWN")  # type: ignore[assignment]
        decision = bundle.get("decision", "UNKNOWN")  # type: ignore[assignment]
        error = bundle.get("error")
        facts = bundle.get("facts") or {}
        active_axioms = bundle.get("active_axioms") or []
        inactive_actions = bundle.get("inactive_actions") or []
        conflicting_axioms = bundle.get("conflicting_axioms") or []

        headline = self._build_headline(
            status=status,
            decision=decision,
            facts=facts,
            error=error,
        )

        reasons = self._build_active_reasons(active_axioms, facts)
        inactive_reasons: List[str] = []
        if self._config.include_inactive:
            inactive_reasons = self._build_inactive_reasons(inactive_actions)

        conflicts: List[str] = []
        if self._config.include_conflicts and conflicting_axioms:
            conflicts = self._build_conflicts(conflicting_axioms)

        return DecisionExplanation(
            decision=decision,
            status=status,
            headline=headline,
            reasons=reasons,
            inactive_reasons=inactive_reasons,
            conflicts=conflicts,
            error=error,
        )

    # ------------------------------------------------------------------ #
    # Prywatne: budowa poszczególnych sekcji wyjaśnienia
    # ------------------------------------------------------------------ #

    def _build_headline(
        self,
        status: DecisionStatus,
        decision: DecisionOutcome,
        facts: Dict[str, Any],
        error: Optional[str],
    ) -> str:
        """Buduje nagłówek (pierwsze zdanie) dla danej decyzji."""
        lang = self._config.language
        fact_summary = self._summarize_facts(facts)

        if lang == "pl":
            if status == "SAT":
                if decision == "FLAGGED":
                    base = "Decyzja: transakcja została OFLAGOWANA (FLAGGED)."
                elif decision == "CLEAN":
                    base = "Decyzja: transakcja została OZNACZONA jako CZYSTA (CLEAN)."
                else:
                    base = f"Decyzja: {decision} (status SAT)."
            elif status == "UNSAT":
                base = (
                    "Decyzja niemożliwa: zestaw reguł jest SPRZECZNY "
                    "dla tego przypadku (UNSAT)."
                )
            elif status == "ERROR":
                base = "Wystąpił błąd podczas ewaluacji reguł."
            else:
                base = f"Status decyzyjny NIEJEDNOZNACZNY (status={status})."

            if fact_summary:
                base += f" Kluczowe dane wejściowe: {fact_summary}."
            return base

        # EN
        if status == "SAT":
            if decision == "FLAGGED":
                base = "Decision: transaction has been FLAGGED."
            elif decision == "CLEAN":
                base = "Decision: transaction has been marked as CLEAN."
            else:
                base = f"Decision: {decision} (status SAT)."
        elif status == "UNSAT":
            base = (
                "Decision impossible: rule set is CONTRADICTORY "
                "for this case (UNSAT)."
            )
        elif status == "ERROR":
            base = "An error occurred during rule evaluation."
        else:
            base = f"Decision status is UNKNOWN (status={status})."

        if fact_summary:
            base += f" Key input facts: {fact_summary}."
        return base

    def _summarize_facts(self, facts: Dict[str, Any]) -> str:
        """
        Prosta, deterministyczna synteza najważniejszych faktów.

        W tej wersji:
        - bierzemy maksymalnie 5 pól,
        - sortujemy po nazwie pola, żeby kolejność była stabilna.
        """
        if not facts:
            return ""

        items = sorted(facts.items(), key=lambda kv: kv[0])
        limited = items[:5]
        return ", ".join(f"{name}={value!r}" for name, value in limited)

    def _build_active_reasons(
        self,
        active_axioms: List[Dict[str, Any]],
        facts: Dict[str, Any],
    ) -> List[str]:
        """
        Buduje listę tekstowych opisów dla aktywnych reguł (antecedent_true=True).
        """
        lang = self._config.language
        max_rules = self._config.max_active_rules
        reasons: List[str] = []

        for entry in active_axioms[:max_rules]:
            ax_id = entry.get("id", "<unknown>")
            desc = entry.get("description", "")
            if lang == "pl":
                text = f"Reguła '{ax_id}': {desc}"
            else:
                text = f"Rule '{ax_id}': {desc}"
            reasons.append(text)

        if len(active_axioms) > max_rules:
            remaining = len(active_axioms) - max_rules
            if lang == "pl":
                reasons.append(f"... oraz {remaining} dalszych aktywnych reguł.")
            else:
                reasons.append(f"... and {remaining} more active rules.")

        return reasons

    def _build_inactive_reasons(
        self,
        inactive_actions: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Buduje listę opisów dla reguł, które były true tylko 'z próżni' (vacuous).

        To są reguły typu: Implies(A, B), gdzie A=False w tym case,
        więc cała implikacja jest True, ale reguła nie zadziałała "aktywnie".
        """
        lang = self._config.language
        max_rules = self._config.max_inactive_rules
        reasons: List[str] = []

        for entry in inactive_actions[:max_rules]:
            ax_id = entry.get("id", "<unknown>")
            desc = entry.get("description", "")
            if lang == "pl":
                text = (
                    f"Reguła '{ax_id}' była spełniona logicznie, ale jej warunek "
                    f"nie dotyczył tego przypadku: {desc}"
                )
            else:
                text = (
                    f"Rule '{ax_id}' was logically satisfied, but its antecedent "
                    f"did not apply to this case: {desc}"
                )
            reasons.append(text)

        if len(inactive_actions) > max_rules:
            remaining = len(inactive_actions) - max_rules
            if lang == "pl":
                reasons.append(
                    f"... oraz {remaining} dodatkowych reguł, które nie miały "
                    "zastosowania w tym przypadku."
                )
            else:
                reasons.append(
                    f"... and {remaining} more rules that did not apply "
                    "in this case."
                )

        return reasons

    def _build_conflicts(self, conflicting_axioms: List[str]) -> List[str]:
        """
        Buduje listę opisów konfliktów na podstawie conflicting_axioms (listy ID).
        """
        lang = self._config.language
        if not conflicting_axioms:
            return []

        if lang == "pl":
            return [
                f"Konflikt między regułami: {', '.join(conflicting_axioms)}."
            ]
        return [f"Conflict between rules: {', '.join(conflicting_axioms)}."]
