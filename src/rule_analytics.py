"""rule_analytics.py

FAZA 4 – Silnik analizy reguł i decyzji na podstawie logów JSONL
generowanych przez DecisionLogger z axiomatic_kernel.py.

Główne założenia:
- Źródłem prawdy są logi decyzji (JSONL), gdzie każda linia ma postać:
    {
      "decision_id": "<uuid>",
      "logged_at_utc": "<ISO timestamp>",
      "decision": {
         "decision_status": "...",
         "decision": "...",
         "facts": {...},
         "model": {...},
         "satisfied_axioms": [...],
         "violated_axioms": [...],
         "active_axioms": [...],
         "inactive_actions": [...],
         "conflicting_axioms": [...],
         "rule_version": "..."
         ...
      }
    }

- Moduł nie zależy od Z3 ani innych ciężkich komponentów – operuje
  wyłącznie na danych z logów.

- Wynikiem analizy jest struktura danych gotowa do dalszego
  raportowania / wizualizacji w PoC bankowym:
  * statystyki decyzji,
  * statystyki reguł,
  * raport pokrycia rulesetu (jeśli podamy RuleSet z rules_io).

Możesz ten moduł wpiąć bezpośrednio do istniejącego projektu.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rules_io import RuleSet, load_ruleset_from_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modele danych
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionRecord:
    """Pojedynczy rekord decyzji odczytany z logu JSONL.

    Attributes:
        decision_id:
            Identyfikator decyzji nadany przez DecisionLogger.
        logged_at_utc:
            Moment zapisania decyzji w formacie datetime (UTC).
        bundle:
            Pełny "proof bundle" zwrócony przez AxiomKernel.evaluate().
    """

    decision_id: str
    logged_at_utc: datetime
    bundle: Dict[str, Any]

    @staticmethod
    def from_json_line(line: str) -> "DecisionRecord":
        """Parsuje jedną linię JSONL i zwraca DecisionRecord.

        Podnosi ValueError przy błędnym formacie.
        """
        raw = json.loads(line)
        decision_id = str(raw.get("decision_id", ""))

        logged_at_raw = raw.get("logged_at_utc")
        if not isinstance(logged_at_raw, str):
            raise ValueError("logged_at_utc must be a string timestamp")

        try:
            logged_at = datetime.fromisoformat(logged_at_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO timestamp in logged_at_utc: {logged_at_raw!r}"
            ) from exc

        bundle = raw.get("decision")
        if not isinstance(bundle, dict):
            raise ValueError("'decision' field must be an object")

        return DecisionRecord(
            decision_id=decision_id or "",
            logged_at_utc=logged_at,
            bundle=bundle,
        )


@dataclass
class DecisionOutcomeStats:
    """Zagregowane statystyki decyzji w logu."""

    total_decisions: int = 0
    by_decision: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    by_rule_version: Dict[str, int] = field(default_factory=dict)

    # liczba przypadków, w których solver zwrócił UNSAT (konflikt reguł)
    unsat_cases: int = 0

    # liczba przypadków, w których status był ERROR lub UNKNOWN
    error_cases: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "by_decision": dict(self.by_decision),
            "by_status": dict(self.by_status),
            "by_rule_version": dict(self.by_rule_version),
            "unsat_cases": self.unsat_cases,
            "error_cases": self.error_cases,
        }


@dataclass
class RuleStats:
    """Statystyki pojedynczej reguły (na przestrzeni wielu decyzji)."""

    rule_id: str
    description: Optional[str] = None

    # liczba decyzji, w których reguła w ogóle się pojawiła
    total_occurrences: int = 0

    # liczba decyzji, w których reguła była logicznie spełniona
    satisfied: int = 0

    # liczba decyzji, w których reguła była logicznie niespełniona
    violated: int = 0

    # liczba decyzji, w których antecedent był TRUE
    active: int = 0

    # liczba decyzji, w których reguła była true "vacuously" (antecedent FALSE)
    inactive: int = 0

    # liczba decyzji, w których reguła wystąpiła w conflicting_axioms
    in_conflict: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "total_occurrences": self.total_occurrences,
            "satisfied": self.satisfied,
            "violated": self.violated,
            "active": self.active,
            "inactive": self.inactive,
            "in_conflict": self.in_conflict,
        }


@dataclass
class RuleCoverageReport:
    """Raport pokrycia rulesetu na podstawie logów.

    Attributes:
        ruleset_id:
            Id rulesetu (z pliku).
        version:
            Wersja rulesetu.
        total_enabled_rules:
            Liczba reguł enabled=True w ruleset.
        used_rules:
            Lista identyfikatorów reguł, które pojawiły się
            w statystykach (czyli wystąpiły w co najmniej jednej decyzji).
        unused_rules:
            Lista identyfikatorów reguł enabled=True, które nie
            pojawiły się w logach (martwe / nieużywane).
    """

    ruleset_id: str
    version: str
    total_enabled_rules: int
    used_rules: List[str] = field(default_factory=list)
    unused_rules: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ruleset_id": self.ruleset_id,
            "version": self.version,
            "total_enabled_rules": self.total_enabled_rules,
            "used_rules": list(self.used_rules),
            "unused_rules": list(self.unused_rules),
        }


@dataclass
class RuleAnalyticsResult:
    """Kompletny wynik analizy reguł i decyzji."""

    outcome_stats: DecisionOutcomeStats
    rule_stats: Dict[str, RuleStats] = field(default_factory=dict)
    coverage_report: Optional[RuleCoverageReport] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outcome_stats": self.outcome_stats.as_dict(),
            "rule_stats": {
                rule_id: stats.as_dict()
                for rule_id, stats in sorted(self.rule_stats.items())
            },
            "coverage_report": (
                None
                if self.coverage_report is None
                else self.coverage_report.as_dict()
            ),
        }


# ---------------------------------------------------------------------------
# Czytnik logów JSONL
# ---------------------------------------------------------------------------


class DecisionLogReader:
    """Prosty reader logów JSONL z DecisionLogger.

    Przechodzi linię po linii, zwraca DecisionRecord. Błędy parsowania
    loguje, ale nie przerywa całej analizy (odrzuca wadliwą linię).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def iter_decisions(self) -> Iterable[DecisionRecord]:
        if not self._path.exists():
            logger.warning(
                "Decision log file %s does not exist – no data to analyze.",
                self._path,
            )
            return

        with self._path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield DecisionRecord.from_json_line(stripped)
                except Exception:  # pragma: no cover - defensywne logowanie
                    logger.exception(
                        "Failed to parse decision log line %d in %s",
                        line_number,
                        self._path,
                    )


# ---------------------------------------------------------------------------
# Silnik analityczny
# ---------------------------------------------------------------------------


class RuleAnalyticsEngine:
    """Główny silnik analizy logów regułowych.

    Typowe użycie:

        engine = RuleAnalyticsEngine()
        result = engine.analyze_log_file(
            log_path="decision_log.jsonl",
            ruleset_path="rules_aml_v1.yaml",
        )
        report = result.as_dict()
    """

    def analyze_log_file(
        self,
        *,
        log_path: str | Path,
        ruleset: Optional[RuleSet] = None,
        ruleset_path: Optional[str | Path] = None,
    ) -> RuleAnalyticsResult:
        """Analizuje podany plik logów JSONL.

        Możesz przekazać:
        - gotowy RuleSet (ruleset),
        - albo ścieżkę do pliku rulesetu (ruleset_path),
        - albo nic (analiza tylko decyzji i reguł obecnych w logach).

        Jeśli podano zarówno ruleset, jak i ruleset_path, priorytet
        ma obiekt ruleset.
        """

        if ruleset is None and ruleset_path is not None:
            ruleset = load_ruleset_from_file(Path(ruleset_path))

        reader = DecisionLogReader(log_path)

        outcome_stats = DecisionOutcomeStats()
        rule_stats: Dict[str, RuleStats] = {}

        for record in reader.iter_decisions():
            bundle = record.bundle

            decision = str(bundle.get("decision", "UNKNOWN"))
            status = str(bundle.get("decision_status", "UNKNOWN"))
            rule_version = str(bundle.get("rule_version", "unknown"))

            outcome_stats.total_decisions += 1
            outcome_stats.by_decision[decision] = (
                outcome_stats.by_decision.get(decision, 0) + 1
            )
            outcome_stats.by_status[status] = (
                outcome_stats.by_status.get(status, 0) + 1
            )
            outcome_stats.by_rule_version[rule_version] = (
                outcome_stats.by_rule_version.get(rule_version, 0) + 1
            )

            if status == "UNSAT":
                outcome_stats.unsat_cases += 1
            if status in {"ERROR", "UNKNOWN"}:
                outcome_stats.error_cases += 1

            # Zbierz reguły występujące w tej decyzji, aby móc policzyć
            # total_occurrences (każda reguła max raz na decyzję).
            rules_in_decision: set[str] = set()

            def _ensure_rule_stats(
                rule_id: str,
                description: Optional[str],
            ) -> RuleStats:
                if rule_id not in rule_stats:
                    rule_stats[rule_id] = RuleStats(
                        rule_id=rule_id,
                        description=description,
                    )
                else:
                    # Jeśli wcześniej description było None, a teraz mamy
                    # jakikolwiek opis, uzupełnijmy go.
                    if description and not rule_stats[rule_id].description:
                        rule_stats[rule_id].description = description
                return rule_stats[rule_id]

            # satisfied_axioms: lista dictów z polami id, description, ...
            for entry in bundle.get("satisfied_axioms", []):
                rule_id = str(entry.get("id", ""))
                if not rule_id:
                    continue
                description = entry.get("description")
                stats = _ensure_rule_stats(rule_id, description)
                stats.satisfied += 1
                rules_in_decision.add(rule_id)

            # violated_axioms
            for entry in bundle.get("violated_axioms", []):
                rule_id = str(entry.get("id", ""))
                if not rule_id:
                    continue
                description = entry.get("description")
                stats = _ensure_rule_stats(rule_id, description)
                stats.violated += 1
                rules_in_decision.add(rule_id)

            # active_axioms
            for entry in bundle.get("active_axioms", []):
                rule_id = str(entry.get("id", ""))
                if not rule_id:
                    continue
                description = entry.get("description")
                stats = _ensure_rule_stats(rule_id, description)
                stats.active += 1
                rules_in_decision.add(rule_id)

            # inactive_actions
            for entry in bundle.get("inactive_actions", []):
                rule_id = str(entry.get("id", ""))
                if not rule_id:
                    continue
                description = entry.get("description")
                stats = _ensure_rule_stats(rule_id, description)
                stats.inactive += 1
                rules_in_decision.add(rule_id)

            # conflicting_axioms: lista id (stringów)
            for rule_id in bundle.get("conflicting_axioms", []):
                rule_id_str = str(rule_id)
                if not rule_id_str:
                    continue
                stats = _ensure_rule_stats(rule_id_str, None)
                stats.in_conflict += 1
                rules_in_decision.add(rule_id_str)

            # Na koniec zwiększamy total_occurrences dla każdej reguły,
            # która pojawiła się w tej decyzji w jakiejkolwiek roli.
            for rule_id in rules_in_decision:
                rule_stats[rule_id].total_occurrences += 1

        coverage_report: Optional[RuleCoverageReport] = None

        if ruleset is not None:
            # Identyfikatory reguł, które występują w statystykach
            used_rule_ids = {rule_id for rule_id in rule_stats}
            enabled_rules = [rule for rule in ruleset.rules if rule.enabled]
            enabled_rule_ids = {rule.rule_id for rule in enabled_rules}
            unused_rule_ids = sorted(enabled_rule_ids - used_rule_ids)
            used_rule_ids_sorted = sorted(enabled_rule_ids & used_rule_ids)

            coverage_report = RuleCoverageReport(
                ruleset_id=ruleset.ruleset_id,
                version=ruleset.version,
                total_enabled_rules=len(enabled_rules),
                used_rules=used_rule_ids_sorted,
                unused_rules=unused_rule_ids,
            )

        return RuleAnalyticsResult(
            outcome_stats=outcome_stats,
            rule_stats=rule_stats,
            coverage_report=coverage_report,
        )
