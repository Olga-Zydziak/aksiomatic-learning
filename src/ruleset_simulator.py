"""
ruleset_simulator.py

Narzędzie pomocnicze do symulacji działania pojedynczego rulesetu
na pliku CSV z transakcjami (lub innymi case'ami) przy użyciu
AxiomKernel + RulesetRegistry + RuleAnalyticsEngine.

Zaprojektowane jako cienka, "bank-ready" warstwa:
- nie miesza logiki reguł (ta jest w YAML + kernel),
- wspiera audit trail (log JSONL),
- zwraca spójne statystyki użycia reguł.

Typowe użycie w notebooku:

    simulator = RulesetSimulator(schema=schema)

    config = SimulationConfig(
        ruleset_id="fraud_rules_v1",
        ruleset_path=fraud_rules_path,
        environment=Environment.DEV,
        input_csv_path=data_dir / "transactions_demo.csv",
        output_csv_path=data_dir / "transactions_simulation_fraud_v1.csv",
        log_path=logs_dir / "fraud_rules_v1_simulation.jsonl",
        language="pl",
        decision_variable="is_suspicious",
    )

    result = simulator.run_simulation(
        config=config,
        case_builder=build_fraud_case_from_row,
    )

    report = result.analytics.as_dict()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

import csv

from axiomatic_kernel import AxiomKernel, DecisionLogger, VariableSchema
from explanation_engine import DecisionExplainer, ExplanationConfig
from rule_analytics import RuleAnalyticsEngine, RuleAnalyticsResult
from ruleset_manager import Environment, RulesetRegistry
from decision_constants import OUTCOME_UNKNOWN, STATUS_UNKNOWN


CaseBuilder = Callable[[Mapping[str, str]], Dict[str, Any]]


@dataclass(frozen=True)
class SimulationConfig:
    """
    Konfiguracja pojedynczej symulacji rulesetu na pliku CSV.

    Attributes:
        ruleset_id: Identyfikator rulesetu (zgodny z YAML i registry).
        ruleset_path: Ścieżka do pliku YAML z regułami.
        environment: Środowisko (DEV / TEST / PROD) używane przez RulesetRegistry.
        input_csv_path: Plik CSV z danymi wejściowymi (np. transakcje).
        output_csv_path: Plik CSV z decyzjami, aktywnymi regułami i wyjaśnieniami.
        log_path: Plik JSONL z logiem decyzji (DecisionLogger).
        language: Język wyjaśnień (aktualnie "pl" lub "en").
        decision_variable: Nazwa zmiennej decyzyjnej w schemacie kernela.
    """

    ruleset_id: str
    ruleset_path: Path
    environment: Environment
    input_csv_path: Path
    output_csv_path: Path
    log_path: Path
    language: str = "pl"
    decision_variable: str = "is_suspicious"


@dataclass
class SimulationResult:
    """
    Wynik symulacji rulesetu na danych CSV.
    """

    config: SimulationConfig
    total_cases: int
    analytics: RuleAnalyticsResult


class RulesetSimulator:
    """
    Silnik wykonujący symulację działania rulesetu na pliku CSV.

    Jest niezależny od domeny (AML/fraud/etc). Oczekuje:
    - schematu zmiennych (schema),
    - konfiguracji symulacji (SimulationConfig),
    - funkcji budującej case z pojedynczego wiersza CSV (CaseBuilder).
    """

    def __init__(self, schema: Iterable[VariableSchema]) -> None:
        self._schema = list(schema)
        self._analytics_engine = RuleAnalyticsEngine()

    def run_simulation(
        self,
        *,
        config: SimulationConfig,
        case_builder: CaseBuilder,
    ) -> SimulationResult:
        """
        Uruchamia symulację rulesetu na pliku CSV.

        Kroki:
        - tworzy DecisionLogger i AxiomKernel,
        - rejestruje i nakłada ruleset przez RulesetRegistry,
        - dla każdego wiersza CSV:
            * buduje case (case_builder),
            * woła kernel.evaluate(case),
            * zapisuje wynik do loga JSONL (robi to kernel),
            * zapisuje rozszerzony wiersz do output_csv (decyzja + wyjaśnienie),
        - uruchamia RuleAnalyticsEngine na wygenerowanym logu,
        - zwraca SimulationResult z podsumowaniem.
        """
        if not config.input_csv_path.exists():
            raise FileNotFoundError(
                f"Wejściowy plik CSV nie istnieje: {config.input_csv_path}"
            )

        logger = DecisionLogger(config.log_path)

        kernel = AxiomKernel(
            schema=self._schema,
            decision_variable=config.decision_variable,
            logger=logger,
            rule_version=config.ruleset_id,
        )

        registry = RulesetRegistry()
        registry.register_ruleset(
            ruleset_id=config.ruleset_id,
            path=config.ruleset_path,
            environment=config.environment,
        )

        summary = registry.apply_ruleset_to_kernel(
            ruleset_id=config.ruleset_id,
            environment=config.environment,
            kernel=kernel,
            schema=self._schema,
            decision_field_fallback=config.decision_variable,
        )

        # Jeśli nic nie załadowano, to znaczy, że ruleset jest pusty
        # albo wszystkie reguły są wyłączone.
        if summary.loaded_rules == 0:
            raise ValueError(
                f"Ruleset '{config.ruleset_id}' nie załadował żadnych reguł "
                f"(enabled_rules={summary.enabled_rules}, "
                f"total_rules={summary.total_rules})."
            )

        explainer = DecisionExplainer(ExplanationConfig(language=config.language))

        total_cases = 0

        # Wczytujemy dane wejściowe i równolegle budujemy output CSV.
        config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with config.input_csv_path.open("r", newline="", encoding="utf-8") as f_in, \
                config.output_csv_path.open("w", newline="", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)
            input_fieldnames = list(reader.fieldnames or [])
            extra_fields = [
                "decision",
                "decision_status",
                "active_rules",
                "explanation_text",
            ]
            fieldnames = input_fieldnames + [
                name for name in extra_fields if name not in input_fieldnames
            ]

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_cases += 1
                case = case_builder(row)

                bundle = kernel.evaluate(case)

                decision = bundle.get("decision", OUTCOME_UNKNOWN)
                status = bundle.get("decision_status", STATUS_UNKNOWN)

                active_rules = [
                    str(ax.get("id"))
                    for ax in bundle.get("active_axioms", [])
                    if ax.get("id") is not None
                ]

                explanation = explainer.explain(bundle).to_text(
                    language=config.language
                )

                out_row: Dict[str, Any] = dict(row)
                out_row["decision"] = decision
                out_row["decision_status"] = status
                out_row["active_rules"] = ",".join(active_rules)
                out_row["explanation_text"] = explanation

                writer.writerow(out_row)

        analytics = self._analytics_engine.analyze_log_file(
            log_path=config.log_path,
            ruleset_path=config.ruleset_path,
        )

        return SimulationResult(
            config=config,
            total_cases=total_cases,
            analytics=analytics,
        )


def build_fraud_case_from_row(row: Mapping[str, str]) -> Dict[str, Any]:
    """
    Domyślna funkcja budowania case'a dla prostego demo FRAUD:

    Oczekiwane kolumny CSV:
        - amount
        - tx_count_24h
        - is_pep

    Jeśli kolumn brakuje lub nie da się sparsować wartości, zgłaszany jest
    ValueError z czytelnym komunikatem (tak, aby nadać się do PoC bankowego).
    """
    missing = [col for col in ("amount", "tx_count_24h", "is_pep") if col not in row]
    if missing:
        raise ValueError(
            f"Brak wymaganych kolumn w wierszu CSV: {', '.join(missing)}. "
            f"Otrzymane kolumny: {list(row.keys())}"
        )

    try:
        amount = int(str(row["amount"]).replace("_", ""))
    except Exception as exc:  # pragma: no cover - defensywne
        raise ValueError(f"Nieprawidłowa wartość 'amount': {row['amount']}") from exc

    try:
        tx_count_24h = int(str(row["tx_count_24h"]).replace("_", ""))
    except Exception as exc:  # pragma: no cover - defensywne
        raise ValueError(
            f"Nieprawidłowa wartość 'tx_count_24h': {row['tx_count_24h']}"
        ) from exc

    is_pep_raw = str(row["is_pep"]).strip().lower()
    is_pep = is_pep_raw in {"true", "1", "yes", "y", "t"}

    return {
        "amount": amount,
        "tx_count_24h": tx_count_24h,
        "is_pep": is_pep,
    }
