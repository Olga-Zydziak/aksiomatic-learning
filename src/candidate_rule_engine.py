from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import csv

from axiomatic_kernel import AxiomKernel, AxiomConflict, VariableSchema
from nl_rule_parser import build_axiom_from_nl, RuleParseError
from rules_io import load_ruleset_from_file, apply_ruleset_to_kernel
from rule_gaps import GapSegmentInsight, RuleGapsResult, SegmentKey
from decision_constants import OUTCOME_CLEAN, OUTCOME_FLAGGED


@dataclass
class CandidateRuleConfig:
    """
    Konfiguracja generowania kandydatów reguł.

    Attributes:
        min_triggered_cases:
            Minimalna liczba transakcji spełniających warunek reguły,
            aby kandydat był raportowany.
        max_candidates_per_gap:
            Maksymalna liczba kandydatów generowanych na jeden GAP.
            W aktualnej implementacji wykorzystujemy tylko 1 kandydat
            na segment, ale parametr zostawiamy na przyszłość.
    """

    min_triggered_cases: int = 10
    max_candidates_per_gap: int = 1


@dataclass
class CandidateRuleMetrics:
    """Metryki oceny kandydata reguły na danych historycznych."""

    total_cases: int
    triggered_total: int
    triggered_flagged: int
    triggered_clean: int
    triggered_other: int
    segment_total: int
    segment_flagged: int
    segment_clean: int
    segment_flagged_rate: float

    @property
    def triggered_share(self) -> float:
        """Udział transakcji spełniających warunek reguły w całym zbiorze."""
        if self.total_cases == 0:
            return 0.0
        return self.triggered_total / self.total_cases

    @property
    def clean_to_flagged_ratio(self) -> float:
        """
        Stosunek CLEAN do FLAGGED w obszarze działania reguły.

        Jeśli nie ma żadnych FLAGGED, zwraca 0.0.
        """
        if self.triggered_flagged == 0:
            return 0.0
        return self.triggered_clean / self.triggered_flagged


@dataclass
class CandidateRuleProofInfo:
    """Informacja o spójności kandydata z istniejącym rulesetem."""

    is_conflict_free: bool
    conflict_count: int
    conflict_details: List[str] = field(default_factory=list)


@dataclass
class RuleCandidate:
    """Pełny opis kandydata reguły."""

    rule_id: str
    segment: GapSegmentInsight
    nl_rule_text: str
    description: str
    metrics: CandidateRuleMetrics
    proof: CandidateRuleProofInfo


class CandidateRuleEngine:
    """
    Silnik generowania kandydatów reguł na podstawie GAP-ów.

    Ten silnik:
    - bierze wynik RuleGapsResult (segmenty zidentyfikowane jako luki),
    - generuje dla każdego segmentu jedną propozycję reguły w stylu NL,
    - ocenia ją na danych historycznych (CSV),
    - sprawdza spójność z istniejącym rulesetem
      poprzez AxiomKernel + add_axiom_safe.

    Cel: dostarczyć analitykowi AML listę kandydatek reguł, które:
    - opisują dobrze zidentyfikowane segmenty danych,
    - są kompatybilne z istniejącymi regułami (brak konfliktów Z3),
    - mają czytelny opis metryk.
    """

    def __init__(
        self,
        *,
        schema: Sequence[VariableSchema],
        decision_field: str,
        ruleset_path: Path,
        config: Optional[CandidateRuleConfig] = None,
        rule_id_prefix: str = "fraud.candidate_gap",
    ) -> None:
        self._schema: List[VariableSchema] = list(schema)
        self._decision_field = decision_field
        self._ruleset_path = Path(ruleset_path)
        self._config = config or CandidateRuleConfig()
        self._rule_id_prefix = rule_id_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_candidates_from_gaps(
        self,
        gaps_result: RuleGapsResult,
        *,
        csv_path: Path,
    ) -> List[RuleCandidate]:
        """
        Generuje listę kandydatów reguł na bazie wyników RuleGapsEngine.

        Args:
            gaps_result:
                Wynik RuleGapsEngine z listą segmentów uznanych za potencjalne luki.
            csv_path:
                Ścieżka do pliku CSV (np. transactions_with_explanations.csv),
                na podstawie którego liczymy metryki kandydata.

        Returns:
            Lista RuleCandidate (posortowana wg liczby trafień w segmencie malejąco).
        """
        candidates: List[RuleCandidate] = []

        for index, gap in enumerate(gaps_result.gap_segments, start=1):
            rule_id = f"{self._rule_id_prefix}.{index:03d}"
            nl_rule_text = self._build_nl_rule_for_segment(gap.key)
            description = (
                "Kandydat reguły dla segmentu: "
                f"{gap.key.label()} "
                "(automatycznie wygenerowany na podstawie Rule Gaps)."
            )

            metrics = self._evaluate_candidate_on_csv(
                segment_key=gap.key,
                csv_path=csv_path,
                segment_stats=gap,
            )

            if metrics.triggered_total < self._config.min_triggered_cases:
                # Kandydat jest zbyt wąski – brak wystarczającej liczby przykładów
                # do sensownej oceny.
                continue

            proof = self._check_candidate_consistency(
                rule_id=rule_id,
                nl_rule_text=nl_rule_text,
            )

            candidate = RuleCandidate(
                rule_id=rule_id,
                segment=gap,
                nl_rule_text=nl_rule_text,
                description=description,
                metrics=metrics,
                proof=proof,
            )
            candidates.append(candidate)

        # Sortujemy: najpierw po liczbie decyzji w segmencie (malejąco),
        # potem po triggered_total (również malejąco).
        candidates.sort(
            key=lambda c: (c.segment.flagged, c.metrics.triggered_total),
            reverse=True,
        )
        return candidates

    # ------------------------------------------------------------------
    # Budowanie tekstu reguły NL
    # ------------------------------------------------------------------

    def _build_nl_rule_for_segment(self, segment_key: SegmentKey) -> str:
        """
        Buduje tekst reguły NL opisującej dany segment.

        Reguła ma postać:
            IF ...warunki_na_amount_i_tx_i_pep... THEN is_suspicious = TRUE
        """
        conditions: List[str] = []

        min_amount, max_amount = self._amount_bounds(segment_key.amount_band)
        if min_amount is not None:
            conditions.append(f"amount >= {min_amount}")
        if max_amount is not None:
            conditions.append(f"amount < {max_amount}")

        min_tx, max_tx = self._tx_bounds(segment_key.tx_band)
        if min_tx is not None:
            conditions.append(f"tx_count_24h >= {min_tx}")
        if max_tx is not None:
            conditions.append(f"tx_count_24h <= {max_tx}")

        pep_literal = "TRUE" if segment_key.is_pep else "FALSE"
        conditions.append(f"is_pep == {pep_literal}")

        condition_text = " AND ".join(conditions)
        return f"IF {condition_text} THEN {self._decision_field} = TRUE"

    # ------------------------------------------------------------------
    # Ocena kandydata na CSV
    # ------------------------------------------------------------------

    def _evaluate_candidate_on_csv(
        self,
        *,
        segment_key: SegmentKey,
        csv_path: Path,
        segment_stats: GapSegmentInsight,
    ) -> CandidateRuleMetrics:
        total_cases = 0
        triggered_total = 0
        triggered_flagged = 0
        triggered_clean = 0
        triggered_other = 0

        path = Path(csv_path)
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_cases += 1

                try:
                    amount_raw = str(row.get("amount", "")).replace("_", "")
                    amount = int(amount_raw)
                except ValueError:
                    # Jeśli nie da się sparsować kwoty, pomijamy wiersz
                    continue

                try:
                    tx_raw = str(row.get("tx_count_24h", "0"))
                    tx_count = int(tx_raw)
                except ValueError:
                    tx_count = 0

                is_pep_raw = str(row.get("is_pep", "")).strip()
                is_pep = is_pep_raw.lower() in {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "t",
                }

                if not self._matches_segment(amount, tx_count, is_pep, segment_key):
                    continue

                triggered_total += 1
                decision = str(row.get("decision", "")).strip().upper()

                if decision == "FLAGGED":
                    triggered_flagged += 1
                elif decision == "CLEAN":
                    triggered_clean += 1
                else:
                    triggered_other += 1

        segment_total = segment_stats.total
        segment_flagged = segment_stats.flagged
        segment_clean = segment_stats.clean
        segment_flagged_rate = segment_stats.flagged_rate

        return CandidateRuleMetrics(
            total_cases=total_cases,
            triggered_total=triggered_total,
            triggered_flagged=triggered_flagged,
            triggered_clean=triggered_clean,
            triggered_other=triggered_other,
            segment_total=segment_total,
            segment_flagged=segment_flagged,
            segment_clean=segment_clean,
            segment_flagged_rate=segment_flagged_rate,
        )

    # ------------------------------------------------------------------
    # Spójność z istniejącym rulesetem (Z3)
    # ------------------------------------------------------------------

    def _check_candidate_consistency(
        self,
        *,
        rule_id: str,
        nl_rule_text: str,
    ) -> CandidateRuleProofInfo:
        """
        Sprawdza, czy kandydat jest spójny z istniejącym rulesetem.

        Procedura:
        - budujemy świeży AxiomKernel z tym samym schema i decision_field,
        - wczytujemy ruleset z pliku YAML/JSON,
        - nakładamy go na kernel,
        - próbujemy dodać nowy aksjomat przez add_axiom_safe z raise_on_conflict=False,
        - na podstawie listy konfliktów budujemy CandidateRuleProofInfo.
        """
        # 1. Budowa tymczasowego kernela
        kernel = AxiomKernel(
            schema=self._schema,
            decision_variable=self._decision_field,
            logger=None,
            rule_version="candidate_eval",
        )

        # 2. Wczytanie bazowego rulesetu
        ruleset = load_ruleset_from_file(self._ruleset_path)

        # 3. Nałożenie rulesetu na kernel
        apply_ruleset_to_kernel(
            kernel=kernel,
            ruleset=ruleset,
            schema=list(self._schema),
            decision_field_fallback=self._decision_field,
            strict=True,
            extra_metadata={"domain": "candidate-eval"},
        )

        # 4. Budowa AxiomDefinition z tekstu reguły
        try:
            axiom = build_axiom_from_nl(
                rule_id=rule_id,
                text=nl_rule_text,
                schema=self._schema,
                decision_field_fallback=self._decision_field,
                description=f"Candidate rule for segment ({nl_rule_text})",
                priority=0,
            )
        except RuleParseError as exc:
            # Jeśli parser nie radzi sobie z naszą regułą, zwracamy informację
            # o błędzie jako "konflikt" specjalnego typu.
            return CandidateRuleProofInfo(
                is_conflict_free=False,
                conflict_count=1,
                conflict_details=[
                    f"RuleParseError: {exc}",
                ],
            )

        # 5. Próba dodania do kernela z detekcją konfliktów
        conflicts: List[AxiomConflict] = kernel.add_axiom_safe(
            axiom,
            raise_on_conflict=False,
        )

        if not conflicts:
            return CandidateRuleProofInfo(
                is_conflict_free=True,
                conflict_count=0,
                conflict_details=[],
            )

        details: List[str] = []
        for conflict in conflicts:
            # Staramy się zbudować krótką, czytelną linijkę dla raportu
            message = (
                f"Konflikt z istniejącą regułą {conflict.existing_axiom_id!r}: "
                f"{conflict.existing_description}"
            )
            details.append(message)

        return CandidateRuleProofInfo(
            is_conflict_free=False,
            conflict_count=len(conflicts),
            conflict_details=details,
        )

    # ------------------------------------------------------------------
    # Pomocnicze: dopasowanie do segmentu i bucketowanie
    # ------------------------------------------------------------------

    def _matches_segment(
        self,
        amount: int,
        tx_count: int,
        is_pep: bool,
        segment_key: SegmentKey,
    ) -> bool:
        min_amount, max_amount = self._amount_bounds(segment_key.amount_band)
        min_tx, max_tx = self._tx_bounds(segment_key.tx_band)

        if min_amount is not None and amount < min_amount:
            return False
        if max_amount is not None and amount >= max_amount:
            return False

        if min_tx is not None and tx_count < min_tx:
            return False
        if max_tx is not None and tx_count > max_tx:
            return False

        if is_pep != segment_key.is_pep:
            return False

        return True

    @staticmethod
    def _amount_bounds(amount_band: str) -> tuple[Optional[int], Optional[int]]:
        """
        Zwraca (min_amount, max_amount) dla danego bucketu kwoty.

        max_amount jest traktowane jako górna granica *wyłączna* (strict <).
        """
        mapping: Dict[str, tuple[Optional[int], Optional[int]]] = {
            "[0, 1k)": (0, 1_000),
            "[1k, 5k)": (1_000, 5_000),
            "[5k, 20k)": (5_000, 20_000),
            "[20k, 100k)": (20_000, 100_000),
            "100k+": (100_000, None),
        }
        if amount_band not in mapping:
            raise ValueError(f"Unknown amount band: {amount_band!r}")
        return mapping[amount_band]

    @staticmethod
    def _tx_bounds(tx_band: str) -> tuple[Optional[int], Optional[int]]:
        """
        Zwraca (min_tx, max_tx) dla danego bucketu liczby transakcji.

        max_tx jest traktowane jako górna granica *włączna* (<=).
        """
        mapping: Dict[str, tuple[Optional[int], Optional[int]]] = {
            "0–1": (0, 1),
            "2–5": (2, 5),
            "6–10": (6, 10),
            "11–20": (11, 20),
            "20+": (20, None),
        }
        if tx_band not in mapping:
            raise ValueError(f"Unknown tx_count band: {tx_band!r}")
        return mapping[tx_band]
