from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import csv


@dataclass(frozen=True)
class SegmentKey:
    """Klucz segmentu: bucket kwoty, bucket liczby transakcji, flaga PEP."""

    amount_band: str
    tx_band: str
    is_pep: bool

    def label(self) -> str:
        """Zwraca czytelną etykietę segmentu."""
        pep_label = "PEP" if self.is_pep else "non-PEP"
        return (
            f"amount={self.amount_band}, "
            f"tx_count_24h={self.tx_band}, "
            f"{pep_label}"
        )


@dataclass
class SegmentStats:
    """Statystyki dla pojedynczego segmentu danych."""

    key: SegmentKey
    total: int = 0
    flagged: int = 0
    clean: int = 0
    other: int = 0
    # liczba decyzji FLAGGED, w których dana reguła była aktywna
    rule_counts_flagged: Dict[str, int] = field(default_factory=dict)
    # liczba decyzji CLEAN, w których dana reguła była aktywna
    rule_counts_clean: Dict[str, int] = field(default_factory=dict)

    def register_case(
        self,
        *,
        decision: str,
        active_rules: Iterable[str],
    ) -> None:
        """Aktualizuje statystyki na podstawie pojedynczego rekordu CSV."""
        self.total += 1

        decision_upper = decision.upper()
        if decision_upper == "FLAGGED":
            self.flagged += 1
            target = self.rule_counts_flagged
        elif decision_upper == "CLEAN":
            self.clean += 1
            target = self.rule_counts_clean
        else:
            self.other += 1
            target = None

        if target is not None:
            # unikamy wielokrotnego liczenia tej samej reguły w jednej decyzji
            for rule_id in set(active_rules):
                target[rule_id] = target.get(rule_id, 0) + 1

    @property
    def flagged_rate(self) -> float:
        """Udział decyzji FLAGGED w segmencie."""
        if self.total == 0:
            return 0.0
        return self.flagged / self.total

    @property
    def distinct_rules_flagged(self) -> int:
        """Liczba różnych reguł aktywnych w decyzjach FLAGGED."""
        return len(self.rule_counts_flagged)

    def top_rules_flagged(self, limit: int = 3) -> List[tuple[str, int]]:
        """Lista (rule_id, count) posortowana malejąco po liczbie wystąpień w FLAGGED."""
        if not self.rule_counts_flagged:
            return []
        items = sorted(
            self.rule_counts_flagged.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return items[:limit]


@dataclass
class RuleGapsConfig:
    """Konfiguracja wykrywania luk w regułach.

    Heurystyki:
    - min_total_cases:
        minimalna liczba decyzji w segmencie, żeby go w ogóle rozważać.
    - min_flagged_rate:
        minimalny udział FLAGGED w segmencie (np. 0.3 = 30%),
        poniżej tego segment raczej nie jest „problemowy”.
    - max_distinct_rules_in_gap:
        maksymalna liczba różnych reguł aktywnych w FLAGGED,
        żeby uznać, że „zbyt mało reguł opisuje ten obszar”.
    - min_dominant_rule_share:
        minimalny udział najsilniejszej reguły we FLAGGED,
        żeby mówić o „dominacji jednej reguły” (np. 0.6 = 60%).
    """

    min_total_cases: int = 20
    min_flagged_rate: float = 0.2
    max_distinct_rules_in_gap: int = 2
    min_dominant_rule_share: float = 0.6


@dataclass
class GapSegmentInsight:
    """Opis pojedynczego segmentu uznanego za lukę w regułach."""

    key: SegmentKey
    total: int
    flagged: int
    clean: int
    flagged_rate: float
    dominant_rules: List[str]
    dominant_rules_share: float
    note_text: str


@dataclass
class RuleGapsResult:
    """Pełny wynik analizy luk w regułach."""

    segments: Dict[SegmentKey, SegmentStats]
    gap_segments: List[GapSegmentInsight]


class RuleGapsEngine:
    """Silnik analizy luk w regułach na podstawie CSV z decyzjami."""

    def __init__(self, config: Optional[RuleGapsConfig] = None) -> None:
        self._config = config or RuleGapsConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_csv(self, csv_path: str | Path) -> RuleGapsResult:
        """
        Analizuje plik CSV z decyzjami i wyjaśnieniami, taki jak
        transactions_with_explanations.csv.

        Wymagane kolumny:
        - amount
        - tx_count_24h
        - is_pep
        - decision
        - active_rules (lista id reguł rozdzielona przecinkami lub pusta).

        Zwraca RuleGapsResult z:
        - pełnymi statystykami segmentów,
        - listą segmentów zakwalifikowanych jako „gapy”.
        """
        path = Path(csv_path)
        segments: Dict[SegmentKey, SegmentStats] = {}

        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                try:
                    amount = int(str(row.get("amount", "0")).replace("_", ""))
                except ValueError:
                    # jeżeli nie da się sparsować – pomijamy wiersz
                    continue

                try:
                    tx_count = int(str(row.get("tx_count_24h", "0")))
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

                decision = str(row.get("decision", "")).strip()

                active_rules_raw = str(row.get("active_rules", "")).strip()
                if not active_rules_raw or active_rules_raw == "–":
                    active_rules: List[str] = []
                else:
                    active_rules = [
                        rule_id.strip()
                        for rule_id in active_rules_raw.split(",")
                        if rule_id.strip()
                    ]

                key = SegmentKey(
                    amount_band=self._bucket_amount(amount),
                    tx_band=self._bucket_tx_count(tx_count),
                    is_pep=is_pep,
                )

                stats = segments.get(key)
                if stats is None:
                    stats = SegmentStats(key=key)
                    segments[key] = stats

                stats.register_case(
                    decision=decision,
                    active_rules=active_rules,
                )

        gap_segments = self._detect_gaps(segments)

        return RuleGapsResult(
            segments=segments,
            gap_segments=gap_segments,
        )

    # ------------------------------------------------------------------
    # Logika wykrywania gapów
    # ------------------------------------------------------------------

    def _detect_gaps(
        self,
        segments: Dict[SegmentKey, SegmentStats],
    ) -> List[GapSegmentInsight]:
        cfg = self._config
        gap_segments: List[GapSegmentInsight] = []

        for stats in segments.values():
            if stats.total < cfg.min_total_cases:
                continue
            if stats.flagged_rate < cfg.min_flagged_rate:
                continue
            if stats.flagged == 0:
                continue

            distinct_rules = stats.distinct_rules_flagged

            # Przypadek skrajny:
            # dużo flagowań, ale żadna reguła nigdy nie jest aktywna w FLAGGED
            if distinct_rules == 0:
                note_text = self._render_note_no_rules(stats)
                gap_segments.append(
                    GapSegmentInsight(
                        key=stats.key,
                        total=stats.total,
                        flagged=stats.flagged,
                        clean=stats.clean,
                        flagged_rate=stats.flagged_rate,
                        dominant_rules=[],
                        dominant_rules_share=0.0,
                        note_text=note_text,
                    )
                )
                continue

            if distinct_rules > cfg.max_distinct_rules_in_gap:
                # w tym segmencie aktywuje się wiele różnych reguł – to raczej
                # nie jest luka, tylko złożony obszar
                continue

            top_rules = stats.top_rules_flagged(
                limit=cfg.max_distinct_rules_in_gap,
            )
            if not top_rules:
                continue

            dominant_rule_id, dominant_count = top_rules[0]
            _ = dominant_rule_id  # zmienna użyta tylko informacyjnie
            dominant_share = dominant_count / stats.flagged

            if dominant_share < cfg.min_dominant_rule_share:
                # żadna pojedyncza reguła nie dominuje – segment nie wygląda
                # jak „gap” tylko na podstawie tej heurystyki
                continue

            note_text = self._render_note_dominant_rules(
                stats=stats,
                top_rules=top_rules,
                dominant_share=dominant_share,
            )

            gap_segments.append(
                GapSegmentInsight(
                    key=stats.key,
                    total=stats.total,
                    flagged=stats.flagged,
                    clean=stats.clean,
                    flagged_rate=stats.flagged_rate,
                    dominant_rules=[rule_id for rule_id, _ in top_rules],
                    dominant_rules_share=dominant_share,
                    note_text=note_text,
                )
            )

        # sortujemy „gapy” malejąco po liczbie FLAGGED, żeby najważniejsze
        # były na górze raportu
        gap_segments.sort(
            key=lambda seg: (seg.flagged, seg.flagged_rate),
            reverse=True,
        )
        return gap_segments

    # ------------------------------------------------------------------
    # Teksty wyjaśniające gapy
    # ------------------------------------------------------------------

    def _render_note_no_rules(self, stats: SegmentStats) -> str:
        key_label = stats.key.label()
        flagged_pct = stats.flagged_rate * 100.0
        if stats.total == 0:
            return (
                f"Segment {key_label} nie zawiera żadnych decyzji – brak danych."
            )

        return (
            f"Segment {key_label} ma {stats.total} decyzji, z czego "
            f"{stats.flagged} ({flagged_pct:.1f}%) jest OFLAGOWANYCH, "
            "ale w transakcjach OFLAGOWANYCH nie aktywowała się żadna reguła. "
            "To silna wskazówka, że brakuje dedykowanych reguł opisujących ten obszar."
        )

    def _render_note_dominant_rules(
        self,
        *,
        stats: SegmentStats,
        top_rules: List[tuple[str, int]],
        dominant_share: float,
    ) -> str:
        key_label = stats.key.label()
        flagged_pct = stats.flagged_rate * 100.0
        dominant_pct = dominant_share * 100.0

        rules_parts = [
            f"{rule_id} ({count} razy)"
            for rule_id, count in top_rules
        ]
        rules_text = ", ".join(rules_parts)

        return (
            f"Segment {key_label} ma {stats.total} decyzji, z czego "
            f"{stats.flagged} ({flagged_pct:.1f}%) jest OFLAGOWANYCH. "
            "W oflagowanych decyzjach aktywuje się bardzo ograniczony zestaw "
            f"reguł (łącznie {stats.distinct_rules_flagged}), z dominującymi: "
            f"{rules_text}. Najmocniejsza reguła pokrywa około "
            f"{dominant_pct:.1f}% wszystkich flagowań w tym segmencie. "
            "To sugeruje, że warto rozważyć doprecyzowanie logiki dla tego "
            "segmentu (np. rozbicie na bardziej szczegółowe reguły) lub "
            "sprawdzenie, czy brak dodatkowych reguł nie powoduje nadmiernego "
            "obciążenia pojedynczej reguły."
        )

    # ------------------------------------------------------------------
    # Bucketowanie cech
    # ------------------------------------------------------------------

    @staticmethod
    def _bucket_amount(amount: int) -> str:
        """Przypisuje kwotę do jednego z przedziałów opisowych."""
        if amount < 1_000:
            return "[0, 1k)"
        if amount < 5_000:
            return "[1k, 5k)"
        if amount < 20_000:
            return "[5k, 20k)"
        if amount < 100_000:
            return "[20k, 100k)"
        return "100k+"

    @staticmethod
    def _bucket_tx_count(tx_count: int) -> str:
        """Przypisuje liczbę transakcji 24h do jednego z bucketów."""
        if tx_count <= 1:
            return "0–1"
        if tx_count <= 5:
            return "2–5"
        if tx_count <= 10:
            return "6–10"
        if tx_count <= 20:
            return "11–20"
        return "20+"
