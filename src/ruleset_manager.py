"""
ruleset_manager.py

FAZA 2.2 – Governance rulesetów: wersje, środowiska (DEV / TEST / PROD).

Ten moduł zapewnia prosty, ale solidny mechanizm zarządzania zestawami
reguł (RuleSet) w różnych środowiskach wdrożeniowych oraz ich
nakładaniem na AxiomKernel.

Cel:
- mieć jasną informację:
    * który ruleset (id, wersja, plik) jest w DEV / TEST / PROD,
    * kiedy został zarejestrowany / zastosowany,
    * z jakim podsumowaniem (RulesetApplicationSummary) się nałożył.
- umożliwić bezpieczne promowanie rulesetu z DEV → TEST → PROD.

Moduł jest niezależny od konkretnej domeny (AML, fraud, itp.) i opiera
się wyłącznie na interfejsach z:
    - rules_io (RuleSet, load_ruleset_from_file, apply_ruleset_to_kernel),
    - axiomatic_kernel (AxiomKernel).

Przykładowe użycie (skrót):

    from axiomatic_kernel import AxiomKernel, VariableSchema
    from rules_io import load_ruleset_from_file
    from ruleset_manager import Environment, RulesetRegistry

    schema = [...]
    kernel = AxiomKernel(schema=schema, decision_variable="flag")

    registry = RulesetRegistry()

    # Rejestrujemy ruleset w środowisku DEV
    registry.register_ruleset(
        ruleset_id="aml_v1",
        path="rules/rules_aml_v1.yaml",
        environment=Environment.DEV,
    )

    # Nakładamy ruleset na kernel w DEV
    summary = registry.apply_ruleset_to_kernel(
        ruleset_id="aml_v1",
        environment=Environment.DEV,
        kernel=kernel,
        schema=schema,
        decision_field_fallback="flag",
    )

    # Promujemy ten sam ruleset do TEST i PROD
    registry.promote_ruleset("aml_v1", source=Environment.DEV, target=Environment.TEST)
    registry.promote_ruleset("aml_v1", source=Environment.TEST, target=Environment.PROD)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import logging

from rules_io import (
    RuleSet,
    RulesetApplicationError,
    RulesetApplicationSummary,
    RulesetLoadError,
    RulesetValidationError,
    apply_ruleset_to_kernel,
    load_ruleset_from_file,
)

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """
    Środowisko wdrożeniowe rulesetu.

    W praktycznym PoC bankowym to typowy podział:
    - DEV: środowisko rozwojowe,
    - TEST: środowisko testowe / UAT,
    - PROD: środowisko produkcyjne.
    """

    DEV = "DEV"
    TEST = "TEST"
    PROD = "PROD"


@dataclass(frozen=True)
class RulesetKey:
    """
    Klucz identyfikujący ruleset w danym środowisku.
    """

    ruleset_id: str
    environment: Environment


@dataclass
class RulesetRecord:
    """
    Informacja o rulesecie w konkretnym środowisku.

    Attributes:
        key:
            Para (ruleset_id, environment).
        file_path:
            Ścieżka do pliku YAML/JSON, z którego wczytywany jest RuleSet.
        version:
            Wersja rulesetu (zawarta w pliku).
        description:
            Opcjonalny opis rulesetu (z pliku).
        registered_at_utc:
            Kiedy ruleset został zarejestrowany w tym środowisku.
        last_loaded_at_utc:
            Kiedy plik został ostatni raz poprawnie wczytany.
        last_applied_at_utc:
            Kiedy reguły zostały ostatni raz nałożone na kernel.
        last_application_summary:
            Podsumowanie ostatniego nałożenia rulesetu na kernel.
        last_error:
            Ostatni błąd związany z ładowaniem / nakładaniem w tym
            środowisku (jeśli wystąpił).
    """

    key: RulesetKey
    file_path: Path
    version: str
    description: Optional[str] = None
    registered_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_loaded_at_utc: Optional[datetime] = None
    last_applied_at_utc: Optional[datetime] = None
    last_application_summary: Optional[RulesetApplicationSummary] = None
    last_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """
        Zwraca słownikową reprezentację rekordu, ułatwiając logowanie,
        serializację lub wyświetlanie w UI.
        """
        return {
            "ruleset_id": self.key.ruleset_id,
            "environment": self.key.environment.value,
            "file_path": str(self.file_path),
            "version": self.version,
            "description": self.description,
            "registered_at_utc": (
                self.registered_at_utc.isoformat()
                if self.registered_at_utc
                else None
            ),
            "last_loaded_at_utc": (
                self.last_loaded_at_utc.isoformat()
                if self.last_loaded_at_utc
                else None
            ),
            "last_applied_at_utc": (
                self.last_applied_at_utc.isoformat()
                if self.last_applied_at_utc
                else None
            ),
            "last_error": self.last_error,
            "last_application_summary": (
                None
                if self.last_application_summary is None
                else {
                    "ruleset_id": self.last_application_summary.ruleset_id,
                    "version": self.last_application_summary.version,
                    "total_rules": self.last_application_summary.total_rules,
                    "enabled_rules": self.last_application_summary.enabled_rules,
                    "loaded_rules": self.last_application_summary.loaded_rules,
                    "skipped_rules": self.last_application_summary.skipped_rules,
                    "errors": dict(self.last_application_summary.errors),
                }
            ),
        }


class RulesetRegistryError(Exception):
    """Ogólny błąd w warstwie zarządzania rulesetami."""


class RulesetRegistry:
    """
    Prosty rejestr rulesetów z podziałem na środowiska (DEV / TEST / PROD).

    Odpowiada za:
    - rejestrowanie rulesetów (pliku + metadanych),
    - przechowywanie informacji o wersjach i środowiskach,
    - bezpieczne promowanie rulesetu między środowiskami,
    - nakładanie rulesetu na AxiomKernel z zachowaniem informacji
      o ostatnim podsumowaniu (RulesetApplicationSummary).

    Ten rejestr nie jest trwały (in-memory), ale jest gotowy do
    podpięcia persystencji (JSON/DB) jeśli zajdzie taka potrzeba.
    """

    def __init__(self) -> None:
        # Mapujemy (ruleset_id, environment) → RulesetRecord
        self._records: Dict[Tuple[str, Environment], RulesetRecord] = {}

    # ------------------------------------------------------------------
    # Operacje pomocnicze
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_environment(environment: Environment | str) -> Environment:
        if isinstance(environment, Environment):
            return environment
        try:
            return Environment[environment.upper()]
        except KeyError as exc:
            raise RulesetRegistryError(
                f"Unknown environment: {environment!r}. "
                f"Expected one of: {', '.join(e.name for e in Environment)}."
            ) from exc

    def _make_key(self, ruleset_id: str, environment: Environment | str) -> RulesetKey:
        env = self._normalize_environment(environment)
        return RulesetKey(ruleset_id=ruleset_id, environment=env)

    # ------------------------------------------------------------------
    # Publiczne API
    # ------------------------------------------------------------------

    def register_ruleset(
        self,
        ruleset_id: str,
        path: str | Path,
        environment: Environment | str = Environment.DEV,
        *,
        expected_version: Optional[str] = None,
        overwrite: bool = True,
    ) -> RulesetRecord:
        """
        Rejestruje plik rulesetu w wybranym środowisku.

        W ramach rejestracji:
        - wczytywany jest RuleSet z pliku (load_ruleset_from_file),
        - weryfikowane jest dopasowanie ruleset_id,
        - opcjonalnie sprawdzana jest zgodność wersji (expected_version),
        - tworzony jest nowy RulesetRecord i odkładany w rejestrze.

        Parametry:
            ruleset_id:
                Oczekiwany identyfikator rulesetu. Musi zgadzać się z
                ruleset.ruleset_id z pliku.
            path:
                Ścieżka do pliku YAML/JSON z regułami.
            environment:
                Środowisko, w którym rejestrujemy ruleset.
            expected_version:
                Opcjonalna wersja, którą spodziewamy się zobaczyć w pliku.
            overwrite:
                Czy nadpisać istniejący rekord dla tego ruleset_id i
                środowiska (domyślnie True).

        Zwraca:
            RulesetRecord – zapisany w rejestrze.

        Raises:
            RulesetLoadError, RulesetValidationError – gdy plik jest zły.
            RulesetRegistryError – przy niezgodnym ruleset_id / wersji
                lub gdy overwrite=False i rekord już istnieje.
        """
        env = self._normalize_environment(environment)
        file_path = Path(path)

        ruleset = load_ruleset_from_file(file_path)

        if ruleset.ruleset_id != ruleset_id:
            raise RulesetRegistryError(
                "ruleset_id mismatch between registry and file: "
                f"expected {ruleset_id!r}, got {ruleset.ruleset_id!r}."
            )

        if expected_version is not None and ruleset.version != expected_version:
            raise RulesetRegistryError(
                "ruleset version mismatch: "
                f"expected {expected_version!r}, got {ruleset.version!r}."
            )

        key_tuple = (ruleset_id, env)
        if not overwrite and key_tuple in self._records:
            raise RulesetRegistryError(
                f"Ruleset {ruleset_id!r} already registered in "
                f"environment {env.value!r} and overwrite=False."
            )

        record = RulesetRecord(
            key=RulesetKey(ruleset_id=ruleset_id, environment=env),
            file_path=file_path,
            version=ruleset.version,
            description=ruleset.description,
            registered_at_utc=datetime.now(timezone.utc),
            last_loaded_at_utc=datetime.now(timezone.utc),
            last_applied_at_utc=None,
            last_application_summary=None,
            last_error=None,
        )
        self._records[key_tuple] = record

        logger.info(
            "Registered ruleset %r version %s in environment %s from %s",
            ruleset_id,
            ruleset.version,
            env.value,
            file_path,
        )

        return record

    def get_record(
        self,
        ruleset_id: str,
        environment: Environment | str,
    ) -> RulesetRecord:
        """
        Zwraca RulesetRecord dla danego ruleset_id i środowiska.

        Raises:
            RulesetRegistryError – jeśli rekord nie istnieje.
        """
        env = self._normalize_environment(environment)
        key_tuple = (ruleset_id, env)
        try:
            return self._records[key_tuple]
        except KeyError as exc:
            raise RulesetRegistryError(
                f"Ruleset {ruleset_id!r} is not registered in environment "
                f"{env.value!r}."
            ) from exc

    def list_records(
        self,
        environment: Optional[Environment | str] = None,
    ) -> List[RulesetRecord]:
        """
        Zwraca listę wszystkich RulesetRecord, opcjonalnie filtrowaną
        po środowisku.
        """
        if environment is None:
            return list(self._records.values())

        env = self._normalize_environment(environment)
        return [
            record
            for (ruleset_id, record_env), record in self._records.items()
            if record_env == env
        ]

    def promote_ruleset(
        self,
        ruleset_id: str,
        *,
        source: Environment | str,
        target: Environment | str,
        overwrite: bool = True,
    ) -> RulesetRecord:
        """
        Promuje ruleset z jednego środowiska do innego.

        Przykład:
            promote_ruleset("aml_v1", source=Environment.DEV, target=Environment.TEST)

        Zasady:
        - ruleset musi być zarejestrowany w środowisku source,
        - tworzony jest nowy RulesetRecord w środowisku target z:
            * tą samą ścieżką pliku,
            * tym samym ruleset_id i version,
            * nowym timestampem registered_at_utc,
            * wyczyszczonymi polami last_*.

        Raises:
            RulesetRegistryError – przy błędnych środowiskach lub
                próbie nadpisania bez overwrite.
        """
        src = self._normalize_environment(source)
        tgt = self._normalize_environment(target)

        if src == tgt:
            raise RulesetRegistryError(
                "Source and target environments must differ."
            )

        source_record = self.get_record(ruleset_id, src)

        target_key_tuple = (ruleset_id, tgt)
        if not overwrite and target_key_tuple in self._records:
            raise RulesetRegistryError(
                f"Ruleset {ruleset_id!r} already exists in environment "
                f"{tgt.value!r} and overwrite=False."
            )

        promoted = RulesetRecord(
            key=RulesetKey(ruleset_id=ruleset_id, environment=tgt),
            file_path=source_record.file_path,
            version=source_record.version,
            description=source_record.description,
            registered_at_utc=datetime.now(timezone.utc),
            last_loaded_at_utc=None,
            last_applied_at_utc=None,
            last_application_summary=None,
            last_error=None,
        )

        self._records[target_key_tuple] = promoted

        logger.info(
            "Promoted ruleset %r version %s from %s to %s",
            ruleset_id,
            source_record.version,
            src.value,
            tgt.value,
        )

        return promoted

    def apply_ruleset_to_kernel(
        self,
        ruleset_id: str,
        environment: Environment | str,
        *,
        kernel: Any,
        schema: Iterable[Any],
        decision_field_fallback: str,
        strict: bool = True,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        update_kernel_rule_version: bool = True,
    ) -> RulesetApplicationSummary:
        """
        Wczytuje ruleset z pliku i nakłada go na podany kernel.

        Po udanym nałożeniu:
        - uaktualnia last_applied_at_utc i last_application_summary
          w odpowiednim RulesetRecord,
        - opcjonalnie ustawia w kernelu rule_version na
          "<ruleset_id>:<version>@<environment>".

        Parametry:
            ruleset_id:
                Identyfikator rulesetu (taki jak w pliku i rejestrze).
            environment:
                Środowisko, z którego chcemy użyć rulesetu.
            kernel:
                Instancja AxiomKernel (lub kompatybilnego kernela).
            schema:
                Lista VariableSchema przekazywana do apply_ruleset_to_kernel.
            decision_field_fallback:
                Nazwa zmiennej decyzyjnej używana w parserze NL.
            strict:
                Przekazywane do apply_ruleset_to_kernel – czy przerywać
                przy pierwszym błędzie.
            extra_metadata:
                Dodatkowe metadane do przekazania przy budowie reguł.
            update_kernel_rule_version:
                Jeśli True, po udanym nałożeniu spróbujemy ustawić
                atrybut rule_version/_rule_version w kernelu, żeby logi
                decyzyjne zawierały informację o użytym rulesetcie.

        Zwraca:
            RulesetApplicationSummary z apply_ruleset_to_kernel.

        Raises:
            RulesetLoadError, RulesetValidationError, RulesetApplicationError
            – zgodnie z rules_io.apply_ruleset_to_kernel.
            RulesetRegistryError – jeśli ruleset nie jest zarejestrowany.
        """
        env = self._normalize_environment(environment)
        record = self.get_record(ruleset_id, env)

        # Świeżo wczytujemy RuleSet z pliku (idempotentnie) – dzięki temu
        # mamy pewność, że używamy aktualnej zawartości pliku.
        ruleset = load_ruleset_from_file(record.file_path)

        if ruleset.ruleset_id != ruleset_id:
            raise RulesetRegistryError(
                "ruleset_id mismatch between registry and file during apply: "
                f"expected {ruleset_id!r}, got {ruleset.ruleset_id!r}."
            )

        if ruleset.version != record.version:
            logger.warning(
                "Ruleset version changed on disk for %r in %s: "
                "was %s, now %s. Updating registry record.",
                ruleset_id,
                env.value,
                record.version,
                ruleset.version,
            )
            record.version = ruleset.version

        try:
            summary = apply_ruleset_to_kernel(
                kernel=kernel,
                ruleset=ruleset,
                schema=list(schema),
                decision_field_fallback=decision_field_fallback,
                strict=strict,
                extra_metadata=extra_metadata,
            )
        except (RulesetApplicationError, RulesetValidationError, RulesetLoadError):
            record.last_error = "Failed to apply ruleset to kernel."
            logger.exception(
                "Failed to apply ruleset %r in environment %s",
                ruleset_id,
                env.value,
            )
            raise

        now = datetime.now(timezone.utc)
        record.last_loaded_at_utc = now
        record.last_applied_at_utc = now
        record.last_application_summary = summary
        record.last_error = None

        if update_kernel_rule_version:
            self._update_kernel_rule_version(
                kernel=kernel,
                ruleset_id=ruleset_id,
                ruleset_version=ruleset.version,
                environment=env,
            )

        logger.info(
            "Applied ruleset %r version %s in environment %s: "
            "%d/%d rules loaded (%d skipped)",
            ruleset_id,
            ruleset.version,
            env.value,
            summary.loaded_rules,
            summary.total_rules,
            summary.skipped_rules,
        )

        return summary

    # ------------------------------------------------------------------
    # Pomocnicze: aktualizacja rule_version w kernelu
    # ------------------------------------------------------------------

    @staticmethod
    def _update_kernel_rule_version(
        *,
        kernel: Any,
        ruleset_id: str,
        ruleset_version: str,
        environment: Environment,
    ) -> None:
        """
        Próbuje ustawić wersję reguł w kernelu w sposób bezpieczny.

        W Twoim AxiomKernel wersja jest przechowywana w atrybucie
        _rule_version, więc metoda próbuje najpierw ustawić to pole.
        Jeśli nie istnieje, spróbuje ustawić publiczny atrybut
        rule_version. Gdy to się nie uda, loguje ostrzeżenie,
        ale nie przerywa działania.

        Format wersji w kernelu:
            "<ruleset_id>:<ruleset_version>@<environment>"
        """
        version_label = f"{ruleset_id}:{ruleset_version}@{environment.value}"

        # Najpierw spróbujmy wewnętrzne pole _rule_version
        if hasattr(kernel, "_rule_version"):
            try:
                setattr(kernel, "_rule_version", version_label)
                return
            except Exception:  # pragma: no cover - bardzo defensywne
                logger.exception(
                    "Failed to set _rule_version on kernel to %r",
                    version_label,
                )

        # Następnie spróbujmy publiczny atrybut rule_version
        if hasattr(kernel, "rule_version"):
            try:
                setattr(kernel, "rule_version", version_label)
                return
            except Exception:  # pragma: no cover - bardzo defensywne
                logger.exception(
                    "Failed to set rule_version on kernel to %r",
                    version_label,
                )

        logger.warning(
            "Kernel does not expose rule_version/_rule_version attribute; "
            "cannot update rule version to %r.",
            version_label,
        )
