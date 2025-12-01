"""
rules_io.py

Niewielki, samodzielny moduł do:
- wczytywania reguł z pliku YAML/JSON,
- reprezentowania ich w postaci obiektów,
- ładowania ich do istniejącego kernela regułowego.

Możesz ten plik po prostu skopiować do swojego projektu.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover - zależność opcjonalna
    yaml = None  # type: ignore[assignment]


class RulesetLoadError(Exception):
    """Ogólny błąd podczas wczytywania pliku z regułami."""


class RulesetValidationError(Exception):
    """Błąd walidacji struktury danych wczytanej z pliku."""


class RulesetApplicationError(Exception):
    """Błąd podczas nakładania reguł na kernel."""


@dataclass(frozen=True)
class RuleDefinition:
    """
    Pojedyncza reguła biznesowa.

    Attributes:
        rule_id: Unikalny identyfikator reguły w obrębie rulesetu.
        text: Treść reguły w Twoim DSL / języku naturalnym,
            np. "IF amount > 10000 THEN is_suspicious = TRUE".
        description: Opis biznesowy reguły.
        enabled: Czy reguła jest aktywna (true) czy wyłączona (false).
        severity: Poziom istotności (np. "LOW", "MEDIUM", "HIGH", "CRITICAL").
        tags: Dowolne tagi pomocne w filtrowaniu/raportowaniu.
        metadata: Dodatkowe metadane, np. autor, data, obszar systemu.
    """

    rule_id: str
    text: str
    description: Optional[str] = None
    enabled: bool = True
    severity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleSet:
    """
    Zestaw reguł wczytany z pliku.

    Attributes:
        ruleset_id: Identyfikator rulesetu (np. "fraud_rules_v1").
        version: Wersja rulesetu (np. "1.0.0").
        description: Opis zestawu reguł.
        rules: Lista zdefiniowanych reguł.
    """

    ruleset_id: str
    version: str
    description: Optional[str]
    rules: List[RuleDefinition]


@dataclass(frozen=True)
class RulesetApplicationSummary:
    """
    Podsumowanie nałożenia rulesetu na kernel.

    Attributes:
        ruleset_id: Identyfikator zastosowanego rulesetu.
        version: Wersja zastosowanego rulesetu.
        total_rules: Łączna liczba reguł w pliku.
        enabled_rules: Liczba reguł oznaczonych enabled=True.
        loaded_rules: Liczba reguł faktycznie załadowanych do kernela.
        skipped_rules: Liczba pominiętych reguł (np. disabled).
        errors: Mapa rule_id -> komunikat błędu dla reguł,
            których nie udało się załadować (tylko gdy strict=False).
    """

    ruleset_id: str
    version: str
    total_rules: int
    enabled_rules: int
    loaded_rules: int
    skipped_rules: int
    errors: Dict[str, str] = field(default_factory=dict)


def load_ruleset_from_file(path: Union[str, Path]) -> RuleSet:
    """
    Wczytuje ruleset z pliku YAML lub JSON i zwraca obiekt RuleSet.

    Obsługiwane rozszerzenia:
        - .yaml, .yml  (wymaga zainstalowanego pakietu PyYAML)
        - .json

    Args:
        path: Ścieżka do pliku z regułami.

    Returns:
        RuleSet wypełniony danymi z pliku.

    Raises:
        RulesetLoadError: Gdy plik nie istnieje lub jest nieobsługiwany.
        RulesetValidationError: Gdy struktura pliku jest nieprawidłowa.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise RulesetLoadError(f"Ruleset file does not exist: {file_path}")

    suffix = file_path.suffix.lower()

    try:
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise RulesetLoadError(
                    "Cannot load YAML file because PyYAML is not installed. "
                    "Install it with: pip install pyyaml"
                )
            with file_path.open("r", encoding="utf-8") as file:
                raw_data = yaml.safe_load(file)
        elif suffix == ".json":
            with file_path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
        else:
            raise RulesetLoadError(
                f"Unsupported ruleset file extension: {suffix!r}. "
                "Use .yaml, .yml or .json."
            )
    except RulesetLoadError:
        # Ponownie wyrzucamy nasze własne błędy bez opakowywania.
        raise
    except Exception as exc:
        raise RulesetLoadError(
            f"Failed to read ruleset file {file_path}: {exc}"
        ) from exc

    return _parse_ruleset_dict(raw_data, file_path)


def _parse_ruleset_dict(raw: Any, source: Path) -> RuleSet:
    """
    Konwertuje słownik wczytany z YAML/JSON na obiekt RuleSet.

    Args:
        raw: Dane ze zdeserializowanego pliku.
        source: Ścieżka do pliku (tylko do komunikatów błędów).

    Returns:
        Obiekt RuleSet.

    Raises:
        RulesetValidationError: Gdy struktura danych jest niepoprawna.
    """
    if not isinstance(raw, Mapping):
        raise RulesetValidationError(
            f"Top-level structure in {source} must be a mapping (dict)."
        )

    ruleset_id = str(raw.get("ruleset_id") or source.stem)
    version = str(raw.get("version") or "0.0.0")
    description = raw.get("description")

    raw_rules = raw.get("rules")
    if raw_rules is None:
        raise RulesetValidationError(
            f"Missing required 'rules' list in ruleset file {source}."
        )
    if not isinstance(raw_rules, list):
        raise RulesetValidationError(
            f"Field 'rules' in {source} must be a list."
        )

    rules: List[RuleDefinition] = []
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, Mapping):
            raise RulesetValidationError(
                f"Rule at position {index} in {source} must be a mapping."
            )

        try:
            rule_id = str(raw_rule["id"])
        except KeyError as exc:
            raise RulesetValidationError(
                f"Rule at position {index} in {source} is missing required "
                "'id' field."
            ) from exc

        try:
            text = str(raw_rule["text"])
        except KeyError as exc:
            raise RulesetValidationError(
                f"Rule {rule_id!r} in {source} is missing required 'text' field."
            ) from exc

        description_value = raw_rule.get("description")
        description_text = (
            str(description_value) if description_value is not None else None
        )

        enabled_value = raw_rule.get("enabled", True)
        enabled = bool(enabled_value)

        severity_value = raw_rule.get("severity")
        severity = str(severity_value) if severity_value is not None else None

        tags_value = raw_rule.get("tags", [])
        if not isinstance(tags_value, list):
            raise RulesetValidationError(
                f"Field 'tags' for rule {rule_id!r} in {source} "
                "must be a list of strings."
            )
        tags = [str(tag) for tag in tags_value]

        metadata_value = raw_rule.get("metadata", {})
        if not isinstance(metadata_value, Mapping):
            raise RulesetValidationError(
                f"Field 'metadata' for rule {rule_id!r} in {source} "
                "must be a mapping (dict)."
            )
        metadata = dict(metadata_value)

        rules.append(
            RuleDefinition(
                rule_id=rule_id,
                text=text,
                description=description_text,
                enabled=enabled,
                severity=severity,
                tags=tags,
                metadata=metadata,
            )
        )

    description_text = str(description) if description is not None else None

    return RuleSet(
        ruleset_id=ruleset_id,
        version=version,
        description=description_text,
        rules=rules,
    )


from nl_rule_parser import build_axiom_from_nl


def apply_ruleset_to_kernel(
    kernel: Any,
    ruleset: RuleSet,
    *,
    schema: List[Any],
    decision_field_fallback: str,
    strict: bool = True,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> RulesetApplicationSummary:
    """
    Nakłada ruleset na kernel.

    Zamiast wymagać metody kernel.add_axiom_from_nl(...),
    samodzielnie budujemy AxiomDefinition poprzez build_axiom_from_nl(...)
    i dodajemy go do kernela metodą add_axiom_safe(...),
    która już istnieje w Twoim projekcie.

    Args:
        kernel: Istniejący AxiomKernel.
        ruleset: Wczytany RuleSet.
        schema: Lista VariableSchema używana przez build_axiom_from_nl.
        decision_field_fallback: Nazwa zmiennej decyzyjnej (np. "flag",
            "is_suspicious") używana jako fallback w parserze NL.
        strict: Czy przerwać przy pierwszym błędzie (True) czy zbierać błędy (False).
        extra_metadata: Dodatkowe metadane, które zostaną dodane do każdej reguły.

    Returns:
        RulesetApplicationSummary z podsumowaniem ładowania.

    Raises:
        RulesetApplicationError: Gdy strict=True i nie uda się załadować reguły.
    """
    total_rules = len(ruleset.rules)
    enabled_rules = 0
    loaded_rules = 0
    skipped_rules = 0
    errors: Dict[str, str] = {}

    base_metadata: Dict[str, Any] = {
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_version": ruleset.version,
    }
    if extra_metadata:
        base_metadata.update(dict(extra_metadata))

    for rule in ruleset.rules:
        if not rule.enabled:
            skipped_rules += 1
            continue

        enabled_rules += 1

        metadata = dict(base_metadata)
        metadata.update(rule.metadata)
        if rule.severity is not None:
            metadata.setdefault("severity", rule.severity)
        if rule.tags:
            metadata.setdefault("tags", list(rule.tags))

        try:
            # 1) NL → AxiomDefinition
            axiom = build_axiom_from_nl(
                rule_id=rule.rule_id,
                text=rule.text,
                schema=schema,
                decision_field_fallback=decision_field_fallback,
            )
            # (opcjonalnie można tu kiedyś podpiąć metadata do axiom, jeśli
            # AxiomDefinition ma odpowiednie pole)

            # 2) Dodanie aksjomatu do kernela
            kernel.add_axiom_safe(axiom)
        except Exception as exc:  # pragma: no cover - prosta obsługa błędów
            errors[rule.rule_id] = str(exc)
            if strict:
                raise RulesetApplicationError(
                    f"Failed to apply rule {rule.rule_id!r} from ruleset "
                    f"{ruleset.ruleset_id!r}: {exc}"
                ) from exc
        else:
            loaded_rules += 1

    return RulesetApplicationSummary(
        ruleset_id=ruleset.ruleset_id,
        version=ruleset.version,
        total_rules=total_rules,
        enabled_rules=enabled_rules,
        loaded_rules=loaded_rules,
        skipped_rules=skipped_rules,
        errors=errors,
    )
