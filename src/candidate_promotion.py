# src/candidate_promotion.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Literal

try:
    import yaml  # type: ignore[import]
except Exception as exc:  # pragma: no cover - optional dependency guard
    yaml = None  # type: ignore[assignment]

from candidate_rule_engine import RuleCandidate  # type: ignore[import]
from rules_io import RuleDefinition, RuleSet, load_ruleset_from_file


@dataclass(frozen=True)
class PromotionConfig:
    """
    Konfiguracja promowania kandydatów do nowego rulesetu.

    Attributes:
        bump_part:
            Którą część wersji semver zwiększyć: 'major', 'minor' lub 'patch'.
        default_severity:
            Domyślny poziom istotności dla nowych reguł.
        base_tags:
            Tagi, które zostaną dodane do każdej promowanej reguły.
    """

    bump_part: Literal["major", "minor", "patch"] = "minor"
    default_severity: str = "MEDIUM"
    base_tags: List[str] | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        """Ustawia bezpieczną domyślną listę tagów.

        dataclass(frozen=True) + mutowalne domyślne wymaga obejścia
        przez bezpośrednie użycie object.__setattr__.
        """
        if self.base_tags is None:
            object.__setattr__(self, "base_tags", ["candidate_from_gap"])


def bump_semver(
    version: str,
    *,
    part: Literal["major", "minor", "patch"] = "minor",
) -> str:
    """Zwiększa wersję semantyczną.

    Akceptuje wersje w formie:
    - '1.0.0'
    - '1.2'
    - '1'
    - opcjonalnie z sufiksami (np. '1.0.0-beta'), które są zachowywane.

    Przykłady:
        bump_semver("1.0.0", part="minor") -> "1.1.0"
        bump_semver("1.2", part="patch") -> "1.2.1"
    """
    if not version:
        return "0.1.0" if part != "major" else "1.0.0"

    core, *suffix_parts = version.split("-", maxsplit=1)
    suffix = f"-{suffix_parts[0]}" if suffix_parts else ""

    segments = core.split(".")
    numeric: List[int] = []
    for segment in segments:
        try:
            numeric.append(int(segment))
        except ValueError:
            numeric.append(0)

    while len(numeric) < 3:
        numeric.append(0)

    major, minor, patch = numeric[:3]

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}{suffix}"


def merge_ruleset_with_candidates(
    *,
    base_ruleset: RuleSet,
    candidates: Sequence[RuleCandidate],
    accepted_ids: Iterable[str],
    new_ruleset_id: str | None = None,
    promotion_config: PromotionConfig | None = None,
) -> RuleSet:
    """Tworzy nowy RuleSet na bazie istniejącego + zaakceptowanych kandydatów.

    Args:
        base_ruleset:
            Istniejący ruleset (np. fraud_rules_v1).
        candidates:
            Lista RuleCandidate z CandidateRuleEngine.generate_candidates_from_gaps(...).
        accepted_ids:
            Identyfikatory reguł (RuleCandidate.rule_id), które mają zostać
            włączone do nowego rulesetu.
        new_ruleset_id:
            Identyfikator nowego rulesetu. Jeśli None, zostanie użyty
            base_ruleset.ruleset_id.
        promotion_config:
            Konfiguracja promowania (bump wersji, domyślne severity/tags).

    Returns:
        Nowy RuleSet, który można zapisać do pliku i zarejestrować w registry.
    """
    cfg = promotion_config or PromotionConfig()
    accepted_set = {rule_id.strip() for rule_id in accepted_ids if rule_id.strip()}

    if not accepted_set:
        raise ValueError("accepted_ids must contain at least one rule id.")

    existing_rule_ids = {rule.rule_id for rule in base_ruleset.rules}

    # Upewniamy się, że żadna nowa reguła nie nadpisze istniejącej.
    conflicting_ids = accepted_set & existing_rule_ids
    if conflicting_ids:
        conflicts_str = ", ".join(sorted(conflicting_ids))
        raise ValueError(
            "Cannot promote candidates with rule_ids that already exist "
            f"in base ruleset: {conflicts_str}."
        )

    # Kopiujemy listę reguł bazowych (RuleDefinition jest niemutowalny).
    new_rules: List[RuleDefinition] = list(base_ruleset.rules)

    # Mapujemy kandydatów po id dla szybkiego lookupu.
    candidates_by_id = {candidate.rule_id: candidate for candidate in candidates}

    missing_ids = accepted_set - set(candidates_by_id)
    if missing_ids:
        missing_str = ", ".join(sorted(missing_ids))
        raise ValueError(
            "Some accepted_ids do not exist in provided candidates list: "
            f"{missing_str}."
        )

    for rule_id in sorted(accepted_set):
        candidate = candidates_by_id[rule_id]
        segment_key = candidate.segment.key

        metrics = candidate.metrics
        proof = candidate.proof

        metadata = {
            "source": "candidate_rule_engine",
            "segment": {
                "amount_band": segment_key.amount_band,
                "tx_band": segment_key.tx_band,
                "is_pep": segment_key.is_pep,
                "label": segment_key.label(),
            },
            "metrics": {
                "total_cases": metrics.total_cases,
                "triggered_total": metrics.triggered_total,
                "triggered_flagged": metrics.triggered_flagged,
                "triggered_clean": metrics.triggered_clean,
                "triggered_other": metrics.triggered_other,
                "segment_total": metrics.segment_total,
                "segment_flagged": metrics.segment_flagged,
                "segment_clean": metrics.segment_clean,
                "segment_flagged_rate": metrics.segment_flagged_rate,
                "triggered_share": metrics.triggered_share,
                "clean_to_flagged_ratio": metrics.clean_to_flagged_ratio,
            },
            "proof": {
                "is_conflict_free": proof.is_conflict_free,
                "conflict_count": proof.conflict_count,
                "conflict_details": list(proof.conflict_details),
            },
        }

        base_tags = cfg.base_tags or []
        tags = list(base_tags)
        tags.append("segment:" + segment_key.label())

        new_rule = RuleDefinition(
            rule_id=candidate.rule_id,
            text=candidate.nl_rule_text,
            description=candidate.description,
            enabled=True,
            severity=cfg.default_severity,
            tags=tags,
            metadata=metadata,
        )
        new_rules.append(new_rule)

    new_version = bump_semver(base_ruleset.version, part=cfg.bump_part)
    ruleset_id = new_ruleset_id or base_ruleset.ruleset_id

    description = (
        base_ruleset.description
        or f"Base ruleset {base_ruleset.ruleset_id}."
    )
    description += " Extended with promoted candidates from Rule Gaps."

    return RuleSet(
        ruleset_id=ruleset_id,
        version=new_version,
        description=description,
        rules=new_rules,
    )


def save_ruleset_to_yaml(
    ruleset: RuleSet,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Zapisuje RuleSet do pliku YAML (lub JSON, jeśli rozszerzenie to .json).

    Struktura pliku jest kompatybilna z rules_io.load_ruleset_from_file().
    """
    output_path = Path(path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Ruleset file already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    data = {
        "ruleset_id": ruleset.ruleset_id,
        "version": ruleset.version,
        "description": ruleset.description,
        "rules": [
            {
                "id": rule.rule_id,
                "text": rule.text,
                "description": rule.description,
                "enabled": rule.enabled,
                "severity": rule.severity,
                "tags": list(rule.tags),
                "metadata": dict(rule.metadata),
            }
            for rule in ruleset.rules
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".json":
        import json

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        return output_path

    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed, cannot write YAML file. "
            "Install it with: pip install pyyaml"
        )

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return output_path


def load_base_ruleset(path: str | Path) -> RuleSet:
    """Prosty wrapper do wczytywania RuleSet z pliku.

    Użyteczne w notebookach, żeby mieć jedno miejsce na import.
    """
    return load_ruleset_from_file(Path(path))
