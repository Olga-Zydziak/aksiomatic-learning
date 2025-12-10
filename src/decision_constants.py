from __future__ import annotations

from typing import Final

from axiomatic_kernel import DecisionOutcome, DecisionStatus  # type: ignore[import]

# Statusy rozwiązywania problemu decyzyjnego przez kernel.
STATUS_SAT: Final[DecisionStatus] = "SAT"
STATUS_UNSAT: Final[DecisionStatus] = "UNSAT"
STATUS_ERROR: Final[DecisionStatus] = "ERROR"
STATUS_UNKNOWN: Final[DecisionStatus] = "UNKNOWN"

# Wyniki decyzji biznesowej.
OUTCOME_FLAGGED: Final[DecisionOutcome] = "FLAGGED"
OUTCOME_CLEAN: Final[DecisionOutcome] = "CLEAN"
OUTCOME_ERROR: Final[DecisionOutcome] = "ERROR"
OUTCOME_UNKNOWN: Final[DecisionOutcome] = "UNKNOWN"

__all__ = [
    "DecisionStatus",
    "DecisionOutcome",
    "STATUS_SAT",
    "STATUS_UNSAT",
    "STATUS_ERROR",
    "STATUS_UNKNOWN",
    "OUTCOME_FLAGGED",
    "OUTCOME_CLEAN",
    "OUTCOME_ERROR",
    "OUTCOME_UNKNOWN",
]
