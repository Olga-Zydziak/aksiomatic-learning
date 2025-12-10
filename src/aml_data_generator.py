from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import csv
import random
import datetime as dt


@dataclass
class AMLScenarioWeights:
    """Prawdopodobieństwa wyboru poszczególnych scenariuszy AML.

    Wartości nie muszą sumować się idealnie do 1.0 – zostaną znormalizowane.
    """

    clean: float = 0.5
    high_amount: float = 0.2
    velocity: float = 0.15
    pep_high_risk: float = 0.1
    structuring: float = 0.03
    noise: float = 0.02


@dataclass
class DatasetSpec:
    """Specyfikacja pojedynczego zbioru danych AML.

    Attributes:
        name: Nazwa zbioru (np. 'train', 'dev', 'test').
        path: Ścieżka do pliku CSV.
        n_records: Liczba rekordów do wygenerowania.
    """

    name: str
    path: Path
    n_records: int


class AMLDataGenerator:
    """Generator syntetycznych danych transakcyjnych w stylu AML.

    Dane są projektowane tak, aby przypominały realne feedy AML,
    ale pozostają w pełni syntetyczne.
    """

    def __init__(
        self,
        *,
        base_country: str = "PL",
        currencies: List[str] | None = None,
        seed: int = 42,
        scenario_weights: AMLScenarioWeights | None = None,
    ) -> None:
        self._base_country = base_country
        self._rng = random.Random(seed)
        self._currencies = currencies or ["PLN", "EUR", "USD"]
        self._scenario_weights = scenario_weights or AMLScenarioWeights()
        self._scenario_thresholds = self._build_scenario_thresholds(
            self._scenario_weights
        )

    @staticmethod
    def _build_scenario_thresholds(
        weights: AMLScenarioWeights,
    ) -> Dict[str, float]:
        raw = {
            "clean": weights.clean,
            "high_amount": weights.high_amount,
            "velocity": weights.velocity,
            "pep_high_risk": weights.pep_high_risk,
            "structuring": weights.structuring,
            "noise": weights.noise,
        }
        total = sum(v for v in raw.values() if v > 0.0) or 1.0
        thresholds: Dict[str, float] = {}
        acc = 0.0
        for key, value in raw.items():
            if value <= 0.0:
                continue
            acc += value / total
            thresholds[key] = acc
        # bezpieczeństwo numeryczne – ostatni próg = 1.0
        thresholds[list(thresholds.keys())[-1]] = 1.0
        return thresholds

    def _sample_scenario(self) -> str:
        r = self._rng.random()
        for scenario, thr in self._scenario_thresholds.items():
            if r <= thr:
                return scenario
        return "clean"

    def generate_csv(self, path: Path | str, n_records: int) -> Path:
        """Generuje plik CSV z n_records syntetycznych transakcji AML.

        Zwraca ścieżkę do wygenerowanego pliku.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "transaction_id",
            "timestamp",
            "customer_id",
            "account_id",
            "amount",
            "currency",
            "transaction_type",
            "channel",
            "country_of_residence",
            "counterparty_account_id",
            "counterparty_country",
            "customer_segment",
            "kyc_risk_level",
            "is_pep",
            "on_sanctions_list",
            "model_risk_score",
            "tx_count_24h",
            "total_amount_24h",
            "tx_count_7d",
            "total_amount_7d",
            "unique_counterparties_30d",
        ]

        with out_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for index in range(1, n_records + 1):
                scenario = self._sample_scenario()
                record = self._generate_record(index=index, scenario=scenario)
                writer.writerow(record)

        return out_path

    def _generate_record(self, *, index: int, scenario: str) -> Dict[str, object]:
        now = dt.datetime.utcnow()
        # Rozsiewamy timestampy +- 30 dni wokół "teraz"
        delta_days = self._rng.randint(-30, 0)
        delta_seconds = self._rng.randint(0, 24 * 3600 - 1)
        ts = now + dt.timedelta(days=delta_days, seconds=delta_seconds)

        base: Dict[str, object] = {
            "transaction_id": f"T{index:09d}",
            "timestamp": ts.replace(microsecond=0).isoformat() + "Z",
            "customer_id": f"C{self._rng.randint(1, max(1, index // 3 + 1)):06d}",
            "account_id": f"AC{self._rng.randint(1, max(1, index // 3 + 1)):08d}",
            "currency": self._rng.choice(self._currencies),
            "country_of_residence": self._base_country,
            "counterparty_account_id": (
                f"CP{self._rng.randint(1, max(1, index * 2)):08d}"
            ),
            "counterparty_country": self._rng.choice(
                [self._base_country, "DE", "NL", "GB", "UA", "US", "AE"]
            ),
        }

        if scenario == "clean":
            record = self._gen_clean(base)
        elif scenario == "high_amount":
            record = self._gen_high_amount(base)
        elif scenario == "velocity":
            record = self._gen_velocity(base)
        elif scenario == "pep_high_risk":
            record = self._gen_pep_high_risk(base)
        elif scenario == "structuring":
            record = self._gen_structuring(base)
        else:
            record = self._gen_noise(base)

        return record

    # --- Scenariusze -------------------------------------------------

    def _gen_clean(self, base: Dict[str, object]) -> Dict[str, object]:
        amount = self._rng.randint(50, 3_000)
        tx24 = self._rng.randint(0, 3)
        tot24 = amount * max(1, tx24)
        tx7 = tx24 + self._rng.randint(0, 5)
        tot7 = int(tot24 * self._rng.uniform(1.0, 3.0))
        uniq30 = self._rng.randint(1, 5)

        record = {
            **base,
            "amount": amount,
            "transaction_type": self._rng.choice(
                ["CARD_POS", "SEPA_CREDIT", "INTERNAL"]
            ),
            "channel": self._rng.choice(["MOBILE", "INTERNET", "POS"]),
            "customer_segment": self._rng.choice(
                ["RETAIL_STANDARD", "RETAIL_PREMIUM"]
            ),
            "kyc_risk_level": self._rng.choice(["LOW", "MEDIUM"]),
            "is_pep": False,
            "on_sanctions_list": False,
            "model_risk_score": round(self._rng.uniform(0.01, 0.25), 2),
            "tx_count_24h": tx24,
            "total_amount_24h": tot24,
            "tx_count_7d": tx7,
            "total_amount_7d": tot7,
            "unique_counterparties_30d": uniq30,
        }
        return record

    def _gen_high_amount(self, base: Dict[str, object]) -> Dict[str, object]:
        amount = self._rng.randint(15_000, 120_000)
        tx24 = self._rng.randint(0, 3)
        tot24 = int(amount * self._rng.uniform(1.0, 2.5))
        tx7 = tx24 + self._rng.randint(0, 4)
        tot7 = int(tot24 * self._rng.uniform(1.5, 3.0))
        uniq30 = self._rng.randint(1, 8)

        record = {
            **base,
            "amount": amount,
            "transaction_type": self._rng.choice(
                ["SWIFT_OUT", "SWIFT_IN", "SEPA_CREDIT"]
            ),
            "channel": self._rng.choice(["INTERNET", "BRANCH"]),
            "customer_segment": self._rng.choice(["SME", "RETAIL_PREMIUM"]),
            "kyc_risk_level": self._rng.choice(["MEDIUM", "HIGH"]),
            "is_pep": self._rng.random() < 0.05,
            "on_sanctions_list": False,
            "model_risk_score": round(self._rng.uniform(0.4, 0.95), 2),
            "tx_count_24h": tx24,
            "total_amount_24h": tot24,
            "tx_count_7d": tx7,
            "total_amount_7d": tot7,
            "unique_counterparties_30d": uniq30,
        }
        return record

    def _gen_velocity(self, base: Dict[str, object]) -> Dict[str, object]:
        amount = self._rng.randint(500, 7_000)
        tx24 = self._rng.randint(6, 40)
        tot24 = int(amount * tx24 * self._rng.uniform(0.8, 1.2))
        tx7 = tx24 + self._rng.randint(10, 60)
        tot7 = int(tot24 * self._rng.uniform(2.0, 5.0))
        uniq30 = self._rng.randint(5, 25)

        record = {
            **base,
            "amount": amount,
            "transaction_type": self._rng.choice(
                ["INTERNAL", "SEPA_CREDIT", "CARD_POS"]
            ),
            "channel": self._rng.choice(["MOBILE", "INTERNET"]),
            "customer_segment": self._rng.choice(
                ["RETAIL_STANDARD", "RETAIL_PREMIUM"]
            ),
            "kyc_risk_level": self._rng.choice(["MEDIUM", "HIGH"]),
            "is_pep": False,
            "on_sanctions_list": False,
            "model_risk_score": round(self._rng.uniform(0.3, 0.9), 2),
            "tx_count_24h": tx24,
            "total_amount_24h": tot24,
            "tx_count_7d": tx7,
            "total_amount_7d": tot7,
            "unique_counterparties_30d": uniq30,
        }
        return record

    def _gen_pep_high_risk(self, base: Dict[str, object]) -> Dict[str, object]:
        amount = self._rng.randint(2_000, 50_000)
        tx24 = self._rng.randint(0, 5)
        tot24 = int(amount * self._rng.uniform(1.0, 2.0))
        tx7 = tx24 + self._rng.randint(0, 10)
        tot7 = int(tot24 * self._rng.uniform(1.5, 4.0))
        uniq30 = self._rng.randint(3, 15)

        record = {
            **base,
            "amount": amount,
            "transaction_type": self._rng.choice(["SWIFT_OUT", "SEPA_CREDIT"]),
            "channel": self._rng.choice(["INTERNET", "BRANCH"]),
            "customer_segment": self._rng.choice(["RETAIL_PREMIUM", "HNWI"]),
            "kyc_risk_level": self._rng.choice(["HIGH"]),
            "is_pep": True,
            "on_sanctions_list": False,
            "model_risk_score": round(self._rng.uniform(0.6, 0.98), 2),
            "tx_count_24h": tx24,
            "total_amount_24h": tot24,
            "tx_count_7d": tx7,
            "total_amount_7d": tot7,
            "unique_counterparties_30d": uniq30,
        }
        return record

    def _gen_structuring(self, base: Dict[str, object]) -> Dict[str, object]:
        amount = self._rng.randint(200, 1_900)
        tx24 = self._rng.randint(10, 40)
        tot24 = int(amount * tx24 * self._rng.uniform(0.9, 1.3))
        tx7 = tx24 + self._rng.randint(20, 80)
        tot7 = int(tot24 * self._rng.uniform(2.0, 6.0))
        uniq30 = self._rng.randint(1, 10)

        record = {
            **base,
            "amount": amount,
            "transaction_type": self._rng.choice(
                ["CASH_IN", "CASH_OUT", "INTERNAL"]
            ),
            "channel": self._rng.choice(["BRANCH", "ATM"]),
            "customer_segment": self._rng.choice(
                ["RETAIL_STANDARD", "SME"]
            ),
            "kyc_risk_level": self._rng.choice(["MEDIUM", "HIGH"]),
            "is_pep": False,
            "on_sanctions_list": False,
            "model_risk_score": round(self._rng.uniform(0.5, 0.97), 2),
            "tx_count_24h": tx24,
            "total_amount_24h": tot24,
            "tx_count_7d": tx7,
            "total_amount_7d": tot7,
            "unique_counterparties_30d": uniq30,
        }
        return record

    def _gen_noise(self, base: Dict[str, object]) -> Dict[str, object]:
        amount = self._rng.randint(10, 150_000)
        tx24 = self._rng.randint(0, 50)
        tot24 = int(amount * max(1, tx24) * self._rng.uniform(0.5, 1.5))
        tx7 = tx24 + self._rng.randint(0, 100)
        tot7 = int(tot24 * self._rng.uniform(1.0, 5.0))
        uniq30 = self._rng.randint(1, 40)

        record = {
            **base,
            "amount": amount,
            "transaction_type": self._rng.choice(
                [
                    "CARD_POS",
                    "CARD_ATM",
                    "INTERNAL",
                    "SEPA_CREDIT",
                    "SWIFT_OUT",
                    "CASH_IN",
                    "CASH_OUT",
                ]
            ),
            "channel": self._rng.choice(
                ["MOBILE", "INTERNET", "ATM", "BRANCH"]
            ),
            "customer_segment": self._rng.choice(
                ["RETAIL_STANDARD", "RETAIL_PREMIUM", "SME"]
            ),
            "kyc_risk_level": self._rng.choice(["LOW", "MEDIUM", "HIGH"]),
            "is_pep": self._rng.random() < 0.03,
            "on_sanctions_list": self._rng.random() < 0.01,
            "model_risk_score": round(self._rng.uniform(0.01, 0.99), 2),
            "tx_count_24h": tx24,
            "total_amount_24h": tot24,
            "tx_count_7d": tx7,
            "total_amount_7d": tot7,
            "unique_counterparties_30d": uniq30,
        }
        return record


# --- Funkcje pomocnicze – interfejs pod notebooka --------------------


def generate_transactions_csv(
    spec: DatasetSpec,
    *,
    generator: AMLDataGenerator | None = None,
) -> Path:
    """Generuje pojedynczy zbiór danych na podstawie DatasetSpec."""
    gen = generator or AMLDataGenerator()
    return gen.generate_csv(path=spec.path, n_records=spec.n_records)


def generate_default_datasets(
    base_dir: Path | str,
    *,
    generator: AMLDataGenerator | None = None,
) -> Dict[str, Path]:
    """Generuje domyślne zbiory train/dev/test pod demo AML."""
    gen = generator or AMLDataGenerator()
    base_dir = Path(base_dir)

    specs = {
        "train": DatasetSpec(
            name="train",
            path=base_dir / "transactions_train.csv",
            n_records=5_000,
        ),
        "dev": DatasetSpec(
            name="dev",
            path=base_dir / "transactions_dev.csv",
            n_records=1_000,
        ),
        "test": DatasetSpec(
            name="test",
            path=base_dir / "transactions_test.csv",
            n_records=1_000,
        ),
    }

    result: Dict[str, Path] = {}
    for ds in specs.values():
        path = gen.generate_csv(path=ds.path, n_records=ds.n_records)
        result[ds.name] = path
    return result


def write_demo_transactions_csv(
    path: Path | str,
    n_records: int = 200,
    *,
    seed: int = 42,
) -> Path:
    """Szybki helper do wygenerowania jednego pliku demo z transakcjami."""
    gen = AMLDataGenerator(seed=seed)
    return gen.generate_csv(path=path, n_records=n_records)
