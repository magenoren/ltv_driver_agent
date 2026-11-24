"""Central configuration.

Two kinds of settings, kept apart on purpose:
- .env         -> environment-specific values and secrets (project ids, API keys). Never committed.
- *.yaml       -> analysis/generator parameters. Committed, so anyone can reproduce your results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name} in .env (copy .env.example to .env and fill it in).")
    return value


@dataclass(frozen=True)
class BigQuerySettings:
    project: str
    dataset: str
    location: str
    table_mmp: str
    table_firebase: str
    table_monetization: str

    def table_id(self, table: str) -> str:
        """Fully-qualified id, e.g. my-project.my_dataset.my_table"""
        return f"{self.project}.{self.dataset}.{table}"

    @property
    def mmp_id(self) -> str:
        return self.table_id(self.table_mmp)

    @property
    def firebase_id(self) -> str:
        return self.table_id(self.table_firebase)

    @property
    def monetization_id(self) -> str:
        return self.table_id(self.table_monetization)


def get_bigquery_settings() -> BigQuerySettings:
    return BigQuerySettings(
        project=_require("BIGQUERY_PROJECT"),
        dataset=_require("BIGQUERY_DATASET"),
        location=os.getenv("BIGQUERY_LOCATION", "US"),
        table_mmp=os.getenv("BQ_TABLE_MMP", "ltv_agent_mmp_cohorts"),
        table_firebase=os.getenv("BQ_TABLE_FIREBASE", "ltv_agent_firebase_cohorts"),
        table_monetization=os.getenv("BQ_TABLE_MONETIZATION", "ltv_agent_monetization_daily"),
    )


def load_yaml(filename: str) -> dict:
    with open(PROJECT_ROOT / filename, encoding="utf-8") as f:
        return yaml.safe_load(f)
