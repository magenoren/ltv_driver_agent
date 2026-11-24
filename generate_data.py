"""Generate the synthetic tables, save them locally, and upload them to BigQuery.

Usage (from the project root, with the virtual environment active):
    python generate_data.py --no-upload     # generate + save to data/synthetic/ only
    python generate_data.py                 # generate + save + upload to BigQuery + sanity check
"""
from __future__ import annotations

import argparse

from src.data.synthetic import generate
from src.settings import PROJECT_ROOT, load_yaml


def print_summary(tables: dict) -> None:
    mmp = tables["mmp"]
    weekly = mmp.groupby(["country", "platform", "date_week"])[["installs", "cohort_rev_d1"]].sum()
    weekly["ltv_d1"] = weekly["cohort_rev_d1"] / weekly["installs"]
    last_two = weekly.groupby(level=[0, 1]).tail(2)["ltv_d1"].unstack("date_week")
    last_two["wow_%"] = (last_two.iloc[:, 1] / last_two.iloc[:, 0] - 1) * 100
    print("\nMMP LTVD1, last two weeks (planted scenarios are in synthetic.yaml):")
    print(last_two.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic KPI tables.")
    parser.add_argument("--no-upload", action="store_true", help="skip the BigQuery upload")
    args = parser.parse_args()

    tables = generate(load_yaml("synthetic.yaml"))

    out_dir = PROJECT_ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)  # CSV so you can open them in Excel
        print(f"{name:13s} {len(df):>7,} rows -> {out_dir / (name + '.csv')}")
    print_summary(tables)

    if args.no_upload:
        return

    # Imported here so --no-upload works even before google-cloud-bigquery is set up.
    from src.data.bigquery import SCHEMAS, ensure_dataset, get_client, sanity_check, upload_dataframe
    from src.settings import get_bigquery_settings

    settings = get_bigquery_settings()
    client = get_client(settings)
    ensure_dataset(client, settings)
    targets = {"mmp": settings.mmp_id, "firebase": settings.firebase_id,
               "monetization": settings.monetization_id}
    print()
    for name, table_id in targets.items():
        n = upload_dataframe(client, tables[name], table_id, SCHEMAS[name])
        print(f"uploaded {n:>7,} rows -> {table_id}")

    print("\nSanity check from BigQuery (US iOS, last 3 weeks):")
    print(sanity_check(client, settings).to_string(index=False))


if __name__ == "__main__":
    main()
