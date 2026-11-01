"""Part A: clean the raw export, build the modelling table and the first delay evidence."""
import numpy as np
import pandas as pd

from src.shipment_prep import (
    FEATURES, WEIGHT_SENTINEL_LIMIT, build_features, clean_shipments, metric_table, write_csv,
)

DISTANCE_BINS = [0, 200, 400, 600, 2000]
DISTANCE_LABELS = ["0-200", "201-400", "401-600", "601+"]


def main():
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # "Before" counts: after removing duplicates, before any filling.
    dedup = raw.drop_duplicates()
    weight = dedup["parcel_weight_kg"]
    before = {
        "raw_rows": len(raw),
        "duplicate_rows_removed": len(raw) - len(dedup),
        "rows_after_dedup": len(dedup),
        "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
        "missing_parcel_weight_before": int(weight.isna().sum()),
        "sentinel_weight_rows": int((weight > WEIGHT_SENTINEL_LIMIT).sum()),
        "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
        "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
        "service_type_spellings_before": int(dedup["service_type"].nunique()),
        "service_type_spellings_after": int(dedup["service_type"].str.strip().str.title().nunique()),
    }

    clean, medians = clean_shipments(raw)
    table = build_features(clean, hubs)
    table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)
    write_csv(table, "data/modelling_table.csv")

    hours = table["delivery_hours"]
    q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    checks = {
        **before,
        "median_distance_km": medians["distance_km"],
        "median_parcel_weight_kg": medians["parcel_weight_kg"],
        "median_declared_value_inr": medians["declared_value_inr"],
        "rows_after_hub_merge": len(table),
        "nulls_in_features_and_target": int(table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        "delayed_count": int(table["is_delayed"].sum()),
        "delay_rate": table["is_delayed"].mean(),
        "hours_q1": q1,
        "hours_q3": q3,
        "hours_lower_bound": lower,
        "hours_upper_bound": upper,
        "hours_outlier_count": int(((hours < lower) | (hours > upper)).sum()),
        "declared_value_mean_inr": table["declared_value_inr"].mean(),
        "declared_value_median_inr": table["declared_value_inr"].median(),
        "log_declared_value_mean": table["log_declared_value"].mean(),
        "log_declared_value_median": table["log_declared_value"].median(),
    }
    write_csv(metric_table(checks), "outputs/part_a_checks.csv")

    table["distance_band"] = pd.cut(table["distance_km"], bins=DISTANCE_BINS,
                                    labels=DISTANCE_LABELS, right=True)
    groups = []
    for dim in ["service_type", "hub_tier", "distance_band"]:
        g = (table.groupby(dim, observed=True)
             .agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                  delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
             .reset_index().rename(columns={dim: "group"}))
        g.insert(0, "dimension", dim)
        g["group"] = g["group"].astype(str)
        groups.append(g)
    write_csv(pd.concat(groups, ignore_index=True), "outputs/part_a_delay_by_group.csv")

    corr = table.drop(columns=["distance_band"]).select_dtypes(include=np.number).corr()
    corr.index.name = "column"
    write_csv(corr.reset_index(), "outputs/part_a_correlations.csv")

    print(metric_table(checks).to_string(index=False))
    print(pd.concat(groups, ignore_index=True).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
