"""Shared constants and preprocessing for the Delay Intelligence pipeline.

Used by every Part (training) and by predict_delay.py (scoring), so the same
cleaning and feature logic runs in both places.
"""
import os

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # network accepts parcels up to 40 kg; anything above 100 is a scanner fault
RANDOM_STATE = 42


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians)."""
    df = raw.drop_duplicates().copy()
    df["service_type"] = df["service_type"].str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype("string").str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).astype(float)
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan

    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])
    return df, medians


def build_features(clean, hubs):
    """Attach hub columns and derive model features. Never touches delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df


def round4(df):
    """Round every float column to 4 decimals."""
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(4)
    return out


def write_csv(df, path, index=False):
    """Write a CSV with decimals rounded to 4 places."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    round4(df).to_csv(path, index=index)


def metric_table(metrics):
    """Turn an ordered dict of metric -> value into a (metric, value) frame."""
    rows = []
    for name, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 4)
        elif isinstance(value, (np.integer,)):
            value = int(value)
        rows.append({"metric": name, "value": value})
    # object dtype keeps counts as integers instead of upcasting them to floats
    return pd.DataFrame(rows, dtype=object)
