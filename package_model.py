"""Part D (packaging): refit the Part C final pipeline and save it with its scoring artefacts."""
import os

import joblib
import numpy as np
import pandas as pd

from part_c_classification import TARGET, build_final_pipeline, stratified_split
from src.shipment_prep import FEATURES, clean_shipments, metric_table, write_csv

MODEL_DIR = "models"


def main():
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    threshold = float(config["threshold"])

    table = pd.read_csv("data/modelling_table.csv")
    X_train, _, X_test, y_train, _, _ = stratified_split(table[FEATURES], table[TARGET])
    pipeline = build_final_pipeline(int(config["n_estimators"]), max_depth, int(config["min_samples_leaf"]))
    pipeline.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "delay_model.pkl")
    joblib.dump(pipeline, model_path)
    write_csv(pd.DataFrame({"threshold": [threshold]}), os.path.join(MODEL_DIR, "delay_threshold.csv"))
    write_csv(pd.DataFrame({"feature": FEATURES}), os.path.join(MODEL_DIR, "feature_order.csv"))
    _, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
    write_csv(pd.DataFrame({"column": list(medians), "median": list(medians.values())}),
              os.path.join(MODEL_DIR, "impute_values.csv"))

    reloaded = joblib.load(model_path)
    diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - pipeline.predict_proba(X_test)[:, 1]).max()
    check = metric_table({"test_rows_scored": len(X_test),
                          "reload_max_abs_probability_diff": float(diff), "threshold": threshold})
    write_csv(check, "outputs/part_d_reload_check.csv")
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
