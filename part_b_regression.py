"""Part B: delivery-hours regression with a leakage-safe split and model comparison."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES, RANDOM_STATE, metric_table, write_csv

TARGET = "delivery_hours"
TREE_DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_row(name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": name, "split": split, "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def main():
    table = pd.read_csv("data/modelling_table.csv")
    X, y = table[FEATURES], table[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    # Scaler fitted on training rows only; test rows are only transformed.
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    checks = {
        "train_rows": len(X_train), "val_rows": len(X_val), "test_rows": len(X_test),
        "train_scaled_max_abs_mean": float(np.abs(train_scaled.mean(axis=0)).max()),
        "test_scaled_max_abs_mean": float(np.abs(test_scaled.mean(axis=0)).max()),
    }
    # Keep the tiny train mean (~1e-16) visible instead of rounding it to 0.
    check_df = metric_table(checks)
    check_df.loc[check_df["metric"] == "train_scaled_max_abs_mean", "value"] = (
        f"{checks['train_scaled_max_abs_mean']:.4e}")
    check_df.to_csv("outputs/part_b_checks.csv", index=False)

    rows = [score_row("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

    depth_rows = []
    for d in TREE_DEPTHS:
        tree = DecisionTreeRegressor(max_depth=d, random_state=RANDOM_STATE).fit(X_train, y_train)
        depth_rows.append({"max_depth": "unlimited" if d is None else d,
                           "train_rmse": rmse(y_train, tree.predict(X_train)),
                           "val_rmse": rmse(y_val, tree.predict(X_val)),
                           "_depth": d})
    depth_df = pd.DataFrame(depth_rows)
    # idxmin returns the first minimum, so ties go to the smaller depth (list is ascending).
    best_depth = depth_df.loc[depth_df["val_rmse"].idxmin(), "_depth"]
    best_depth = None if pd.isna(best_depth) else int(best_depth)
    write_csv(depth_df.drop(columns="_depth"), "outputs/part_b_tree_depth.csv")

    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                       random_state=RANDOM_STATE),
    }
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        rows.append(score_row(name, "val", y_val, model.predict(X_val)))

    lr_rows = []
    for lr in LEARNING_RATES:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=RANDOM_STATE)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_val)
        lr_rows.append({"learning_rate": lr, "val_rmse": rmse(y_val, pred), "val_r2": r2_score(y_val, pred)})
    write_csv(pd.DataFrame(lr_rows), "outputs/part_b_gb_learning_rate.csv")

    ranking = (pd.DataFrame(rows[1:]).sort_values("RMSE", kind="stable")["model"].tolist())
    X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name in ranking[:2]:
        scores = cross_val_score(candidates[name], X_trval, y_trval, cv=kfold)
        cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
    write_csv(pd.DataFrame(cv_rows), "outputs/part_b_cv.csv")

    # Final model: lowest validation RMSE, already fitted on training rows, scored once on test.
    final_name = ranking[0]
    rows.append(score_row(final_name, "test", y_test, candidates[final_name].predict(X_test)))
    metrics = pd.DataFrame(rows)
    write_csv(metrics, "outputs/part_b_metrics.csv")

    print(check_df.to_string(index=False))
    print(depth_df.drop(columns="_depth").round(4).to_string(index=False))
    print(metrics.round(4).to_string(index=False))
    print(pd.DataFrame(lr_rows).round(4).to_string(index=False))
    print(pd.DataFrame(cv_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
