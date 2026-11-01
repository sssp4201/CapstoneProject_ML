"""Part C: delay classifier, feature-signal check, tuning and the operations alert threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS, RANDOM_STATE, write_csv

TARGET = "is_delayed"
TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
SELECT_K = [5, 7, 10, 12, 17]
FINAL_K = 7
PARAM_GRID = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_RECALL = 0.80
# Tie-break order for the final configuration when validation F1 scores are equal.
TIE_ORDER = ["default_forest_same_7_features", "grid_search", "randomized_search"]


def stratified_split(X, y):
    """Part B's two-step 70/15/15 split, stratified on the target in both calls."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def classification_row(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def build_final_pipeline(n_estimators, max_depth, min_samples_leaf):
    return Pipeline([
        ("selector", SelectKBest(f_classif, k=FINAL_K)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=n_estimators,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf)),
    ])


def main():
    table = pd.read_csv("data/modelling_table.csv")
    X, y = table[FEATURES], table[TARGET]
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    print("sizes", len(X_train), len(X_val), len(X_test),
          "delayed", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))

    # Task 2: blind baseline and three classifiers at the default 0.5 rule.
    model_rows = [{"model": "blind_all_on_time", **classification_row(y_val, np.zeros(len(y_val), dtype=int))}]
    logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    model_rows.append({"model": "logistic_regression", **classification_row(y_val, logreg.predict(X_val))})

    best_tree = None
    for d in TREE_DEPTHS:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        row = classification_row(y_val, tree.predict(X_val))
        if best_tree is None or row["f1"] > best_tree[1]["f1"]:  # strict > keeps the smaller depth on ties
            best_tree = (d, row)
    model_rows.append({"model": f"decision_tree_depth_{best_tree[0]}", **best_tree[1]})

    rf_default = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train)
    model_rows.append({"model": "random_forest_default", **classification_row(y_val, rf_default.predict(X_val))})
    models_df = pd.DataFrame(model_rows)
    write_csv(models_df, "outputs/part_c_models.csv")

    # Task 3: which columns carry signal?
    selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
    scores = pd.DataFrame({"feature": FEATURES, "f_score": selector_all.scores_,
                           "rf_importance": rf_default.feature_importances_})
    scores["f_rank"] = scores["f_score"].rank(ascending=False, method="first").astype(int)
    scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
    scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
    write_csv(scores, "outputs/part_c_feature_scores.csv")

    selectk_rows = []
    for k in SELECT_K:
        pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                         ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))])
        pipe.fit(X_train, y_train)
        selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
    write_csv(pd.DataFrame(selectk_rows), "outputs/part_c_selectk.csv")

    # Task 4: tuning on the 7 selected columns; selector fitted on training rows only.
    selector = SelectKBest(f_classif, k=FINAL_K).fit(X_train, y_train)
    Xs_train, Xs_val = selector.transform(X_train), selector.transform(X_val)

    grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID, cv=5)
    grid.fit(Xs_train, y_train)
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), PARAM_GRID,
                              n_iter=9, cv=5, random_state=RANDOM_STATE)
    rand.fit(Xs_train, y_train)
    default7 = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(Xs_train, y_train)

    default_params = {"max_depth": None, "min_samples_leaf": 1, "n_estimators": 100}
    tuning = pd.DataFrame([
        {"search": "grid_search", "candidates": len(grid.cv_results_["params"]), "cv_folds": 5,
         "best_params": grid.best_params_, "best_cv_accuracy": grid.best_score_,
         "val_f1": f1_score(y_val, grid.predict(Xs_val))},
        {"search": "randomized_search", "candidates": len(rand.cv_results_["params"]), "cv_folds": 5,
         "best_params": rand.best_params_, "best_cv_accuracy": rand.best_score_,
         "val_f1": f1_score(y_val, rand.predict(Xs_val))},
        {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
         "best_params": default_params, "best_cv_accuracy": np.nan,
         "val_f1": f1_score(y_val, default7.predict(Xs_val))},
    ])
    tuning["total_fits"] = np.where(tuning["cv_folds"] > 0, tuning["candidates"] * tuning["cv_folds"], 1)
    tuning["best_params"] = tuning["best_params"].map(lambda p: str(dict(sorted(p.items()))))
    tuning = tuning[["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]]
    write_csv(tuning, "outputs/part_c_tuning.csv")

    # Task 5: highest val_f1, ties resolved by TIE_ORDER (compared on the rounded value written to file).
    ranked = tuning.assign(f1_4=tuning["val_f1"].round(4), tie=tuning["search"].map(TIE_ORDER.index))
    chosen_from = ranked.sort_values(["f1_4", "tie"], ascending=[False, True]).iloc[0]["search"]
    chosen_params = {"grid_search": grid.best_params_, "randomized_search": rand.best_params_,
                     "default_forest_same_7_features": default_params}[chosen_from]
    final = build_final_pipeline(**chosen_params).fit(X_train, y_train)

    # Task 6: alert threshold from validation probabilities.
    val_proba = final.predict_proba(X_val)[:, 1]
    threshold_rows = [{"threshold": t, **classification_row(y_val, (val_proba >= t).astype(int))}
                      for t in THRESHOLDS]
    thresholds = pd.DataFrame(threshold_rows)
    write_csv(thresholds, "outputs/part_c_thresholds.csv")
    passing = thresholds[thresholds["recall"] >= MIN_RECALL]
    threshold = float(passing["threshold"].max()) if len(passing) else 0.1

    # Task 7: the one and only test evaluation.
    test_proba = final.predict_proba(X_test)[:, 1]
    final_test = pd.DataFrame([{"threshold": threshold,
                                **classification_row(y_test, (test_proba >= threshold).astype(int))}])
    write_csv(final_test, "outputs/part_c_final_test.csv")
    max_depth = chosen_params["max_depth"]
    config = pd.DataFrame([{"selector_k": FINAL_K, "n_estimators": chosen_params["n_estimators"],
                            "max_depth": "unlimited" if max_depth is None else max_depth,
                            "min_samples_leaf": chosen_params["min_samples_leaf"],
                            "threshold": threshold, "chosen_from": chosen_from}])
    write_csv(config, "outputs/part_c_final_config.csv")

    for name, df in [("models", models_df), ("feature scores", scores), ("select k", pd.DataFrame(selectk_rows)),
                     ("tuning", tuning), ("thresholds", thresholds), ("final test", final_test),
                     ("config", config)]:
        print(f"\n== {name} ==")
        print(df.round(4).to_string(index=False))
    noise_share = scores.loc[scores["is_noise"], "rf_importance"].sum()
    print(f"\nnoise share of forest importance: {noise_share:.4f}")


if __name__ == "__main__":
    main()
