"""Score new parcels: delay probability, alert flag and the action operations should take."""
import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments, write_csv

ESCALATE_SEGMENT = "Escalate"


def choose_action(flagged, segment):
    if not flagged:
        return "No action"
    if segment == ESCALATE_SEGMENT:
        return "Escalate to hub manager"
    return "Proactive customer call"


def main():
    new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    impute = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute["column"], impute["median"]))
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

    clean, _ = clean_shipments(new, medians=medians)
    features = build_features(clean, hubs)

    out = features[["shipment_id", "hub_id"]].copy()
    out["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
    out["flagged"] = (out["delay_probability"] >= threshold).astype(int)
    out = out.merge(segments, on="hub_id", how="left", validate="many_to_one")  # left merge keeps file order
    out["action"] = [choose_action(f, s) for f, s in zip(out["flagged"], out["segment"])]
    out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    write_csv(out, "outputs/new_shipment_actions.csv")

    print(out.to_string(index=False))
    print(f"\nflagged {out['flagged'].sum()} of {len(out)}; "
          f"escalations {(out['action'] == 'Escalate to hub manager').sum()}")


if __name__ == "__main__":
    main()
