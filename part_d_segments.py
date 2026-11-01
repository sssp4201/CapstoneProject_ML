"""Part D (segments): hub profile, K-Means operating segments and the tier-label check."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import RANDOM_STATE, write_csv

PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
N_SEGMENTS = 3  # fixed by the business: operations runs exactly three playbooks
SEPARATION_K = [2, 3, 4, 5]
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]  # ordered by ascending mean delay_rate
TIER_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def fit_kmeans(matrix, k):
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(matrix)


def across_over_within(matrix, labels):
    """Mean distance of cross-cluster hub pairs divided by mean distance of same-cluster pairs."""
    across, within = [], []
    for i, j in combinations(range(len(matrix)), 2):
        dist = np.linalg.norm(matrix[i] - matrix[j])
        (within if labels[i] == labels[j] else across).append(dist)
    return float(np.mean(across) / np.mean(within))


def main():
    table = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    profile = (table.groupby("hub_id")
               .agg(shipments=("shipment_id", "size"), mean_delivery_hours=("delivery_hours", "mean"),
                    delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                    mean_n_stops=("n_stops", "mean"))
               .reset_index().sort_values("hub_id").reset_index(drop=True))
    write_csv(profile, "outputs/hub_profile.csv")

    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])

    separation = []
    for k in SEPARATION_K:
        labels = fit_kmeans(scaled, k)
        sizes = sorted(np.bincount(labels), reverse=True)
        separation.append({"K": k, "across_over_within": across_over_within(scaled, labels),
                           "cluster_sizes": "-".join(str(s) for s in sizes)})
    write_csv(pd.DataFrame(separation), "outputs/part_d_separation.csv")

    profile["cluster"] = fit_kmeans(scaled, N_SEGMENTS)
    order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
    profile["segment"] = profile["cluster"].map(dict(zip(order, SEGMENT_NAMES)))

    segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
    write_csv(segments, "outputs/hub_segments.csv")

    seg_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "size"), **{c: (c, "mean") for c in PROFILE_COLUMNS})
                   .reindex(SEGMENT_NAMES).reset_index())
    write_csv(seg_profile, "outputs/segment_profile.csv")

    mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_SEGMENT)]
    write_csv(mismatch, "outputs/tier_vs_segment_mismatch.csv")

    print(profile.round(4).to_string(index=False))
    print(pd.DataFrame(separation).round(4).to_string(index=False))
    print(seg_profile.round(4).to_string(index=False))
    print(mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
