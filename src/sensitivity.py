"""
Sensitivity analysis of the Cotonou -> training-zone transfer.

Question: if the archetype assumptions behind the transfer are somewhat wrong, how much do the
predictions change?

Part A - Swap effect: what happens to the predicted risk (and to the alert decision) if a hub is
         attached to another training zone (any zone, and the runner-up by profile distance)?
Part B - Target perturbation: add random noise to each hub's target profile, re-run the optimal
         assignment many times, and measure (1) how often each hub keeps its zone, (2) the expected
         shift in predicted risk caused by the reassignment.

Run from the project root, after src/model.py:
    python src/sensitivity.py            # default noise sd = 0.15
    python src/sensitivity.py 0.25       # stronger noise

Output: models/sensitivity_results.json (also displayed in the app, Engineering tab).
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

FEATURES = ["Hour", "Day_of_Week", "Zone_ID", "Rain_mm"]
DIMS = [("demand", "demand_rank"), ("peak_share", "peak_share_rank"), ("night_share", "night_share_rank")]
ALERT_THRESHOLD = 50.0
RAIN_LEVELS = (0.0, 10.0)
N_SIM = 1000


def cost_matrix(profiles, targets):
    """Euclidean distance between each hub's target (n_hubs x 3) and each zone's observed ranks."""
    zone_ranks = profiles[[col for _, col in DIMS]].to_numpy(dtype=float)
    return np.sqrt(((targets[:, None, :] - zone_ranks[None, :, :]) ** 2).sum(axis=2))


def assign(cost):
    """Optimal one-to-one assignment. Returns, for each hub, the index of its zone."""
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(cost.shape[0], dtype=int)
    out[rows] = cols
    return out


def risk_grid(model, zone_ids):
    """Predicted risk (%) for every zone over all weekdays x hours x rain levels."""
    D, H, R = np.meshgrid(np.arange(7), np.arange(24), RAIN_LEVELS, indexing="ij")
    base = pd.DataFrame({"Hour": H.ravel(), "Day_of_Week": D.ravel(), "Rain_mm": R.ravel()})
    arrays = []
    for z in zone_ids:
        X = base.assign(Zone_ID=z)[FEATURES]
        arrays.append(model.predict_proba(X)[:, 1] * 100)
    return np.vstack(arrays)


def pairwise_effects(grid, threshold=ALERT_THRESHOLD):
    """Mean absolute risk difference and alert-flip rate between every pair of zones."""
    diff = np.abs(grid[:, None, :] - grid[None, :, :]).mean(axis=2)
    alert = grid >= threshold
    flips = (alert[:, None, :] != alert[None, :, :]).mean(axis=2)
    return diff, flips


def simulate(profiles, targets, base_idx, diff, sd, n_sim=N_SIM, seed=42):
    """
    Perturb the targets with Gaussian noise (sd), re-assign, and record per hub:
      stability  = share of simulations in which the hub keeps its baseline zone
      shift      = mean absolute risk change (points) caused by the reassignment
    """
    rng = np.random.default_rng(seed)
    n_hubs = targets.shape[0]
    same = np.zeros(n_hubs)
    shift = np.zeros(n_hubs)
    for _ in range(n_sim):
        noisy = np.clip(targets + rng.normal(0, sd, targets.shape), 0, 1)
        new_idx = assign(cost_matrix(profiles, noisy))
        same += (new_idx == base_idx)
        shift += diff[base_idx, new_idx]
    return same / n_sim, shift / n_sim


def verdict(stability, shift):
    if stability >= 0.8 and shift < 5:
        return "robust"
    if stability >= 0.5 or shift < 10:
        return "moderate"
    return "fragile"


def main():
    sd = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    needed = ["models/traffic_rf_model.pkl", "models/zone_profiles.csv", "models/hub_zone_mapping.json"]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing {missing}. Run `python src/model.py` first.")

    with open("models/traffic_rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    profiles = pd.read_csv("models/zone_profiles.csv", index_col=0)
    with open("models/hub_zone_mapping.json", "r", encoding="utf-8") as f:
        hubs = json.load(f)
    if any("target" not in h for h in hubs.values()):
        raise SystemExit("Targets missing in hub_zone_mapping.json: re-run `python src/model.py`.")

    zone_ids = list(profiles.index)
    hub_names = list(hubs.keys())
    targets = np.array([[hubs[n]["target"][k] for k, _ in DIMS] for n in hub_names], dtype=float)

    cost = cost_matrix(profiles, targets)
    base_idx = assign(cost)
    for name, idx in zip(hub_names, base_idx):
        if int(zone_ids[idx]) != int(hubs[name]["zone_id"]):
            print(f"Warning: recomputed zone for {name} differs from the mapping file.")

    grid = risk_grid(model, zone_ids)
    diff, flips = pairwise_effects(grid)
    stability, shift = simulate(profiles, targets, base_idx, diff, sd)

    rows = []
    for i, name in enumerate(hub_names):
        a = base_idx[i]
        others = [j for j in range(len(zone_ids)) if j != a]
        runner = min(others, key=lambda j: cost[i, j])
        rows.append({
            "hub": name,
            "assigned_zone": int(zone_ids[a]),
            "runner_up_zone": int(zone_ids[runner]),
            "avg_shift_any_zone_pts": round(float(diff[a, others].mean()), 2),
            "alert_flip_any_zone_pct": round(float(flips[a, others].mean()) * 100, 1),
            "shift_vs_runner_up_pts": round(float(diff[a, runner]), 2),
            "alert_flip_vs_runner_up_pct": round(float(flips[a, runner]) * 100, 1),
            "stability_pct": round(float(stability[i]) * 100, 1),
            "expected_shift_noise_pts": round(float(shift[i]), 2),
            "verdict": verdict(stability[i], shift[i]),
        })

    results = {"noise_sd": sd, "n_sim": N_SIM, "alert_threshold": ALERT_THRESHOLD,
               "rain_levels_mm": list(RAIN_LEVELS), "hubs": rows}
    os.makedirs("models", exist_ok=True)
    with open("models/sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Sensitivity of the Cotonou -> zone transfer (noise sd = {sd}, {N_SIM} simulations)\n")
    table = pd.DataFrame(rows).set_index("hub")
    print(table[["assigned_zone", "runner_up_zone", "stability_pct", "expected_shift_noise_pts",
                 "shift_vs_runner_up_pts", "alert_flip_vs_runner_up_pct", "verdict"]].to_string())
    print("\nReading: 'stability_pct' = how often the hub keeps its zone when the archetype targets are perturbed;"
          "\n'expected_shift_noise_pts' = average change in predicted risk (points) caused by such reassignments;"
          "\n'alert_flip_vs_runner_up_pct' = share of weekday-hour-rain cases where swapping to the runner-up zone"
          f"\nwould flip the alert decision at {ALERT_THRESHOLD:.0f}%.")


if __name__ == "__main__":
    main()