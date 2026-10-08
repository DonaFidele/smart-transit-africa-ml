"""
Decide, with rules fixed IN ADVANCE, whether the forecast model should switch to the relative target.

The criteria below are written before looking at the results, so the decision cannot be tuned to
whatever looks best. The relative target ("surge above the zone's usual level") is adopted only if,
for BOTH horizons (1 h and 24 h), all three conditions hold:

  C1  Real skill      : AUC of the Random Forest (lags) on the relative target >= 0.65
  C2  Beats trivial   : lower bound of its F1 95% CI > F1 of "always alert" (no skill)
  C3  Zone-robust     : when zone labels are wrong, the F1 of the no-lags model drops LESS on the relative
                        target than on the absolute target (the transfer depends less on the matching)

If any condition fails, the absolute target is kept and the report says which one failed.

Run from the project root, after src/evaluate.py:
    python src/choose_target.py
Output: models/target_decision.json
"""
import json
import os

AUC_MIN = 0.65
RF_LAGS, RF_NOLAGS, NO_SKILL = "Random Forest (lags)", "Random Forest (no lags)", "Always alert (no skill)"


def _row(block, name):
    return next(r for r in block["models"] if r["model"] == name)


def decide_horizon(abs_block, rel_block):
    """Evaluate the three criteria for one horizon. Returns a dict with the values and the pass/fail flags."""
    rf = _row(rel_block, RF_LAGS)
    no_skill = _row(rel_block, NO_SKILL)
    auc = rf.get("auc")
    drop_rel = rel_block["zone_swap"][RF_NOLAGS]["drop"]
    drop_abs = abs_block["zone_swap"][RF_NOLAGS]["drop"]
    c1 = auc is not None and auc >= AUC_MIN
    c2 = rf["ci_low"] > no_skill["f1"]
    c3 = drop_rel > drop_abs  # less negative = less sensitive to a wrong zone
    return {
        "auc": auc, "f1_ci_low": rf["ci_low"], "no_skill_f1": no_skill["f1"],
        "zone_drop_relative": drop_rel, "zone_drop_absolute": drop_abs,
        "C1_skill": bool(c1), "C2_beats_trivial": bool(c2), "C3_zone_robust": bool(c3),
        "passes": bool(c1 and c2 and c3),
    }


def decide(evaluation):
    """Overall decision from the content of models/evaluation_results.json."""
    per_horizon = {}
    for h, hblock in evaluation["horizons"].items():
        per_horizon[h] = decide_horizon(hblock["targets"]["absolute"], hblock["targets"]["relative"])
    adopt = all(v["passes"] for v in per_horizon.values())
    return {"target": "relative" if adopt else "absolute", "horizons": per_horizon,
            "rules": {"AUC_MIN": AUC_MIN}}


def main():
    path = "models/evaluation_results.json"
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run `python src/evaluate.py` first.")
    with open(path, "r", encoding="utf-8") as f:
        evaluation = json.load(f)
    if "targets" not in next(iter(evaluation["horizons"].values())):
        raise SystemExit("Old evaluation format: re-run `python src/evaluate.py` with the current version.")

    result = decide(evaluation)
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    print("Decision rules (fixed in advance): relative target adopted only if C1, C2 and C3 pass at BOTH horizons.\n")
    for h, v in result["horizons"].items():
        print(f"Horizon {h} h")
        auc_txt = f"{v['auc']:.2f}" if v["auc"] is not None else "n/a"
        print(f"  C1 skill          AUC {auc_txt} (needs >= {AUC_MIN})                          -> {mark(v['C1_skill'])}")
        print(f"  C2 beats trivial  F1 CI lower bound {v['f1_ci_low']:.2f} vs always-alert {v['no_skill_f1']:.2f}      -> {mark(v['C2_beats_trivial'])}")
        print(f"  C3 zone-robust    F1 change if zone wrong: relative {v['zone_drop_relative']:+.2f} vs absolute "
              f"{v['zone_drop_absolute']:+.2f} -> {mark(v['C3_zone_robust'])}")
    print("\n" + "=" * 70)
    if result["target"] == "relative":
        print("DECISION: switch the forecast model to the RELATIVE target.")
        print("Next command:  python src/forecast_model.py --target relative")
    else:
        print("DECISION: KEEP the absolute target (the relative target did not meet all criteria).")
        print("Report the relative-target result as a documented negative finding.")
    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    with open("models/target_decision.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
