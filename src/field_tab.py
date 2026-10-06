"""Field validation tab: import counts collected in Cotonou."""
import pandas as pd
import streamlit as st

from field_validation import (
    compare_with_targets, load_counts, model_check, observed_profiles, template_csv,
)


def render(hubs, model, feature_columns):
    st.markdown("""
    **Local validation with field counts.** Import the counts collected in Cotonou
    (see `docs/PROTOCOLE_COMPTAGE.md`) to check two things:
    **(1)** does each site's real profile match its assumed archetype, and
    **(2)** are the model's probabilities consistent with the saturation actually observed?
    """)

    st.download_button(
        "⬇️ Download the CSV template",
        data=template_csv(),
        file_name="field_counts_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Field counts file (CSV)", type=["csv"], key="field_csv")
    if uploaded is None:
        st.info("No file imported yet. Results will appear here once you upload your counts.")
        return

    try:
        raw = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        return

    df, problems = load_counts(raw, hubs)
    for p in problems:
        st.warning(p)
    if df is None:
        return

    st.success(f"{len(df)} usable observation windows across {df['site'].nunique()} site(s).")
    with st.expander("Preview of cleaned data"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- 1. Profiles ----------------
    st.subheader("1. Does the observed profile match the archetype?")
    prof = observed_profiles(df)
    table, note = compare_with_targets(prof, hubs)
    if table is None:
        st.info(note)
        st.dataframe(prof.round(2), use_container_width=True)
    else:
        view = table[[
            "n_windows", "mean_flow", "demand_rank_obs", "demand_rank_target",
            "peak_rank_obs", "peak_rank_target", "night_rank_obs", "night_rank_target",
            "mean_gap", "verdict",
        ]].round(2)
        st.dataframe(view, use_container_width=True)
        st.bar_chart(table["mean_gap"].rename("Mean gap: observed vs target profile"), color="#22d3ee")
        st.caption(
            "Ranks compare sites **with each other** (0 = lowest, 1 = highest). "
            "Gap < 0.25: consistent; 0.25 to 0.40: partial; > 0.40: divergent. "
            "The night rank is only computed if 10 pm to 4 am windows were observed."
        )

    # ---------------- 2. Model vs observed saturation ----------------
    st.subheader("2. Is the model consistent with observed saturation?")
    alert_thr = st.slider("Alert threshold (%)", 10, 90, 50, key="field_thr")
    checked, summary = model_check(df, model, feature_columns, alert_thr)

    c1, c2, c3, c4 = st.columns(4)
    if summary["n_labelled"]:
        c1.metric("Alert / saturation agreement", f"{summary['accuracy']:.0%}")
        c2.metric("False alarms", summary["false_alarms"])
        c3.metric("Missed saturations", summary["missed"])
        c4.metric("AUC", f"{summary['auc']:.2f}" if summary.get("auc") is not None else "n/a")
        st.caption(
            f"Computed on {summary['n_labelled']} windows with saturation filled in "
            f"({summary['n_saturated']} saturated)."
        )
        if summary["n_labelled"] < 30:
            st.warning("Fewer than 30 labelled windows: results are indicative, not conclusive.")
        if summary.get("auc") is None:
            st.info("AUC needs both situations (saturated and not saturated) to have been observed.")
    else:
        st.info("No `observed_saturation` values filled in: only the shape check is possible.")

    if summary["corr_by_site"]:
        corr = pd.Series(summary["corr_by_site"], name="Rank correlation (observed flow vs model probability)")
        st.dataframe(corr.round(2), use_container_width=True)
        st.caption("Positive: the model rises when real flow rises. Only sites with at least 4 windows.")

    with st.expander("Window-by-window detail"):
        st.dataframe(
            checked[["site", "date", "hour", "flow_per_hour", "rain_mm", "saturation", "model_prob"]].round(1),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Limits")
    st.markdown(
        "- A few dozen windows give an **exploratory** validation, not a statistical one.\n"
        "- The model predicts high demand in the training dataset; observed saturation is a different, local measure.\n"
        "- Results depend on the definition of “saturation” fixed before fieldwork (see the protocol)."
    )