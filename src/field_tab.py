"""Onglet « Validation terrain » : importer des comptages réalisés à Cotonou."""
import pandas as pd
import streamlit as st

from field_validation import (
    compare_with_targets, load_counts, model_check, observed_profiles, template_csv,
)


def render(hubs, model, feature_columns):
    st.markdown("""
    ### Validation locale par comptages terrain
    Importe les comptages réalisés à Cotonou (voir `docs/PROTOCOLE_COMPTAGE.md`) pour vérifier deux choses :
    **(1)** le profil réel de chaque site ressemble-t-il à l'archétype supposé, et
    **(2)** les probabilités du modèle sont-elles cohérentes avec la saturation observée ?
    """)

    st.download_button(
        "⬇️ Télécharger le modèle de fichier CSV",
        data=template_csv(),
        file_name="field_counts_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Fichier de comptages (CSV)", type=["csv"], key="field_csv")
    if uploaded is None:
        st.info("Aucun fichier importé. Les résultats apparaîtront ici dès l'import de tes comptages.")
        return

    try:
        raw = pd.read_csv(uploaded, sep=None, engine="python")
    except Exception as exc:
        st.error(f"Lecture impossible : {exc}")
        return

    df, problems = load_counts(raw, hubs)
    for p in problems:
        st.warning(p)
    if df is None:
        return

    n_sites = df["site"].nunique()
    st.success(f"{len(df)} fenêtres d'observation exploitables sur {n_sites} site(s).")
    with st.expander("Aperçu des données nettoyées"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- 1. Profils ----------------
    st.subheader("1. Le profil observé correspond-il à l'archétype ?")
    prof = observed_profiles(df)
    table, note = compare_with_targets(prof, hubs)
    if table is None:
        st.info(note)
        st.dataframe(prof.round(2), use_container_width=True)
    else:
        view = table[[
            "n_fenetres", "debit_moyen", "rang_demande_obs", "rang_demande_cible",
            "rang_pointe_obs", "rang_pointe_cible", "rang_nuit_obs", "rang_nuit_cible",
            "ecart_moyen", "verdict",
        ]].round(2)
        st.dataframe(view, use_container_width=True)
        st.bar_chart(table["ecart_moyen"].rename("Écart moyen profil observé / cible"))
        st.caption(
            "Les rangs comparent les sites **entre eux** (0 = le plus faible, 1 = le plus fort). "
            "Écart < 0,25 : cohérent ; 0,25 à 0,40 : partiel ; > 0,40 : divergent. "
            "Le rang « nuit » n'est calculé que si des créneaux de 22 h à 4 h ont été observés."
        )

    # ---------------- 2. Modèle vs saturation observée ----------------
    st.subheader("2. Le modèle est-il cohérent avec la saturation observée ?")
    alert_thr = st.slider("Seuil d'alerte (%)", 10, 90, 50, key="field_thr")
    checked, summary = model_check(df, model, feature_columns, alert_thr)

    c1, c2, c3, c4 = st.columns(4)
    if summary["n_labelled"]:
        c1.metric("Accord alerte / saturation", f"{summary['accuracy']:.0%}")
        c2.metric("Fausses alertes", summary["false_alarms"])
        c3.metric("Saturations manquées", summary["missed"])
        c4.metric("AUC", f"{summary['auc']:.2f}" if summary.get("auc") is not None else "n/a")
        st.caption(
            f"Calculé sur {summary['n_labelled']} fenêtres avec saturation renseignée "
            f"(dont {summary['n_saturated']} saturées)."
        )
        if summary["n_labelled"] < 30:
            st.warning("Moins de 30 fenêtres annotées : résultats indicatifs, pas concluants.")
        if summary.get("auc") is None:
            st.info("L'AUC n'est calculable que si les deux situations (saturé / non saturé) ont été observées.")
    else:
        st.info("Aucune valeur de `saturation_observee` renseignée : seul le test de forme est possible.")

    if summary["corr_by_site"]:
        corr = pd.Series(summary["corr_by_site"], name="Corrélation de rang (débit observé vs probabilité modèle)")
        st.dataframe(corr.round(2), use_container_width=True)
        st.caption("Positive : le modèle monte quand le débit réel monte. Sites avec au moins 4 fenêtres seulement.")

    with st.expander("Détail fenêtre par fenêtre"):
        st.dataframe(
            checked[["site", "date", "hour", "flow_per_hour", "pluie_mm", "saturation", "proba_modele"]].round(1),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Limites")
    st.markdown(
        "- Un échantillon de quelques dizaines de fenêtres donne une validation **exploratoire**, pas statistique.\n"
        "- Le modèle prédit une demande élevée dans le jeu d'entraînement ; la saturation observée est une mesure locale différente.\n"
        "- Les résultats dépendent de la définition de « saturation » fixée avant le terrain (voir le protocole)."
    )