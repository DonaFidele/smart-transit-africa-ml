# SmartTransit Africa – Urban Congestion Risk Lab (Cotonou)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-transit-africa.streamlit.app/)

A decision-support prototype that estimates the risk of traffic saturation at five hubs of Cotonou, Benin,
and turns it into prescriptive recommendations (drainage, traffic regulation, logistics policy).

**Author:** Dona Fidèle Houekpoeha, software engineering graduate (IFRI, University of Abomey-Calavi, Benin), working on AI for transport and supply chains in Benin.
This prototype is a first step toward my research interest in predictive and optimization models for transport and supply chains under uncertainty.

> **Status: demonstration prototype (method transfer).**
> No mobility data from Cotonou was available. The models are trained on a public urban-mobility dataset from
> outside Cotonou, with the **real rainfall** observed at that place and dates (Open-Meteo). Each Cotonou hub is
> matched to the training zone whose demand profile resembles its archetype. The results illustrate the
> *data → model → decision* chain; they are **not a validated forecast for Cotonou**. Local validation with field
> counts is built into the project (see below).

## Key results

Chronological (rolling-origin) validation: 4 folds, 15 test days, 5,400 test rows. With about one month of data, confidence intervals are wide. Results below are for the pre-specified model, Random Forest (lags). The no-skill baseline (always raising an alert) scores F1 = 0.41 on the absolute target and 0.46 on the relative target.

| Horizon | Target   | F1 [95% CI]       | AUC   | Gain vs historical rate        |
|---------|----------|-------------------|-------|--------------------------------|
| 1 h     | absolute | 0.91 [0.89, 0.93] | 0.990 | +0.008 [-0.003, +0.019], n.s.  |
| 1 h     | relative | 0.51 [0.44, 0.56] | 0.741 | +0.121 [+0.064, +0.179], sig.  |
| 24 h    | absolute | 0.89 [0.87, 0.90] | 0.985 | -0.013 [-0.027, +0.001], n.s.  |
| 24 h    | relative | 0.42 [0.36, 0.47] | 0.681 | +0.031 [-0.016, +0.074], n.s.  |

Decision rule (fixed in advance): the relative ("surge") target is adopted only if three criteria pass at both horizons. Skill (AUC ≥ 0.65) and robustness to a wrong zone assignment pass, but the F1 confidence interval does not clear the always-alert baseline at either horizon (lower bounds 0.44 and 0.36 vs 0.46). The absolute target is therefore kept, and the relative-target result is reported as a documented negative finding.

What the results show:

- On the absolute target, F1 is around 0.90 but the models do not significantly beat the historical rate: this target mostly reflects each zone's usual demand.
- Recent history (last hour, same hour yesterday, 24 h rolling mean) dominates feature importance; rainfall contributes under 4%.
- If a hub is matched to the wrong zone, the lag-based Random Forest is unaffected (F1 0.91 → 0.91), while the version without lags collapses (0.90 → 0.33): the transfer to Cotonou depends on the matching.

These results illustrate the *data → model → decision* chain; they are not a validated forecast for Cotonou.

<!-- Add a screenshot of the app here once saved in docs/images/ (remove the comment markers):
![Risk profile tab](docs/images/risk_profile.png)
-->

### Sensitivity of the Cotonou → zone transfer

Each Cotonou hub is matched to a training zone through assumed archetype profiles. To test how much this assumption matters, the archetype targets were perturbed with noise (sd = 0.15, 1,000 simulations).

| Hub                                       | Keeps its zone (%) | Expected shift (pts) | Alert flips vs runner-up zone (%) | Verdict  |
|-------------------------------------------|--------------------|----------------------|-----------------------------------|----------|
| Grand Marché de Dantokpa (Commercial Hub) | 60.1               | 22.8                 | 63.7                              | moderate |
| Échangeur de Godomey (West Gateway)       | 58.9               | 18.2                 | 58.0                              | moderate |
| Zone Portuaire / Akpakpa (Logistics Hub)  | 32.2               | 2.4                  | 1.5                               | moderate |
| Carrefour Vèdoko (Central Junction)       | 41.7               | 28.1                 | 65.2                              | fragile  |
| Carrefour Cadjehoun (Avenue Jean-Paul II) | 25.7               | 17.2                 | 9.5                               | fragile  |

No hub is classified as stable. For three hubs (Dantokpa, Godomey, Vèdoko), swapping the assigned zone for the runner-up would flip the alert decision in 58–65% of weekday-hour-rain cases, so their predictions depend strongly on the matching assumption. For the port hub and Cadjehoun, the assignment is unstable but the runner-up zone behaves similarly, so the alert decision is little affected. This is why local field counts (`docs/PROTOCOLE_COMPTAGE.md`) are required before drawing local conclusions.

## What is inside

| Part | File | Role |
|---|---|---|
| Data adapter | `src/data_adapter.py`, `data_config.json` (optional) | Turns any dataset into one canonical hourly table; see "Use your own data" |
| Training | `src/model.py` | Zones (KMeans), Cotonou → zone matching, Random Forest (hour, weekday, zone, rain) |
| Weather | `src/weather.py` | Open-Meteo: historical rain (training) and live forecast (app) |
| Forecasting | `src/forecast_model.py`, `src/forecast_features.py` | Lag-based models, 1 h and 24 h ahead |
| Evaluation | `src/evaluate.py` | Rolling-origin validation, model comparison, 95% CIs, calibration, absolute vs relative target |
| Target decision | `src/choose_target.py` | Pre-registered rules deciding whether to adopt the relative ("surge") target |
| Sensitivity | `src/sensitivity.py` | How much predictions move if the archetype assumptions are wrong |
| Field validation | `src/field_validation.py`, `docs/PROTOCOLE_COMPTAGE.md` | Compare real counts with profiles and model alerts |
| App | `app.py`, `src/theme.py`, `src/*_tab.py` | Streamlit dashboard (dark theme) |
| Tests | `tests/test_core.py` | Automated checks (synthetic data) |

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Put the training dataset at data/urban_mobility_raw_data.csv
# (columns: Date/Time, Lat, Lon)

python src/model.py             # trains the main model, writes models/ (needs internet once, for the rain data)
python src/forecast_model.py    # trains the 1 h / 24 h lag models
python src/sensitivity.py       # sensitivity of the Cotonou -> zone transfer
python src/evaluate.py          # statistical evaluation (CIs, model comparison, calibration)
python src/choose_target.py     # applies the fixed decision rules (absolute vs relative target)
# only if choose_target says so:
python src/forecast_model.py --target relative
streamlit run app.py
```

Run every command from the project root (the scripts use relative paths).

## Use your own data (e.g. a dataset from Benin)

The whole pipeline reads ONE canonical table (hour x zone x activity), produced by `src/data_adapter.py`.
Without any config file, the original behaviour is kept (event file `data/urban_mobility_raw_data.csv`).
To use another dataset, copy one of the examples from `config_examples/` to `data_config.json` and edit it:

| Your data | Example config | What happens |
|---|---|---|
| Trips / pickups with timestamp + coordinates, from another city | `other_city_events.json` | Events are clustered into zones; each Cotonou hub is matched to the most similar zone (`profile_matching`) |
| Hourly or finer **vehicle counts** per site, measured in Cotonou | `cotonou_counts.json` | Each site is a zone; each hub IS its own zone (`direct`); no matching, no transfer |
| **Speeds** per road segment, measured in Cotonou | `cotonou_speeds.json` | Speed is converted to a congestion load = max(free-flow speed - speed, 0); then as above |

Then run the usual commands (`python src/model.py`, `python src/forecast_model.py`, ...). In `direct` mode the
sensitivity analysis is skipped (nothing to match) and the app adapts its wording and tables automatically.

Checklist before trusting local data: at least one rainy season covered (otherwise the rain effect is not learned),
several months of history, known sensor outages (they are filled and reported in the console), a defined unit.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## App tabs

- **Risk profile** – 24-hour risk curve for the selected hub and comparison of the five hubs.
- **Hub map** – hubs coloured by status (critical / watch / stable).
- **Action log** – archive of simulations and prescriptive guidelines.
- **Engineering** – feature importances, real-weather validation, transfer table, statistical evaluation, sensitivity.
- **Forecast** – backtest of the lag models (1 h / 24 h ahead) against baselines.
- **Field data** – upload field counts (CSV) to test the transfer with real observations.

## Method in brief

1. Pickups are clustered into 15 spatial zones. Two target definitions are compared: *absolute* ("demand in the top 25%
   of slots", strongly tied to the zone) and *relative* ("surge above the zone's own usual level"). The relative target
   is adopted only if it passes rules fixed in advance (`src/choose_target.py`).
2. Each Cotonou hub has an archetype (commercial hub, logistics hub, …) with a target profile
   (demand level, peak share, night share). An optimal one-to-one assignment matches hubs to zones.
3. Rain enters the model as a continuous variable (mm/h) from real weather data.
4. Validation is chronological (never random), against naive baselines.

## Known limitations

- Training data come from outside Cotonou; the transfer rests on assumed archetype profiles.
- The learned effect of rain is that of the training location, not of Cotonou's tropical rain.
- The target measures high *demand*, not congestion itself.
- About one month of data: confidence intervals are wide; see `python src/evaluate.py`.
- Field counts (`docs/PROTOCOLE_COMPTAGE.md`) are required before drawing local conclusions.


## Deployment note

Model files (`models/*.pkl`) must be read with the same scikit-learn version that created them.
Pin your versions before deploying: `pip freeze | grep -iE "scikit-learn|pandas|numpy|scipy|streamlit"`.