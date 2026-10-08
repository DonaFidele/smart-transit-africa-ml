# SmartTransit Africa – Urban Congestion Risk Lab (Cotonou)

[![SmartTransit Africa App](https://streamlit.io)](https://smart-transit-africa.streamlit.app/)
A decision-support prototype that estimates the risk of traffic saturation at five hubs of Cotonou, Benin,
and turns it into prescriptive recommendations (drainage, traffic regulation, logistics policy).

> **Status: demonstration prototype (method transfer).**
> No mobility data from Cotonou was available. The models are trained on a public urban-mobility dataset from
> outside Cotonou, with the **real rainfall** observed at that place and dates (Open-Meteo). Each Cotonou hub is
> matched to the training zone whose demand profile resembles its archetype. The results illustrate the
> *data → model → decision* chain; they are **not a validated forecast for Cotonou**. Local validation with field
> counts is built into the project (see below).

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