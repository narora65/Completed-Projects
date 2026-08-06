# Completed Projects

A collection of data science, meteorology, and analytics projects spanning statistical modeling, machine learning, and geospatial visualization.

---

## Projects

### 2026 College Football Score & Playoff Prediction Pipeline
**Files:** `cfb_pipeline.py` · `index.html` (**Live site:** [2026 CFB Predictions](https://narora65.github.io/Completed-Projects/))  
**Language:** Python

An end-to-end machine learning pipeline that predicts every FBS college football game for the 2026 season from 24 years of historical box scores (2002–2025), then simulates conference standings, conference championships, and the 12-team College Football Playoff. Combines a trained XGBoost score-prediction model with a sequential Elo rating system, incorporating recruiting composite rankings, transfer portal data, returning production, SP+ ratings, and coaching-hire quality as preseason adjustments. The final projection is built from the median outcome across 20 independently simulated seasons rather than a single random draw, so one lucky or unlucky sequence of results doesn't dominate the result. Published as a self-contained, interactive website with weekly rankings, conference standings, upset tracking, and a full team-by-team schedule lookup.

**Methods & Tools:**
- Sequential Elo rating system with margin-of-victory scaling, home-field adjustment, and season-to-season regression toward the mean
- XGBoost regression models for home/away score prediction, trained/validated/tested on a chronological 2002–2023 / 2024 / 2025 split
- Opponent-adjusted rolling scoring form (a team's last 4/8 games measured against what each specific opponent normally allows/scores, not raw point margins)
- Preseason team-strength blend combining 247Sports recruiting composite, transfer portal rankings, returning production, and SP+ ratings
- Coaching-change detection with additional Elo regression plus a standalone, non-negative coaching-hire-grade adjustment
- Monte Carlo simulation (20 independent seasons) with per-game median aggregation for the published projection
- Custom conference standings engine with a full tiebreaker chain (head-to-head, overall record, point differential, PPG, PAPG, coin flip) and divisional support
- 12-team College Football Playoff simulation with realistic seeding rules, including conference-champion auto-bids and the Notre Dame at-large exception
- Real NCAA overtime rules simulated for any game that lands on a tie after rounding
- Self-contained interactive front end (vanilla JS/HTML/CSS) with live weekly rankings, conference standings, upset alerts, and team lookup

**Packages:** `pandas`, `numpy`, `xgboost`, `scipy`

---

### Quantifying U.S. Hail Risk
**Files:** `Quantifying U.S. Hail Risk Code.R` · `Quantifying U.S. Hail Risk Report.pdf`  
**Language:** R

An extreme value analysis of severe hail (≥1 inch diameter) across U.S. counties using NOAA Storm Events data from 1950–2024. The project quantifies hail risk through county-level return levels for 50-, 100-, and 500-year recurrence intervals.

**Methods & Tools:**
- Mann-Kendall trend tests to classify counties as stationary vs. nonstationary
- GEV (Generalized Extreme Value) distribution fitting via L-moments (stationary) and MLE (nonstationary, time-varying location parameter)
- Bootstrap and MCMC uncertainty quantification on example counties
- Geospatial choropleth mapping with `ggplot2` and `maps`
- Property and crop damage analysis by hail size category

**Packages:** `tidyverse`, `extRemes`, `lmom`, `Kendall`, `revdbayes`, `ggplot2`, `maps`, `scales`

---

### Forecasting Omaha Temperatures Using Recurrent Neural Networks
**Files:** `Forecasting_Omaha_Temperatures_Using_Recurrent_Neural_Networks_.ipynb` · `Forecasting Omaha Temperatures Using Recurrent Neural Networks Report.pdf`  
**Language:** Python (Jupyter Notebook)

A deep learning project applying recurrent neural networks (RNNs/LSTMs) to forecast temperature in Omaha, Nebraska. Explores sequence modeling for meteorological time series.

---

### Dynamic Meteorology: Modeling the Barotropic Vorticity Equation
**File:** `Dynamic_Meteorology_Modeling_The_Barotropic_Vorticity_Equation.ipynb`  
**Language:** Python (Jupyter Notebook)

A numerical simulation of the barotropic vorticity equation, a fundamental model in dynamic meteorology used to simulate large-scale atmospheric flow patterns.

---

### Global GDP Rankings & Visualizations
**Files:** `Global GDP Rankings & Visualizations.sql` · `Global GDP Rankings & Visualizations.twbx`  
**Languages:** SQL · Tableau


Data analysis and interactive dashboard exploring global GDP rankings and trends. SQL used for querying and transforming data; Tableau used for visualization and storytelling.


## Contact

Questions or feedback? Feel free reach out through GitHub.
