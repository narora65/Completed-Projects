# Completed Projects

A collection of data science, meteorology, and analytics projects spanning statistical modeling, machine learning, and geospatial visualization.

---

## Projects

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
