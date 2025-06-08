# Forecasting Case

This repository provides a modular and extensible framework for performing time series forecasting on hierarchical or multi-level datasets. It includes exploratory data analysis, forecasting model classes, and executable scripts to handle forecasting across different aggregation levels.

##  Quick Start

### 1.Clone the Repository

```bash
git clone https://github.com/lcszarpellon/Forecasting_case.git
cd Forecasting_case
```

---
### 2. 📊 Run Exploratory Data Analysis

```bash
python code_data_analysis/main_data_analysis.py
```

This script will load `toy_dataset.xlsx`, perform basic EDA, and save charts/tables in `exports/`.

---

### 4.Train and Forecast

#### Run model on single-level data:

```bash
python code_forecast_model/main_forecast_model.py
```

#### Run model on multiple hierarchical levels:

```bash
python code_forecast_model/main_forecast_model_multiple_levels.py
```

Forecast results will be exported to `code_forecast_model/exports/`.

---

## Forecasting Architecture

The forecasting logic is built using modular Python classes (inside `code_forecast_model/classes/`) that handle:

* Data preparation
* Model training
* Forecast generation
* Evaluation metrics

This allows easy extension with new models (e.g., ARIMA, Prophet, LSTM).

---

## Data

The repository includes a small illustrative dataset:

```
data/toy_dataset.xlsx
```

Make sure your own datasets follow a similar structure (e.g., time series format with group levels if needed).

---

## Example Use Case

```python
from classes.forecasting_model import ForecastModel

model = ForecastModel(data_path='data/toy_dataset.xlsx')
model.train()
predictions = model.forecast()
model.plot_results()
```

---
## Contact

Developed by **Lucas Ingles Zarpellon**
📧 \[[lucasiz.zarpellon0@gmail.com](mailto:lucasiz.zarpellon0@gmail.com)]
🔗 [LinkedIn](https://www.linkedin.com/in/lucasingleszarpellon/)

---

