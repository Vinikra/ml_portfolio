# Energy Consumption Forecasting with XGBoost

## Overview
This project analyzes and forecasts energy consumption using historical hourly data. The analysis includes exploratory data analysis (EDA), time series decomposition, feature engineering, and predictive modeling using XGBoost.

## Dataset
The dataset used for this project is `PJME_hourly.csv`, which contains hourly energy consumption data. The script automatically loads this dataset from the `data/` directory.

## Project Structure
```
project_directory/
│-- data/                 # Directory containing the dataset
│-- output/               # Directory where plots and metrics are saved
│-- script.py             # Main script for data analysis and forecasting
│-- README.md             # Project documentation
```

## Dependencies
Ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `xgboost`
- `scikit-learn`
- `statsmodels`

Install them using:
```bash
pip install pandas numpy matplotlib xgboost scikit-learn statsmodels
```

## Workflow
### 1. Load and Explore Data
- Load the dataset and set the `Datetime` column as the index.
- Check for missing values, outliers, and data consistency.
- Perform exploratory data analysis (EDA) including:
  - Time series visualization
  - Hourly average energy consumption
  - Seasonal decomposition

### 2. Preprocessing
- Create time-based features (hour, day, month, etc.).
- Engineer lag features and rolling mean.
- Handle missing values using interpolation.
- Ensure all features are numerical.

### 3. Model Training
- Split the dataset into training (after the first year) and testing (first year).
- Train an `XGBRegressor` model using relevant features.
- Generate predictions for the test set.

### 4. Evaluation
- Compute evaluation metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
- Visualize prediction errors and compare actual vs. predicted values.
- Save metrics and plots in the `output/` directory.

## Results
The script outputs:
- `eda_plots.png`: Daily average consumption for the first year.
- `hourly_avg.png`: Average consumption per hour of the day.
- `decomposition.png`: Seasonal decomposition of energy consumption.
- `forecast_1day.png`: XGBoost forecast vs. actual values for the first 24 hours.
- `error_distribution.png`: Histogram of forecast errors.
- `metrics.txt`: Model performance metrics.

## Running the Script
Simply execute:
```bash
python script.py
```
Ensure that the dataset is placed inside the `data/` folder before running.

## Future Improvements
- Experiment with different forecasting models (LSTM, ARIMA, etc.).
- Improve feature engineering with external data (e.g., temperature, holidays).
- Optimize hyperparameters for better prediction accuracy.

