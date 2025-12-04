# Commodity Price Forecasting Using Machine Learning

This project develops a machine learning framework to forecast **directional price movements** (Up / Down / Neutral) for **Brent Crude Oil futures** over a **5-day horizon**.  
It fuses traditional market data with alternative information sources and follows a complete, production-style research workflow.  
The final ensemble model delivers **statistically significant predictive power** and a **Sharpe Ratio > 1.0** in backtesting.

---

## Project Overview

Starting from a failed linear regression baseline (**R² < 0**), the pipeline evolves toward a **Voting Ensemble** combining Random Forest, XGBoost, and Logistic Regression.  
This ensemble achieves robust out-of-sample performance and generates **positive, stable alpha**.

### Key Features

- **Multi-Source Data Fusion**
  - 23 years of daily OHLCV data  
  - Google Trends sentiment (monthly → forward-filled to daily)  
  - Geopolitical Risk Index (GPR)  
  - Daily macroeconomic indicators (SOFR, CPI)

- **Advanced Feature Engineering**
  - 400+ engineered features: Moving Averages, RSI, MACD, Rolling Volatility, etc.  
  - Aggressive feature selection → **Top 20** predictors retained  
  - Addresses the “Curse of Dimensionality”

- **Quantile-Based Labeling**
  - Uses dynamic 33% quantiles (Up / Neutral / Down)  
  - Ensures balanced classes and captures relative market moves

- **Walk-Forward Validation**
  - Strict temporal splits  
  - Eliminates look-ahead bias  
  - Simulates real-world deployment constraints

- **Ensemble Modeling**
  - XGBoost: low bias  
  - Random Forest: low variance  
  - Logistic Regression: linear interpretability  
  - Combined via a Voting Classifier for stability and accuracy

---

## Technology Stack

- **Language:** Python 3.9+  
- **Data Processing:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Machine Learning:**  
  - scikit-learn (Random Forest, Logistic Regression, Voting Classifier)  
  - xgboost  
  - tensorflow / keras (LSTM prototype)  
- **Statistics:** SciPy (normality tests, skewness, kurtosis)

---

## Installation

```bash
git clone https://github.com/adbayre/commodity-price-prediction.git
cd commodity-price-prediction
