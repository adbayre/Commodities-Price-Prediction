# Commodity Price Forecasting Using Machine Learning

## Project Overview

This project develops a machine learning framework to predict directional price movements (Up/Down/Neutral) for Brent Crude Oil futures over a 5-day horizon. By integrating traditional OHLCV data with alternative sources (Google Trends sentiment, Geopolitical Risk Index, Macroeconomic indicators), the system achieves statistically significant predictive power and positive alpha in backtesting.

The project demonstrates a rigorous data science workflow, moving from a failed linear regression baseline ($R^2 < 0$) to a sophisticated Voting Ensemble (Random Forest + XGBoost + Logistic Regression) that generates a Sharpe Ratio > 1.0.

### Key Features

Multi-Source Data Fusion: Merges 23 years of daily commodity prices with monthly Google Trends data (imputed via forward-fill) and daily Macroeconomic indicators (SOFR, CPI).

Advanced Feature Engineering: Generates 400+ features including Moving Averages, RSI, MACD, and Volatility, followed by aggressive feature selection (Top 20) to combat the "Curse of Dimensionality."

Quantile-Based Labeling: Defines target classes using dynamic 33% quantiles to ensure balanced training data and capture relative market moves.

Walk-Forward Validation: Implements strict temporal splitting to prevent look-ahead bias, simulating real-world trading conditions.

Ensemble Modeling: Combines the low-bias of XGBoost with the low-variance of Random Forest to create a stable, high-performance trading strategy.

### Technology Stack

Language: Python 3.9+

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

### Machine Learning:

scikit-learn (Random Forest, Logistic Regression, Voting Classifier)

xgboost (Gradient Boosting)

tensorflow / keras (LSTM Neural Network)

Statistics: SciPy (Normality tests, Skew/Kurtosis)

Installation & Usage

Clone the repository:

git clone [https://github.com/adbayre/commodity-price-prediction.git](https://github.com/adbayre/commodity-price-prediction.git)
cd commodity-price-prediction

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Authors

Adrien Bayre | Martin Jondeau | Maxime Gruez - Financial Engineering Students (ESILV)
