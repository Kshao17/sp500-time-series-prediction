# S&P 500 Time-Series Prediction

This project explores machine learning approaches for predicting S&P 500 market returns using historical financial signals.

The main focus is on time-series feature engineering, chronological evaluation, nonlinear models, ensemble learning, and JAX-based implementation.

## Project Overview

Financial time-series prediction requires careful handling of temporal information.

Unlike standard machine learning datasets, randomly shuffling financial observations can introduce look-ahead bias by allowing information from the future to influence model training.

This project therefore uses a strict chronological train-validation split and compares multiple modeling approaches.

## Key Features

- Chronological train-validation split
- Lag-based feature engineering
- Rolling-window features
- Linear Regression baseline
- Random Forest regression
- Ensemble modeling
- JAX implementation
- RMSE and directional accuracy evaluation

## Data Processing

The data was sorted chronologically before splitting.

- First 80%: Training set
- Last 20%: Validation set
- No random shuffling

Lag features were created to represent previous market states, while rolling means were used to capture short-term trends.

## Models

### Linear Regression

A linear regression model was used as the baseline to establish a reference performance level.

### Random Forest

A Random Forest regressor was used to capture nonlinear interactions between financial signals.

Tree depth was restricted to reduce overfitting to noisy historical market movements.

### Ensemble Model

Multiple models were combined to improve robustness and reduce dependence on a single estimator.

### JAX Implementation

A JAX-based regression model was also implemented to explore functional programming and machine learning workflows using JAX.

## Results

The baseline Linear Regression model achieved:

- Validation RMSE: `0.0159`
- Directional Accuracy: `47.45%`

The baseline results suggested that simple linear relationships were insufficient to capture the nonlinear and noisy behavior of market returns.

Further experiments therefore focused on nonlinear and ensemble-based approaches.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- JAX
- Matplotlib
- Time-Series Feature Engineering
- Ensemble Learning

## Notes

This project was completed as part of a machine learning course project.

The repository focuses on the machine learning methodology and experimental workflow and should not be interpreted as financial or investment advice.
