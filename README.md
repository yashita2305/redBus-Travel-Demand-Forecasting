RedBus Travel Demand Forecasting (2025)

Data-driven demand forecasting pipeline for the redBus Data Decode Hackathon 2025 — predicting 15-day advance travel demand using booking and search data.

-Project Overview

This project addresses one of the most pressing challenges in travel-tech:

"How can we accurately forecast passenger demand for specific bus routes 15 days in advance, when only ~20% of bookings occur early?"

I designed and implemented a scalable forecasting solution that leverages real-time search logs, booking trends, and route-level features to predict demand, enabling better fleet allocation, pricing, and operational planning for travel platforms like redBus.

-Problem Statement

Industry Challenge: Limited early bookings → weak demand signals.

Objective: Build a robust, explainable, and production-ready forecasting pipeline for route-level daily demand prediction.

Impact: Improved accuracy of demand planning, reduced operational inefficiencies, and better customer experience.

-Methodology & Approach
1. Data Engineering & Feature Design

Extracted temporal features: weekday/weekend patterns, holidays, and event proximity.

Route-level aggregations & clustering to capture demand similarity.

Constructed lag, rolling, and trend features from booking/search data.

Search-to-booking conversion ratios as leading demand indicators.

2. Modeling Framework

Ensemble Forecasting combining:

Gradient boosting models: XGBoost, LightGBM

Prophet for capturing seasonality & benchmarking

Stacked ensemble meta-learners for hybrid predictions

Integrated time-series decomposition for trend & seasonality.

3. Evaluation

Metrics: SMAPE and RMSE on unseen dates.

Route-specific error breakdown to highlight under- vs. over-prediction cases.

-Results & Outcomes

  -Improved accuracy compared to baseline benchmarks.

  -Developed an explainable ML pipeline adaptable across different routes.

  -Demonstrated practical scalability for real-world deployment.

-Tech Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Prophet

Visualization: Matplotlib, Seaborn, Plotly

Workflow: Jupyter Notebooks, GitHub, Colab

-Author

  Yashita Yadav

