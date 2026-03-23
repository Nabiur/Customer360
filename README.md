# Customer360

## Overview
Customer360 is a Streamlit machine learning app for retail customer analytics. It combines three decision workflows in one interface: churn prediction, customer lifetime value (CLV) prediction, and customer segmentation.

## Features
- Customer churn prediction for at-risk customers.
- CLV prediction for revenue-focused prioritization.
- Customer segmentation with cluster-based business actions.
- Country-aware inputs and interactive prediction forms.

## Repository structure
- `streamlit_code.py`: main Streamlit application.
- `requirements.txt`: dependency list.
- Model files (`*.pkl`): trained models, encoders, and scalers.
- `retail.png`: visual asset used in the app home page.

## Tech stack
- Python
- Streamlit
- scikit-learn, xgboost
- pandas, numpy, scipy
- matplotlib, seaborn, plotly

## Getting started
1. Clone the repository.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run streamlit_code.py
   ```

## Use cases
- Retention teams can prioritize likely churners.
- Marketing teams can allocate budget based on predicted CLV.
- CRM teams can personalize outreach by segment type.

## Future improvements
- Add model explainability (SHAP) for better decision transparency.
- Add model performance dashboard and drift monitoring.
- Package training and inference as a reproducible pipeline.
