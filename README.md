# CreditGuard-AI

> **AI-powered credit risk assessment platform that predicts loan default probability using XGBoost and Explainable AI (SHAP), exposed through a FastAPI backend and an interactive Streamlit dashboard.**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)

---

# Overview

CreditGuard-AI is an end-to-end machine learning system for predicting loan default risk using the **Home Credit Default Risk** dataset.

The project simulates a real-world credit risk assessment workflow by combining large-scale feature engineering, machine learning, explainable AI, REST APIs, and an interactive dashboard into a production-style pipeline.

Unlike conventional credit scoring systems that only generate a prediction, CreditGuard-AI provides **feature-level explanations** for every decision using **SHAP (SHapley Additive Explanations)**, improving transparency and interpretability.

---

# Why CreditGuard-AI?

Financial institutions require more than accurate predictions—they also need to understand **why** a customer is considered high or low risk.

CreditGuard-AI addresses this by combining:

- Predictive machine learning using XGBoost
- Explainable AI using SHAP
- REST API deployment with FastAPI
- Interactive visualization through Streamlit

The result is a transparent and modular credit risk prediction platform.

---

# Key Features

- End-to-end credit risk prediction pipeline
- Feature engineering across multiple Home Credit relational datasets
- Customer-level feature aggregation
- Automated preprocessing pipeline
- XGBoost-based classification model
- Hyperparameter optimization using Optuna
- Threshold optimization for imbalanced classification
- SHAP-based feature attribution
- FastAPI inference API
- Interactive Streamlit dashboard
- Modular project architecture for experimentation and deployment

---

# Dataset

This project uses the **Home Credit Default Risk** dataset.

The feature engineering pipeline integrates information from multiple relational datasets:

- application_train
- bureau
- bureau_balance
- previous_application
- installments_payments
- credit_card_balance
- POS_CASH_balance

Customer-level aggregations are generated from each dataset before being merged into the primary application dataset for model training.

> **Note:** The original dataset is not included in this repository due to GitHub file size limitations.

Dataset:
https://www.kaggle.com/competitions/home-credit-default-risk

---

# System Architecture

```
                 Home Credit Dataset
                         │
                         ▼
            Feature Engineering Pipeline
                         │
                         ▼
               Customer-Level Aggregation
                         │
                         ▼
                Data Preprocessing
                         │
                         ▼
             XGBoost Classification Model
                         │
                         ▼
              SHAP Explainability Engine
                         │
                         ▼
               Prediction Pipeline
                         │
                         ▼
                 FastAPI REST API
                         │
                         ▼
            Interactive Streamlit Dashboard
```

---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Explainable AI | SHAP |
| Hyperparameter Optimization | Optuna |
| Backend | FastAPI |
| Frontend | Streamlit |
| Visualization | Plotly |
| Model Serialization | Joblib |

---

# Model Performance

The current production XGBoost model achieves:

| Metric | Score |
|---------|-------|
| ROC-AUC | **0.7902** |
| Accuracy | **85.69%** |
| Precision | **27.03%** |
| Recall | **45.50%** |
| F1 Score | **33.92%** |

The model was optimized using **Optuna** with threshold tuning to improve performance on the highly imbalanced loan default dataset.

---

# Explainable AI

Traditional machine learning models often operate as black boxes.

CreditGuard-AI integrates **SHAP (SHapley Additive Explanations)** to provide feature-level reasoning for every prediction.

Example output:

| Feature | Impact |
|----------|---------|
| External Credit Score (Mean) | Increases Risk |
| External Credit Score (Maximum) | Increases Risk |
| External Credit Score 3 | Increases Risk |
| Age | Reduces Risk |
| Debt-to-Credit Ratio | Reduces Risk |

This allows users to understand the key drivers influencing the predicted credit risk.

---

# REST API

The prediction pipeline is exposed through a FastAPI backend.

Example endpoint:

```
GET /customers/{customer_id}
```

Example response:

```json
{
    "customer_id":100002,
    "probability":0.7587,
    "risk":"High Risk",
    "top_factors":[
        ...
    ]
}
```

---

# Project Structure

```
CreditGuard-AI
│
├── backend
│   ├── api
│   └── ml
│       ├── artifacts
│       ├── models
│       └── src
│
├── frontend
│   ├── dashboard_professional_v2.py
│   └── ui
│
├── docs
│
├── requirements.txt
│
└── README.md
```

# Model Experiments

Multiple machine learning models were evaluated throughout the development of CreditGuard-AI before selecting the final production model.

| Model | Purpose | Status |
|--------|---------|--------|
| Logistic Regression | Baseline linear classifier for benchmarking | Evaluated |
| Random Forest | Ensemble tree-based baseline | Evaluated |
| XGBoost | Gradient boosting model with Optuna hyperparameter optimization | **Selected for Production** |
| LightGBM | Gradient boosting comparison model | Evaluated |
| XGBoost (Cross Validation) | Stratified K-Fold validation | Evaluated |
| XGBoost Ensemble | Ensemble experimentation using multiple boosting strategies | Experimental |

### Model Selection

The final production model was selected after comparing multiple approaches on the Home Credit Default Risk dataset.

The selection process included:

- Baseline model benchmarking
- Hyperparameter optimization using Optuna
- Threshold optimization for imbalanced classification
- Cross-validation experiments
- Performance comparison using ROC-AUC, Precision, Recall, F1 Score, and Accuracy

After experimentation, the **Optuna-tuned XGBoost model** consistently delivered the best balance between predictive performance and model stability, making it the production model used by the prediction pipeline and FastAPI service.
---

# Installation

Clone the repository

```bash
git clone https://github.com/Dhaanya-Nair-13/CreditGuard-AI.git

cd CreditGuard-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the FastAPI backend

```bash
uvicorn backend.api.app:app --reload
```

Launch the Streamlit dashboard

```bash
streamlit run frontend/dashboard_professional_v2.py
```

---

# Future Enhancements

- Docker containerization
- Cloud deployment
- User authentication
- Batch prediction support
- Database integration
- Model monitoring
- CI/CD pipeline
- React-based frontend

---

# About the Developer

**Dhaanya Nair**

B.Tech Computer Science and Engineering (AI & Robotics)

VIT Chennai

---

# License

This project is intended for educational and research purposes.
