# Customer Retention Intelligence Dashboard

**An end-to-end machine learning project: bank customer churn prediction with an interactive Streamlit analytics app for retention teams.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Gradient%20Boosting-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.856-2E8B57)
![License](https://img.shields.io/badge/License-MIT-blue)

![Dashboard preview](docs/screenshot.png)

---

## Overview

Acquiring a new customer typically costs far more than keeping an existing one, so identifying customers who are about to leave is a high-value problem for banks. This project turns a trained **Gradient Boosting churn model** into a decision-support dashboard that helps a retention team answer four practical questions:

1. How does customer engagement relate to churn risk?
2. Which product-holding patterns are linked to higher or lower churn?
3. Which high-balance, high-salary customers are disengaged and should be contacted first?
4. How strong is each customer's overall relationship with the bank?

The app scores customers in real time, lets users filter and slice the portfolio, and exports a prioritized outreach list.

## Key Features

| Module | What it does |
|---|---|
| **Engagement vs Churn Overview** | KPIs, churn risk for active vs inactive members, risk-probability distribution, age-band by engagement heatmap, country-level comparison |
| **Product Utilization Impact** | Churn risk by number of products, products by engagement heatmap, credit card effect, summary table with automatic insight text |
| **High-Value Disengaged Detector** | Finds customers above user-set balance and salary thresholds who are inactive, ranks them by *expected balance at risk* (balance x churn probability), and exports a CSV |
| **Retention Strength Scoring** | 0-100 retention score per customer, Strong / Moderate / Weak segment panels, model feature drivers, single-customer score breakdown |

**Interactive controls:** engagement status and score filters, product count slider, minimum balance and salary thresholds, geography, gender and age filters, adjustable risk threshold, and adjustable model weight in the retention score.

## Model Performance

The final model was selected after comparing nine classification algorithms.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Gradient Boosting (selected)** | **85.50%** | **67.16%** | **56.27%** | **61.23%** | **0.8557** |
| Random Forest | 83.25% | 59.38% | 56.02% | 57.65% | 0.8432 |
| AdaBoost | 83.50% | 60.27% | 55.53% | 57.80% | 0.8359 |

- **Cross-validated accuracy:** 88.33%
- **Decision threshold:** 0.59 (tuned rather than left at the default 0.50)
- **Hyperparameters:** `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`, `min_samples_leaf=10`, `subsample=0.8`

**Top predictive features:** Age (31%), Active-member status (16%), Number of products (11%), Gender (7%), Balance (4%), Geography (4%), Engagement score (4%).

## Feature Engineering

The model uses **24 features**: 10 raw customer attributes plus engineered features, including:

- `EngagementScore` (active status x number of products)
- `BalancePerProduct`, `ActiveHighBalance`, `AtRiskPremium`, `SalaryBalanceMismatch`
- Binned groups: `AgeGroup`, `TenureGroup`, `CreditScoreBin`
- Interactions: `ProductTenureInteraction`, `AgeBalanceInteraction`, `ActiveProductScore`
- Encoded categoricals: `Geography_enc`, `Gender_enc`

## Retention Strength Score

The Retention Strength Score (0-100) is a transparent, blended heuristic:

```
Score = w x (1 - churn probability) x 100  +  (1 - w) x mean(behaviour components)
```

Behaviour components are percentile ranks against the training population (computed with the fitted `StandardScaler`): **engagement, tenure, balance, and credit health**. The weight `w` is adjustable in the app. Bands: Strong (70+), Moderate (45-70), Weak (below 45).

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Machine learning | scikit-learn (GradientBoostingClassifier, StandardScaler), joblib |
| Data processing | pandas, NumPy, SciPy |
| Visualization | Plotly |
| Web application | Streamlit |

## Project Structure

```
.
├── app.py                      # Streamlit application
├── best_churn_model.joblib     # Trained Gradient Boosting classifier
├── feature_scaler.joblib       # Fitted StandardScaler (used for retention scoring)
├── model_metadata.joblib       # Feature names, threshold, evaluation metrics
├── requirements.txt            # Python dependencies
├── docs/
│   └── screenshot.png          # Dashboard preview
└── README.md
```

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the app**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. It starts in **demo mode** with synthetic customers; upload your own CSV from the sidebar for real results.

## Input Data Format

| Column | Type | Notes |
|---|---|---|
| `CreditScore` | number | |
| `Geography` | text | France, Germany, or Spain |
| `Gender` | text | Female or Male |
| `Age` | number | |
| `Tenure` | number | Years with the bank |
| `Balance` | number | |
| `NumOfProducts` | number | |
| `HasCrCard` | 0 / 1 | |
| `IsActiveMember` | 0 / 1 | |
| `EstimatedSalary` | number | |
| `Exited` | 0 / 1 | *Optional.* Enables actual-vs-predicted churn comparison |
| `CustomerId` | any | *Optional.* Used for labelling in tables |

## Engineering Decisions

- **Raw features for the tree model:** the Gradient Boosting model was trained on unscaled inputs, so the app does not scale features before prediction. The scaler is used only for retention-component scoring.
- **Threshold tuning:** the decision threshold was tuned for the churn use case instead of using the default 0.50.
- **Business-oriented ranking:** outreach lists are ranked by expected balance at risk, not by probability alone, so effort goes to the most valuable customers first.
- **Input validation:** the app checks required columns, coerces data types, and reports skipped rows instead of failing silently.
- **Caching:** model loading and scoring are cached so filters respond quickly.

## Limitations and Future Work

- Recall is 56%, so some churners will be missed; the risk-threshold slider lets users trade precision against recall.
- Feature-engineering logic is reimplemented in `app.py` for inference; packaging it as a single scikit-learn `Pipeline` would remove any training/serving mismatch risk.
- Planned improvements: SHAP-based per-customer explanations, Docker container, FastAPI prediction endpoint, MLflow experiment tracking, cloud deployment.

## Author

**Sarfaraz Ali**

## License

Released under the [MIT License](LICENSE).
