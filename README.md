# Bank Marketing Analytics - Term Deposit Subscription Prediction

BUU33803 Business Analytics Group Assignment | Trinity College Dublin

## Project Overview

This project analyzes a Portuguese bank's direct marketing campaign data to predict term deposit subscriptions. Using machine learning techniques, we investigate two research questions to provide actionable insights for optimizing customer targeting and campaign execution.

## Group Members

| Name    | Student Number |
|---------|----------------|
| Rachel  |                |
| Kshama  |                |
| Rwan    |                |
| Luisa   |                |

## Research Questions

**RQ1: Financial Profile Impact**  
Does having existing loans (housing or personal) and account balance affect the likelihood of subscribing to a term deposit?

- Methods: Logistic Regression, Random Forest, Decision Trees
- Purpose: Identify high-conversion customer segments for better targeting

**RQ2: Campaign Effectiveness**  
Does the date, duration, and type of campaign call affect the success rate?

- Methods: Logistic Regression, Random Forest, K-Means Clustering
- Purpose: Optimize campaign timing, contact method, and call duration for higher ROI

## Dataset

- Source: UCI Machine Learning Repository (Portuguese Bank Marketing Dataset)
- Records: 4,521 customer contacts
- Variables: 17 features including demographics, financial status, and campaign attributes
- Target: Term deposit subscription (binary: yes/no)
- Class imbalance: ~11% positive class, addressed using SMOTE oversampling

## Repository Structure

```
.
├── Q1.py          # Research Question 1 analysis script
├── Q2.py          # Research Question 2 analysis script
├── Q1_visuals/             # Generated visualizations for RQ1
│   ├── 01_subscription_by_loan_status.png
│   ├── 02_balance_comparison.png
│   ├── 03_subscription_by_balance.png
│   ├── 04_roc_curves.png
│   ├── 05_precision_recall_curves.png
│   ├── 06_confusion_matrices.png
│   ├── 07_key_feature_importance.png
│   ├── 08_decision_tree.png
│   ├── 09_model_comparison.png
│   ├── 10_cross_validation.png
│   └── 11_key_insights.png
├── Q2_visuals/             # Generated visualizations for RQ2
│   ├── 01_success_by_contact_type.png
│   ├── 02_success_by_month.png
│   ├── 03_success_by_duration.png
│   ├── 04_success_by_contacts.png
│   ├── 05_success_by_day.png
│   ├── 06_roc_curves.png
│   ├── 07_precision_recall_curves.png
│   ├── 08_confusion_matrices.png
│   ├── 09_feature_importance.png
│   ├── 10_successful_clusters.png
│   ├── 11_cluster_profiles.png
│   ├── 12_previous_outcome_impact.png
│   └── 13_key_insights.png
├── Bank_Marketing_Data.xlsx
└── README.md
```

## Requirements

Install dependencies:
```bash
pip install requirements.txt
```

## Running the Analysis

```bash
# Run Research Question 1 analysis
python Q1.py

# Run Research Question 2 analysis
python Q2.py
```

Outputs are saved to `Q1_visuals/` and `Q2_visuals/` respectively.

## Key Findings

### Research Question 1
- Customers without existing loans convert at 16.9% vs 6.2% for those with both loans
- Higher account balance correlates with higher subscription likelihood
- Random Forest achieved best performance (AUC: 0.856, F1: 0.476)
- Balance is the most predictive feature among loan/balance variables

### Research Question 2
- Call duration is the dominant predictor of success
- Calls over 10 minutes achieve 45% conversion vs 1.7% for calls under 2 minutes
- Single contact attempts yield 13.8% conversion; diminishing returns after 2+ contacts
- October, December, and March are peak months for conversions
- Cellular contact outperforms other methods

## Methodology

1. Data preprocessing and exploratory analysis
2. SMOTE oversampling to address class imbalance (11% positive class)
3. 70/30 train-test split with stratification
4. 5-fold cross-validation for model stability assessment
5. Feature standardization for Logistic Regression
6. K-Means clustering to identify patterns in successful campaigns

## References

Moro, S., Cortez, P., and Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.

## License

This project was completed as part of the BUU33803 Business Analytics module at Trinity College Dublin.