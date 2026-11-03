# Credit Card Fraud Detection

Detecting fraudulent credit card transactions: a binary classification problem on highly imbalanced tabular data. This project walks the full machine learning pipeline, from exploratory analysis to an interpretable, evaluated model.

## Problem

Card payment networks process enormous volumes of transactions every day, and a very small fraction of them are fraudulent (someone using a card or card details that are not theirs). Fraud is costly for banks and customers alike, so institutions rely on automated systems that flag suspicious transactions in real time for review or blocking. The aim of this project is to build and evaluate such a classifier.

Formally, given a set of anonymized numerical features describing a transaction, the task is to predict whether the transaction is **fraudulent (1)** or **legitimate (0)**. This is a binary classification problem.

**Why this problem is hard:**

- **Severe class imbalance.** Fraud accounts for only about 0.17% of transactions. A model that always predicts "legitimate" would be 99.8% accurate while catching no fraud at all, which makes plain accuracy a misleading metric. Evaluation therefore uses Precision-Recall AUC, recall, precision, and the confusion matrix.
- **Asymmetric error costs.** The two mistakes are not equal. A *missed fraud* (false negative) means money is lost, while a *false alarm* (false positive) inconveniences an honest customer and creates review workload. A good model balances catching fraud (recall) against keeping false alarms low (precision).
- **Need for explainability.** In regulated finance, a flag must be justifiable. The model's decisions are therefore examined with SHAP to show which features drive a prediction, rather than treating the model as a black box.

**Why it matters:** fraud detection is a widely used, high-impact application of machine learning. The techniques used here (handling imbalance, choosing the right metrics, and explaining predictions) transfer to many other real-world classification problems where the positive class is rare and the cost of errors is uneven, such as medical screening or network intrusion detection.

## Dataset

- **Name:** Credit Card Fraud Detection
- **Original source:** Machine Learning Group, Université Libre de Bruxelles (ULB) / Worldline. Used in peer-reviewed research. Mirrored on Kaggle.
- **Size:** 284,807 transactions, 492 fraud cases (0.172%).
- **Features:** 28 PCA-anonymized numeric features (`V1`–`V28`), plus `Time`, `Amount`, and the target `Class` (1 = fraud, 0 = legitimate).

> The raw data is not committed to this repo (see `.gitignore`). The notebook downloads it automatically from Kaggle. A Kaggle API key (`kaggle.json`) is provided once when the first cell asks for it.

## Approach

The project follows a complete, standard machine learning pipeline. Each stage is summarised below, with the reasoning behind the main choices.

| Pipeline stage | What this project does |
|---|---|
| Framing | Binary classification; cost of a missed fraud vs. a false alarm |
| EDA | Class imbalance, feature distributions, correlations |
| Preprocessing | Scale `Amount` and `Time`; train/test split (stratified) |
| Imbalance handling | Compare class weights vs. SMOTE |
| Baseline model | Logistic Regression |
| Main model | XGBoost |
| Evaluation | Precision-Recall AUC, recall, F1, confusion matrix (not accuracy) |
| Interpretability | SHAP feature importance |

**Notes on the key decisions:**

- **Stratified split.** The train/test split preserves the same tiny fraud ratio in both sets. Without stratification, a random split could leave very few (or zero) fraud cases in the test set, making the evaluation unreliable.
- **Scaling only `Amount` and `Time`.** The features `V1`–`V28` are already the output of a PCA transformation and are on a comparable scale, so only the two raw columns need standardising.
- **Baseline before the main model.** Logistic Regression is trained first to set a reference score. A more complex model is only worth using if it clearly beats this baseline, and reporting both makes the improvement measurable.
- **Precision-Recall AUC over ROC-AUC.** On a heavily imbalanced dataset, the precision-recall curve gives a more honest picture of performance on the rare positive class than ROC-AUC, which can look optimistic when negatives dominate.
- **Two imbalance strategies compared.** Class weighting and SMOTE (synthetic oversampling) are both tried so the effect of the imbalance-handling choice is visible rather than assumed.

## Results

Tested on 71,202 held-out transactions (123 of them real fraud). The professional model (XGBoost) clearly beat the simple baseline.

| Metric | Logistic Regression (baseline) | **XGBoost (main model)** |
|---|---|---|
| PR-AUC | 0.706 | **0.852** |
| Recall (fraud caught) | n/a | **82.1%** |
| Precision (flags correct) | n/a | **85.6%** |
| F1 score | n/a | **0.838** |

**On the test set, XGBoost caught 101 of 123 frauds (82.1%), missed 22, and raised only 17 false alarms** out of more than 71,000 normal transactions. The jump in PR-AUC from 0.706 to 0.852 shows the stronger model earned its place over the baseline.

Key takeaway: the model catches most fraud while keeping false accusations low. This is a good balance for a real bank, where both missed fraud and false alarms are costly.

### Discussion

A few observations stand out from the results:

- **The stronger model earned its place.** XGBoost lifted PR-AUC from 0.706 to 0.852 over the Logistic Regression baseline. Gradient boosting captures non-linear interactions between features that a linear model cannot, which matters when fraud patterns are subtle combinations of signals rather than a single tell-tale feature.
- **Recall and precision are balanced.** Catching 82% of fraud while keeping precision at 86% means the model is useful without drowning analysts in false alarms. In practice the decision threshold could be adjusted to catch even more fraud at the cost of more false positives, depending on the bank's risk appetite.
- **The model is explainable.** The SHAP summary shows that a small number of the anonymized features drive most predictions. Even though the features are anonymized, being able to point to *which* features mattered is important in regulated finance, where automated decisions must be justifiable.

### Figures

**Confusion matrix (XGBoost):** predictions vs. reality on the test set.

![Confusion matrix](reports/figures/confusion_matrix.png)

**SHAP summary:** which features drive the model's fraud decisions.

![SHAP summary](reports/figures/shap_summary.png)

**Class imbalance:** fraud is a tiny fraction of all transactions, which is why accuracy alone is misleading.

![Class imbalance](reports/figures/class_imbalance.png)

## How to run

### Option A: Google Colab (recommended)
1. Open `notebooks/fraud_detection.ipynb` in [Google Colab](https://colab.research.google.com).
2. Run the first cell and select the `kaggle.json` key when the "Choose Files" button appears. The dataset downloads automatically.
3. Run the remaining cells top to bottom (Runtime → Run all).

### Option B: Local
```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```

## Repository structure

```
credit-fraud-detection/
├── README.md                     project overview (this file)
├── LICENSE                       MIT license
├── requirements.txt              Python dependencies
├── .gitignore                    excludes data, models, and credentials
├── notebooks/
│   └── fraud_detection.ipynb     the full analysis, end to end
└── reports/
    └── figures/                  saved charts used in this README
```

## Future work

Several extensions would strengthen the project further:

- **Threshold tuning and cost analysis.** Choose the decision threshold from the precision-recall curve to hit a target recall, and attach a monetary cost to missed fraud vs. false alarms to find the business-optimal operating point.
- **Additional models.** Compare against other gradient-boosting libraries (such as LightGBM) and a simple neural network.
- **Temporal validation.** Split by time rather than randomly, to check that the model holds up on transactions that occur after those it was trained on.
- **Monitoring plan.** In production, fraud patterns drift over time, so the model would need regular retraining and performance monitoring.

## Limitations

- Features are PCA-anonymized, so individual feature meanings are unknown.
- The dataset is from a specific time period and region, so results may not generalize to other banks, countries, or periods.
- This is a portfolio project, not a production fraud system. It has not been validated for real financial use.

## License

MIT. See [LICENSE](LICENSE).
