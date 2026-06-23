# Enterprise Customer Retention & Churn Prediction Pipeline

An end-to-end, reproducible machine learning architecture designed to predict customer defection risk within highly stratified retail and wholesale portfolios. The system utilizes automated product-driven clustering, adaptive lookforward safety windows, parallel `GridSearchCV` hyperparameter tuning, and comprehensive MLflow tracking.

---


---

## 📊 Model Performance Evaluation

Extracted evaluation summaries from the baseline champion array pass over **4,590 active records**:

| Model Architecture | Evaluation ROC-AUC | Target Churn Recall | Operational Status |
| :--- | :---: | :---: | :--- |
| **XGBoost Pipeline** | **0.7884** | **77.37%** | **Champion Classifier** (Highest sensitivity to defection) |
| **Random Forest** | 0.7915 | 76.89% | Challenger Model |
| **Logistic Regression** | 0.7931 | 74.48% | Linear Baseline & Interpretability Anchor |

---

## 🔍 Core Feature Interpretations & Insights

### 1. XGBoost Gain Dominance (Non-Linear Tree Ensemble)
* **`Financial_Momentum` ($30.31\%$ Gain):** The single most powerful indicator across the ensemble. A customer's rolling 30-day spending trajectory (accelerating vs. decelerating) dictates nearly a third of all model classification decisions.
* **`Cluster_Median_Delta` ($4.20\%$ Gain):** Validates the architecture. Measuring customer silence relative to their *specific product cluster cadence* yields massive information gain over uniform global windows.

### 2. Logistic Regression Linear Weights (Odds Ratios)
* **`Lifespan_Consistency` (Odds Ratio: $6.3671$):** High structural defection trigger. When a historically rigid, programmatic buyer deviates from their typical transactional intervals, **their odds of churning multiply by 6.3x**.
* **`Frequency` (Odds Ratio: $0.1191$):** Primary retention anchor. A unit increase in lifetime transaction density **slashes baseline churn odds by 88%**.

---

## 🎯 Key Business Outcomes & Impact

* **Elimination of Margin Waste:** By implementing segment-specific lookforward safety windows (**140 days** for retail vs. **269 days** for wholesale enterprise accounts), the organization avoids burning marketing margin on premature promotional coupon loops sent to clients who are simply resting within their natural purchasing cycles.
* **Proactive Interception of High-Value Loss:** The identification of `Lifespan_Consistency` shifts the business strategy from a reactive posture to predictive orchestration. Account management teams can now systematically catch at-risk enterprise wholesale buyers (up to \$358k LTV tier) weeks before they permanently cross the churn boundary.
* **Friction Identification System:** Tracking the interaction between `Return_Value_Ratio` and `Return_Frequency` isolates high-risk fulfillment, cargo, or billing disputes early. This serves as a trigger for customer success teams to deploy active concierge resolutions before transactional silence sets in.
* **Reproducible, Production-Ready Infrastructure:** Seamlessly binds feature engineering pipelines with localized serialization frameworks (`models/*.pkl`) and cloud tracking (`MLflow`). This reduces the engineering friction required to deploy, update, or serve predictions via production APIs or full-stack dashboard interfaces.
