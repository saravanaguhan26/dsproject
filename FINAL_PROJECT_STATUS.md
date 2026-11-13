# MedPredict — Final Project Status Report

---

## 1. Project Identification
- **Project Name:** MedPredict – Patient Readmission Risk Prediction Using Machine Learning and Exploratory Data Analysis
- **Domain:** Clinical Machine Learning & Predictive Healthcare Informatics
- **Application Framework:** Streamlit (Multi-Page Interactive Dashboard)
- **Core Technology Stack:** Python 3.12, Scikit-Learn, XGBoost, Pandas, NumPy, Plotly, Pytest

---

## 2. Dataset Specifications
- **Original Source Dataset:** `dataset/diabetic_data.csv` (Diabetes 130-US Hospitals 1999–2008)
- **Dataset File Integrity:** **Preserved 100% read-only and unmodified**.
- **Total Encounter Records:** **101,766** rows.
- **Initial Features:** **50** columns.
- **Engineered Features:** 8 features added (`total_previous_visits`, `has_previous_inpatient_visit`, `has_previous_emergency_visit`, `has_previous_outpatient_visit`, `age_group`, `stay_category`, `active_diabetes_meds_count`, `medication_complexity`).
- **Transformed ML Matrix Dimension:** **2,384** orthogonal features after imputation, scaling, and one-hot encoding.

---

## 3. Target Formulation & Class Distribution
- **Target Variable Name:** `readmission_risk` (binary integer)
- **Source Feature:** `readmitted` (preserved intact in `cleaned_diabetic_data.csv`)
- **Mathematical Mapping:**
  - `<30` days $\rightarrow 1$ (High Risk)
  - `NO` or `>30` days $\rightarrow 0$ (Low Risk)
- **Empirical Class Breakdown:**
  - **Negative Class (0):** 90,409 encounters (**88.84%**)
  - **Positive Class (1):** 11,357 encounters (**11.16%**)
  - **Class Imbalance Multiplier:** 7.96x (~8:1 imbalance ratio)

---

## 4. Leakage Prevention Audit
- **Exclusion of Patient Identifiers:** `encounter_id` (100% unique primary key) and `patient_nbr` (internal identifier) dropped.
- **Exclusion of Post-Discharge Leakage:** `discharge_disposition_id` (codes for hospice transfer/death) and `payer_code` (billing reconciliation data) dropped.
- **Preprocessor Fitting:** Scikit-Learn `ColumnTransformer` fitted **strictly on training fold** ($X_{train}$, 81,412 rows). Test fold ($X_{test}$, 20,354 rows) was strictly transformed without re-fitting.

---

## 5. Machine Learning Models Trained
Five classification algorithms were trained using cost-sensitive class weighting:
1. **Logistic Regression** (L2 Regularization, Balanced Weighting)
2. **Decision Tree Classifier** (Gini Impurity, Pruned, Balanced Weighting)
3. **Random Forest Classifier** (100 Estimators, Balanced Weighting)
4. **Gradient Boosting Classifier** (Deviance Loss, Subsampled)
5. **XGBoost Classifier** (Histogram Tree Method, `scale_pos_weight=7.96`)

---

## 6. Champion Model & Verified Test Metrics
- **Selected Champion Model:** **Random Forest Classifier (Tuned)**
- **Selection Rationale:** Achieved the strongest balance of cross-validated ROC-AUC (0.6444) and held-out test discrimination (0.6536) while prioritizing clinical sensitivity (Recall = 52.40%) to intercept 1,190 of 2,271 at-risk patients under severe class imbalance.
- **Untouched Holdout Test Set ($N = 20,354$):**
  - **Accuracy:** **0.6694** (66.94%)
  - **Precision:** **0.1740** (17.40%)
  - **Recall (Sensitivity):** **0.5240** (52.40%)
  - **F1-Score:** **0.2613**
  - **ROC-AUC:** **0.6536**
  - **PR-AUC:** **0.2045**
  - **Confusion Matrix:**
    - True Negatives (TN): **12,435**
    - False Positives (FP): **5,648**
    - False Negatives (FN): **1,081**
    - True Positives (TP): **1,190**

---

## 7. Model Explainability
- **Methodology:** Tree-based Mean Decrease in Impurity (Gini Importance) across all 2,384 features.
- **Key Artifacts:** `models/feature_importance.csv` and `eda_outputs/feature_importance.png`.
- **Top 5 Predictive Drivers:**
  1. `number_inpatient` (13.17%)
  2. `total_previous_visits` (10.27%)
  3. `has_previous_inpatient_visit` (8.97%)
  4. `time_in_hospital` (2.87%)
  5. `num_medications` (2.57%)
- **Causality Notice:** Importance reflects mathematical decision weighting within historical records and does **not** establish medical causation.

---

## 8. Streamlit Web Application Status
- **File Location:** `app/app.py` & `app/components/ui_elements.py`
- **Current Status:** **100% Functional & Verified via Chromium Browser Subagent**
- **Navigation Structure (7 Modules):**
  1. `1. Overview Dashboard`: Dynamic population telemetry cards, class distribution breakdown, stay duration trajectories.
  2. `2. Dataset Explorer`: Feature dictionary, schema inventory, missing value audits, deep-dive column inspector, and paginated patient encounter table.
  3. `3. Exploratory Data Analysis (EDA)`: Interactive Plotly demographic profiles, utilization trends, diagnostic analyses, and custom bivariate explorer.
  4. `4. Model Performance`: 5-model benchmark comparison, confusion matrices, ROC/PR curves, and live decision threshold simulator.
  5. `5. Patient Prediction (Readmission Prediction)`: 1-click clinical presets, structured form, live inference, speedometer risk gauge, radar chart, targeted clinical recommendations, and ward batch triage.
  6. `6. Model Explainability (Feature Importance)`: Gini Impurity feature rankings, interactive top 15-20 predictor bar chart, and physiological factors considered.
  7. `7. About and Limitations`: Governance, data provenance, ethical limitations, and the 55-question Viva Voce oral defense suite.

---

## 9. Automated Testing Status
- **Test Framework:** Pytest 9.1.1 & Custom Diagnostic Runner (`src/run_tests.py`)
- **Test Suites:**
  - `tests/test_data.py`: 5 tests passed
  - `tests/test_dataset_explorer_and_app.py`: 5 tests passed
  - `tests/test_model.py`: 4 tests passed
  - `tests/test_prediction.py`: 8 tests passed
- **Total Pytest Count:** **22 Passed / 0 Failed (100% Pass Rate)**
- **Diagnostic Test Runner:** **86 Passed / 0 Failed (100% Pass Rate)**

---

## 10. Known Limitations
1. **Historical EHR Demographics:** Dataset originates from 1999–2008 and does not reflect post-2020 EHR standards or modern medication classes (e.g., SGLT2 inhibitors).
2. **False Positive Trade-Off:** High sensitivity (52.40%) results in lower precision (17.40%), reflecting the inherent clinical cost trade-off between false alarms and missed interventions.
3. **No Social Determinants:** Lacks data on patient housing, socioeconomic status, and outpatient follow-up adherence.

---

## 11. Future Improvements
- Integrate unstructured discharge notes using transformer-based clinical NLP (ClinicalBERT).
- Implement Platt Scaling or Isotonic Regression for calibrated probability estimation.
- Implement HL7/FHIR API standards for direct electronic health record interoperability.

---

## 12. Run Instructions
To launch the application on Windows:
```cmd
run.bat
```
Or execute manually:
```bash
streamlit run app/app.py
```
To run the automated test suite:
```bash
pytest tests/ -v
```

---

## 13. Mandatory Medical & Legal Disclaimer
> **This prediction system is based on statistical patterns learned from historical data and is NOT a medical diagnosis or a substitute for professional clinical judgment.**
