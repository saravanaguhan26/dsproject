# MedPredict — Comprehensive Project Audit Report

**Date of Audit:** October 2026  
**Auditor:** Senior Python Developer, Machine Learning Engineer & QA Specialist  
**Project:** MedPredict – Patient Readmission Risk Prediction Using Machine Learning and EDA  
**Workspace:** `DataScience/`  

---

## 1. Executive Summary

A comprehensive, end-to-end audit was conducted on the MedPredict repository. The project implements a complete clinical machine learning pipeline and interactive Streamlit web dashboard to predict 30-day unplanned hospital readmission for diabetic patients, utilizing the historical 130-US Hospitals (1999–2008) dataset (101,766 encounters, 50 initial features).

The audit verified that:
1. The original raw dataset (`dataset/diabetic_data.csv`) is **fully intact and unmodified** (101,766 rows × 50 columns, preserving original `'?'` missing markers).
2. The target formulation strictly maps `<30` days to `1` (High Risk, 11,357 encounters, 11.16%) and `NO` / `>30` days to `0` (Low Risk, 90,409 encounters, 88.84%), preserving the original `readmitted` column in cleaned data.
3. Strict data leakage controls are enforced: identifiers (`encounter_id`, `patient_nbr`) and post-discharge administrative outcome features (`discharge_disposition_id`, `payer_code`) were removed prior to model training.
4. Preprocessing via Scikit-Learn `ColumnTransformer` is fitted exclusively on the 80% training set ($X_{train}$, 81,412 rows) and applied without fitting to the 20% test set ($X_{test}$, 20,354 rows).
5. Five classifiers were evaluated with cost-sensitive class weighting (~7.96x). The tuned **Random Forest Classifier** is selected as the champion model (Test ROC-AUC: 0.6536, Test Recall: 52.40% intercepting 1,190 of 2,271 at-risk encounters).
6. 17 unit and integration tests under `pytest` pass with 100% success.

### Key Deficiencies Identified:
1. **Missing "Dataset Explorer" Navigation Section:** The dashboard lacked a dedicated interactive dataset exploration module (requested in Phase 4 D) with schema summaries, column-level drill-downs, missing value tables, and interactive paginated/filterable data views.
2. **Hardcoded Dashboard Metrics:** The Overview Dashboard contained hardcoded class distribution counts (`values = [90409, 11357]`) instead of dynamically loading them from preprocessing metadata or dataset stats.
3. **Suboptimal Model Caching in Dashboard:** The dashboard used `@st.cache_data` for JSON files but lacked `@st.cache_resource` for the pipeline artifact, and batch screening loaded the 4.2 MB model pipeline from disk on every rerun.
4. **Test Runner Incompatibilities in `src/run_tests.py`:** 
   - Section 11 (Explainability) assumed a linear model with `.coef_`, causing an `AttributeError` when inspecting the champion `RandomForestClassifier` (which uses `feature_importances_`).
   - Section 12 (Streamlit App File) checked for specific string tokens that differed slightly in naming conventions (`st.cache_resource`, `Patient Prediction`, `Feature Importance`, `Important Factors Considered`, specific disclaimer wording).
5. **Navigation Structure:** The navigation console had 6 pages rather than the comprehensive 7-section hierarchy requested in Phase 4:
   - 1. Overview Dashboard
   - 2. Dataset Explorer *(to be added)*
   - 3. Exploratory Data Analysis
   - 4. Model Performance
   - 5. Readmission Prediction
   - 6. Model Explainability
   - 7. About and Limitations

---

## 2. Directory & Architecture Audit

```
DataScience/
├── dataset/
│   ├── diabetic_data.csv              # [VERIFIED] Original UCI dataset (101,766 x 50, read-only)
│   ├── cleaned_diabetic_data.csv      # [VERIFIED] Cleaned data (101,766 x 44, '?' replaced, target mapped)
│   ├── feature_engineered_data.csv    # [VERIFIED] Engineered features added (101,766 x 52)
│   ├── sample_inpatient_batch.csv     # [VERIFIED] 15 patient records for ward triage screening
│   ├── X_train.npy, X_test.npy        # [VERIFIED] Preprocessed float32 feature matrices (2,384 dims)
│   └── y_train.npy, y_test.npy        # [VERIFIED] Ground truth target labels (81,412 train / 20,354 test)
├── models/
│   ├── medpredict_pipeline.pkl        # [VERIFIED] End-to-end ColumnTransformer + Champion Classifier (4.2 MB)
│   ├── preprocessor.joblib            # [VERIFIED] Standalone ColumnTransformer
│   ├── best_model_info.json           # [VERIFIED] Full hyperparameters, test metrics & selection justification
│   ├── model_comparison.csv          # [VERIFIED] Benchmark across all 5 models (Accuracy, Recall, AUC, PR-AUC)
│   ├── feature_importance.csv         # [VERIFIED] Ranked 2,384 features with MDI & cumulative importance
│   ├── feature_names.txt              # [VERIFIED] 50 canonical input feature names
│   ├── rf_test_probs.npy              # [VERIFIED] Test set predicted probabilities for threshold simulation
│   └── *.joblib                       # [VERIFIED] Individual model weights (LR, DT, RF, GB, XGB)
├── eda_outputs/                       # [VERIFIED] 39 analytical and diagnostic charts
├── reports/                           # [VERIFIED] 9 formal text reports covering all lifecycle stages
├── app/
│   ├── app.py                         # [AUDITED] Streamlit entry point (needs upgrade to 7 sections + caching)
│   └── components/
│       └── ui_elements.py             # [VERIFIED] Custom CSS, KPI cards, radar chart, gauge, threshold matrix
├── src/
│   ├── utils.py                       # [VERIFIED] Centralized paths and constants
│   ├── inspect_dataset.py             # [VERIFIED] Dataset verification and schema inspection
│   ├── data_cleaning.py               # [VERIFIED] Cleaning, leakage removal, and target formulation
│   ├── eda.py                         # [VERIFIED] Exploratory data analysis & 16 Matplotlib figures
│   ├── feature_engineering.py         # [VERIFIED] Clinical domain feature derivations
│   ├── preprocessing.py               # [VERIFIED] Train/test split, ColumnTransformer, leakage audit
│   ├── train_models.py                # [VERIFIED] Model training with class weighting (~7.96x)
│   ├── evaluate_models.py             # [VERIFIED] Holdout evaluation, ROC/PR curves, confusion matrices
│   ├── tune_models.py                 # [VERIFIED] Stratified 3-Fold CV tuning, pipeline persistence
│   ├── explainability.py              # [VERIFIED] Tree MDI feature attribution & top-15 chart
│   ├── predict.py                     # [VERIFIED] Inference engine with schema validation
│   └── run_tests.py                   # [AUDITED] 86-check test script (needs Explainability fix)
└── tests/
    ├── test_data.py                   # [VERIFIED] 5 pytest tests (data integrity, target, leakage)
    ├── test_model.py                  # [VERIFIED] 4 pytest tests (pipeline steps, benchmark, metrics)
    └── test_prediction.py             # [VERIFIED] 8 pytest tests (input validation, ranges, types)
```

---

## 3. Data Flow & Machine Learning Pipeline Audit

| Pipeline Stage | Implementation | Verified Status | Leakage Audit |
|---|---|---|---|
| **1. Ingestion** | `src/inspect_dataset.py` | Original CSV loaded read-only; 101,766 rows, 50 cols | N/A (Read-only inspection) |
| **2. Cleaning** | `src/data_cleaning.py` | '?' replaced with NaN; missing numerical imputed with median, categorical with 'Unknown'; zero-variance cols dropped (`examide`, `citoglipton`); `weight` (>96% missing) dropped | Zero target or post-discharge leakage introduced |
| **3. Target Prep** | `src/data_cleaning.py` | `readmission_risk`: `<30` $\rightarrow 1$ (11,357, 11.16%), `NO` / `>30` $\rightarrow 0$ (90,409, 88.84%). Original `readmitted` preserved intact | Target derived strictly from ground truth column |
| **4. Feature Eng.** | `src/feature_engineering.py` | Derived `total_previous_visits`, utilization flags, `stay_category`, `age_group`, `active_diabetes_meds_count`, `medication_complexity` | Only baseline encounter & pre-admission features used; zero future/post-discharge data |
| **5. Preprocessing** | `src/preprocessing.py` | Stratified 80/20 split ($N_{train}=81,412$, $N_{test}=20,354$). ColumnTransformer fitted **strictly on $X_{train}$** | Test fold transformed strictly without fitting |
| **6. Training** | `src/train_models.py` | 5 models trained with cost-sensitive inverse weighting (~7.96x) to counter 8:1 class imbalance | Balanced class weighting applied only inside training workflow |
| **7. Tuning & Selection**| `src/tune_models.py` | Stratified 3-Fold CV on $X_{train}$; Random Forest selected for superior sensitivity (Recall = 52.40%) & ROC-AUC (0.6536) | Held-out test set evaluated strictly after parameter selection |
| **8. Persistence** | `src/tune_models.py` | Unified `Pipeline(steps=[('preprocessor', ...), ('classifier', ...)])` exported to `models/medpredict_pipeline.pkl` | Self-contained pipeline; prevents training/test feature discrepancies |
| **9. Inference Engine**| `src/predict.py` | Production inference module with input range validation, schema alignment, and structured JSON return | Feature alignment checks against canonical 50 features |

---

## 4. Problems Identified & Action Plan

| ID | Issue Description | Location | Severity | Planned Fix |
|---|---|---|---|---|
| **ISS-01** | Missing Dataset Explorer section in navigation | `app/app.py` | High | Add dedicated "2. Dataset Explorer" section with shape, data types, missingness summary, column inspector, and paginated/filterable data table. |
| **ISS-02** | Hardcoded target distribution values in Dashboard | `app/app.py:270` | Medium | Dynamically populate donut chart and KPI metrics from `load_preprocessing_metadata()` and dataset stats with safe fallbacks. |
| **ISS-03** | Lack of `@st.cache_resource` for pipeline loading | `app/app.py:1026` | Medium | Add cached resource loader `@st.cache_resource def get_cached_pipeline()` and reuse across single and batch prediction workflows. |
| **ISS-04** | `src/run_tests.py` Explainability test expects `.coef_` | `src/run_tests.py:485` | Medium | Support both tree-based ensemble (`feature_importances_`) and linear models (`coef_`). |
| **ISS-05** | `src/run_tests.py` app string token mismatches | `src/run_tests.py:511-532` | Low | Align app text and headings with test tokens (`cache_resource`, `Patient Prediction`, `Feature Importance`, `Important Factors Considered`, `educational machine learning prototype`). |
| **ISS-06** | Absolute paths stored in metadata JSON | `models/best_model_info.json` | Low | Ensure all runtime code resolves relative paths dynamically via `src/utils.py`. |
| **ISS-07** | Missing comprehensive test coverage for app components | `tests/` | Medium | Add `tests/test_pipeline_and_app.py` testing caching, dataset explorer helper, and feature integrity. |

---

## 5. Verification & Test Plan

1. **Unit & Integration Testing:**
   - Execute `pytest tests/ -v` to ensure all existing and new tests pass 100%.
   - Execute `python src/run_tests.py` to ensure all 86 comprehensive project checks pass.
2. **Pipeline Invariance:**
   - Verify that `dataset/diabetic_data.csv` checksum/shape remains unchanged.
   - Verify that `models/medpredict_pipeline.pkl` accurately generates predictions on diverse profiles.
3. **Streamlit Application Verification:**
   - Test headless import: `python -c "import app.app; print('Import OK')"`
   - Verify Streamlit launch: `python -m streamlit run app/app.py`
   - Test all 7 navigation tabs:
     - 1. Overview Dashboard (Dynamic KPIs, interactive donut, population trends)
     - 2. Dataset Explorer (Schema, missingness, column distribution, search & filter, export)
     - 3. Exploratory Data Analysis (Interactive Plotly charts, selectors, clinical insights)
     - 4. Model Performance (Benchmark table, ROC/PR curves, confusion matrix, threshold slider)
     - 5. Readmission Prediction (Preset profiles, form validation, risk gauge, radar chart, what-if sensitivity, ward batch triage)
     - 6. Model Explainability (Top feature importance chart, ranked table, clinical interpretations, causal disclaimers)
     - 7. About and Limitations (Governance, methodology, limitations, 55 Viva Voce Q&A examination suite)
