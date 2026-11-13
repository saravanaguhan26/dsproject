# MedPredict — Patient Readmission Risk Prediction Using Machine Learning and Exploratory Data Analysis

[![Python 3.12](https://img.shields.io/badge/Python-3.12.4-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-100%25%20Passing-brightgreen.svg)](https://pytest.org/)

---

## 1. Project Overview
**MedPredict** is an end-to-end, production-grade clinical machine learning platform designed to identify diabetic patients at elevated risk of unplanned 30-day hospital readmission. Utilizing historical medical encounter data from 130 US hospitals (1999–2008), the system integrates data cleaning, exploratory data analysis (EDA), clinically guided feature engineering, leakage-free preprocessing pipelines, multi-model benchmarking, hyperparameter tuning, model explainability, and an interactive Streamlit decision-support dashboard.

---

## 2. Problem Statement
Unplanned hospital readmissions within 30 days of discharge impose severe clinical and economic burdens on healthcare systems worldwide:
- **Patient Impact:** Indicates premature discharge, post-discharge complications, or inadequate transitional care.
- **Financial Penalties:** Under regulatory frameworks such as the US Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP), hospitals face financial penalties for excessive readmission rates.
- **Challenge:** Clinical workflows require automated, interpretable tools to triage patients prior to discharge without introducing data leakage or false clinical assumptions.

---

## 3. Objective
1. **Target Formulation:** Accurately classify hospital encounters into 30-day readmission risk (`<30` days = `1`, `NO` / `>30` days = `0`).
2. **Clinical Sensitivity:** Prioritize Recall (sensitivity) to minimize costly False Negatives (unintercepted vulnerable patients) while balancing overall discrimination (ROC-AUC).
3. **Leakage Prevention:** Ensure strict separation of training and evaluation folds, with zero post-discharge administrative leakage.
4. **Interpretability:** Expose global feature importances and patient-level risk drivers to empower clinical awareness without claiming medical causation.
5. **Interactive Deployment:** Deliver a production-ready, modular Streamlit web application.

---

## 4. Dataset Overview & Provenance
- **Source:** *Diabetes 130-US Hospitals for Years 1999–2008* (Center for Machine Learning and Intelligent Systems, UC Irvine).
- **Total Encounters:** **101,766** records.
- **Initial Features:** **50** attributes across demographics, admission sources, diagnostic codes (ICD-9), medications, and healthcare utilization.
- **Integrity Rule:** The raw dataset file at `dataset/diabetic_data.csv` is preserved strictly read-only and unmodified.

---

## 5. Target Definition
The original dataset column `readmitted` contains three empirical categories:
- `<30`: Patient readmitted within 30 days of discharge.
- `>30`: Patient readmitted after 30 days.
- `NO`: No record of subsequent hospital admission.

### Binary Transformation:
$$\text{readmission\_risk} = \begin{cases} 1, & \text{if } \text{readmitted} = \text{'<30'} \\ 0, & \text{if } \text{readmitted} \in \{\text{'NO'}, \text{'>30'}\} \end{cases}$$

- **Class 1 (High Risk):** 11,357 encounters (**11.16%**)
- **Class 0 (Low Risk):** 90,409 encounters (**88.84%**)
- **Class Imbalance Ratio:** ~8 : 1 (7.96x negative-to-positive ratio)
- *Note:* The original `readmitted` column is retained intact in `cleaned_diabetic_data.csv` for exploratory validation.

---

## 6. Data Cleaning & Integrity Audit
Implemented in `src/data_cleaning.py`:
1. **Missing Value Harmonization:** Converted literal `'?'` markers to missing representations. Preserved meaningful `'None'` strings in lab assays (`max_glu_serum`, `A1Cresult`) as "Test Not Ordered".
2. **High-Missingness Pruning:** Evaluated feature `weight` (>96% missing); dropped to prevent introducing severe artificial bias.
3. **Identifier Elimination:** Dropped primary keys `encounter_id` (100% unique) and `patient_nbr` to prevent target memorization and overfitting.
4. **Administrative Leakage Exclusion:** Dropped `discharge_disposition_id` (codes 11, 13, 14, 19, 20 indicate hospice or expiration, trivially guaranteeing zero readmission) and `payer_code` (administrative billing code often finalized post-discharge).
5. **Zero-Variance Removal:** Eliminated constant features providing 0 entropy (`examide`, `citoglipton`).
6. **Imputation Strategy:** Median imputation for numerical features; constant `'Unknown'` for categorical features.

---

## 7. Exploratory Data Analysis (EDA)
Implemented in `src/eda.py` (generating 16 high-resolution figures in `eda_outputs/`):
- **Target Distribution:** Quantifies the 11.16% readmission class imbalance.
- **Demographics:** Analyzes readmission incidence across age cohorts and biological sex.
- **Stay Duration:** Correlates prolonged inpatient stays (>7 days) with elevated readmission rates.
- **Prior Utilization:** Demonstrates that prior inpatient admissions form the single strongest empirical risk differentiator.
- **Pharmacotherapy:** Examines glycemic adjustments (insulin dose decreases/increases) and active medication counts.
- **Diagnoses:** Maps primary ICD-9 categories (circulatory CHF 428, diabetes 250, respiratory 486).
- **Multivariable Correlation:** Pearson heatmap of numerical clinical indicators.

---

## 8. Feature Engineering
Implemented in `src/feature_engineering.py`:
1. `total_previous_visits`: Aggregate past-year healthcare contacts (`number_outpatient + number_emergency + number_inpatient`).
2. `has_previous_inpatient_visit`: Binary flag indicating prior acute hospitalization.
3. `has_previous_emergency_visit`: Binary flag indicating prior emergency department encounter.
4. `has_previous_outpatient_visit`: Binary flag indicating prior outpatient management.
5. `age_group`: Categorization into `Young_Adults` (0–40), `Middle_Aged` (40–70), and `Seniors` (70–100).
6. `stay_category`: Distribution-grounded brackets (`Short` $\le 3$ days, `Medium` 4–7 days, `Long` $> 7$ days).
7. `active_diabetes_meds_count`: Tally of active antidiabetic agents prescribed.
8. `medication_complexity`: Categorized as `None` (0 meds), `Simple` (1 med), or `Complex` ($\ge 2$ meds).

---

## 9. Machine Learning Preprocessing Pipeline
Implemented in `src/preprocessing.py`:
- **Stratified Split:** 80% Training ($N = 81,412$) / 20% Testing ($N = 20,354$) stratified strictly by `readmission_risk` with `random_state=42`.
- **Numerical Sub-Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Sub-Pipeline:** `SimpleImputer(strategy='constant', fill_value='missing')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Transformed Feature Space:** 50 raw features expanded into **2,384** orthogonal features.
- **Strict Leakage Prevention:** The `ColumnTransformer` is fitted **strictly on $X_{train}$**. The test set ($X_{test}$) is transformed strictly with `.transform()` without re-fitting.

---

## 10. Model Training & Comparison Benchmark
Implemented in `src/train_models.py` and `src/evaluate_models.py`:
All models evaluated strictly on the untouched holdout test set ($N = 20,354$):

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest (Tuned)** 🏆 | **0.6694** | **0.1740** | **0.5240** | **0.2613** | **0.6536** | **0.2045** |
| XGBoost Classifier | 0.6630 | 0.1740 | 0.5394 | 0.2632 | 0.6564 | 0.2096 |
| Gradient Boosting | 0.6683 | 0.1748 | 0.5302 | 0.2629 | 0.6509 | 0.2066 |
| Logistic Regression | 0.6385 | 0.1664 | 0.5588 | 0.2565 | 0.6438 | 0.2015 |
| Decision Tree | 0.6217 | 0.1580 | 0.5522 | 0.2457 | 0.6223 | 0.1777 |

*Why not unweighted Random Forest (88.8% accuracy)?* An unweighted classifier achieves high accuracy by trivially predicting all patients as Class 0, yielding a near-zero recall (0.6%). In clinical readmission management, missing a high-risk patient (Type II error) is far costlier than a false alarm. Cost-sensitive weighting (~7.96x) successfully intercepts **1,190 of 2,271** at-risk patients.

---

## 11. Hyperparameter Tuning & Final Model
Implemented in `src/tune_models.py`:
- **Search Method:** `RandomizedSearchCV` over candidate spaces utilizing Stratified 3-Fold Cross-Validation on $X_{train}$ only.
- **Optimization Metric:** ROC-AUC.
- **Optimal Random Forest Configuration:**
  - `n_estimators`: 100
  - `max_depth`: 18
  - `min_samples_split`: 20
  - `min_samples_leaf`: 10
  - `class_weight`: `'balanced'`
- **Persistence Artifact:** Exported complete pipeline to `models/medpredict_pipeline.pkl`.

---

## 12. Model Explainability
Implemented in `src/explainability.py`:
- **Technique:** Tree-based Mean Decrease in Impurity (Gini Importance) across all 2,384 features, saved to `models/feature_importance.csv`.
- **Top 5 Predictors:**
  1. `number_inpatient` (13.17%)
  2. `total_previous_visits` (10.27%)
  3. `has_previous_inpatient_visit` (8.97%)
  4. `time_in_hospital` (2.87%)
  5. `num_medications` (2.57%)
- **Ethical Guideline:** Importance indicates statistical weighting in historical records, **not** clinical causation.

---

## 13. Streamlit Multi-Page Interactive Web Application

The interactive clinical decision-support application is organized into 7 comprehensive modules:
1. **Overview Dashboard:** Population KPI telemetry (101,766 records, 50 features, 11.16% readmission rate), dynamic class breakdown donut chart, length of stay curves, and prior utilization patterns.
2. **Dataset Explorer:** Full cohort schema dictionary, data types, missingness and imputation audit, raw three-class outcome analysis, column distribution deep-dive inspector, and a paginated, filterable encounter table with sanitized CSV export.
3. **Exploratory Data Analysis (EDA):** Interactive Plotly visualizations across patient demographics (Age, Race, Gender), inpatient stay durations, admission characteristics, medication patterns, ICD-9 diagnoses, Pearson correlation heatmap, and custom bivariate exploratory plotting.
4. **Model Performance:** Comparative benchmark table across all 5 models (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC), interactive multi-metric bar chart, confusion matrices, ROC/PR curves, and an interactive **Clinical Decision Threshold & Trade-Off Simulator** operating on the held-out test cohort ($N = 20,354$).
5. **Patient Prediction (Readmission Prediction):** Real-time clinical inference form with 1-click presets (High-Risk Geriatric, Moderate-Risk Adult, Low-Risk Elective), live model risk scoring (`HIGH READMISSION RISK` / `LOW READMISSION RISK`), speedometer gauge, patient vs. cohort multi-axis radar chart, targeted transitional care protocol recommendations, interactive What-If sensitivity sliders, and batch inpatient ward discharge triage.
6. **Model Explainability (Feature Importance):** Interactive top 15-20 clinical predictors horizontal bar chart, attribution ranking table, and a dedicated **Important Factors Considered** analysis of physiological mechanisms (prior inpatient admissions, length of stay, polypharmacy, insulin adjustments).
7. **About and Limitations:** System architecture lifecycle, data provenance, governance, ethical limitations, and the **Viva Voce & Oral Defense Examination Suite** (55 comprehensive technical questions with keyword search and mock interview simulator).

---

## 14. Project Folder Structure

```
DataScience/
├── dataset/
│   ├── diabetic_data.csv              # Read-only original dataset (101,766 x 50)
│   ├── cleaned_diabetic_data.csv      # Cleaned dataset (101,766 x 44)
│   ├── feature_engineered_data.csv    # Feature-engineered dataset (101,766 x 52)
│   ├── sample_inpatient_batch.csv     # 15 sample encounters for ward triage
│   ├── X_train.npy, X_test.npy        # Preprocessed float32 feature matrices (2,384 dims)
│   └── y_train.npy, y_test.npy        # Ground truth target labels (81,412 train / 20,354 test)
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── inspect_dataset.py             # Phase 1: Dataset audit
│   ├── data_cleaning.py               # Phase 2: Data cleaning & target setup
│   ├── eda.py                         # Phase 3: Exploratory data analysis
│   ├── feature_engineering.py         # Phase 4: Feature engineering
│   ├── preprocessing.py               # Phase 5: Leakage-free preprocessing
│   ├── train_models.py                # Phase 6: Multi-model training
│   ├── evaluate_models.py             # Phase 7: Model benchmark & charts
│   ├── tune_models.py                 # Phase 8: Hyperparameter tuning
│   ├── explainability.py              # Phase 9: Model explainability
│   ├── predict.py                     # Phase 10: Prediction engine & validation
│   ├── run_tests.py                   # 86-item comprehensive diagnostic test suite
│   └── utils.py                       # Common constants and path utilities
├── models/
│   ├── medpredict_pipeline.pkl        # Complete unified production pipeline (4.2 MB)
│   ├── model_results.csv              # Initial training benchmark metrics
│   ├── model_comparison.csv           # Final evaluated model comparison
│   ├── best_model_info.json           # Champion model metadata & parameters
│   ├── feature_importance.csv         # Full 2,384 feature importance rankings
│   └── preprocessing_info.json        # Preprocessing shapes and configurations
├── eda_outputs/                       # 39 high-resolution visualizations
├── reports/                           # 9 comprehensive text engineering reports
├── app/
│   ├── app.py                         # Upgraded 7-section Streamlit dashboard
│   └── components/
│       ├── __init__.py
│       └── ui_elements.py             # Healthcare analytics CSS, KPI cards, radar, gauge
├── tests/
│   ├── test_data.py                   # Data integrity and leakage tests
│   ├── test_dataset_explorer_and_app.py # Dataset explorer and app caching tests
│   ├── test_model.py                  # Pipeline artifact and benchmark tests
│   └── test_prediction.py             # Inference and validation unit tests
├── README.md                          # Master documentation
├── PROJECT_AUDIT.md                   # Systematic audit, issue tracking & action plan
├── PROJECT_REPORT.md                  # Comprehensive academic/technical report
├── VIVA_QUESTIONS.md                  # 55 viva and defense Q&As across 10 domains
├── TEST_REPORT.md                     # Verification audit and test matrix
├── FINAL_PROJECT_STATUS.md            # Final delivery checklist and verification
├── requirements.txt                   # Dependency specifications
└── run.bat                            # One-click Windows application launcher
```

---

## 15. Installation & Setup

### Prerequisites
- Python 3.10 to 3.12 installed on your system.

### 1. Open Workspace
```bash
cd "c:\Users\C M SARAVANA GUHAN\OneDrive\Desktop\DataScience"
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 16. How to Run

### Option A: One-Click Windows Launcher
Double-click `run.bat` or run in terminal:
```cmd
run.bat
```

### Option B: Manual Streamlit Launch
```bash
python -m streamlit run app/app.py
```
The application will launch and open automatically in your browser at:
`http://localhost:8501`

### Option C: Re-run Full Machine Learning Pipeline
To execute the pipeline from inspection to explainability:
```bash
python src/inspect_dataset.py
python src/data_cleaning.py
python src/eda.py
python src/feature_engineering.py
python src/preprocessing.py
python src/train_models.py
python src/evaluate_models.py
python src/tune_models.py
python src/explainability.py
```

---

## 17. Automated Testing

### Pytest Test Suite
Execute the automated test suite using `pytest`:
```bash
pytest tests/ -v
```
**Result:** 22/22 tests passing (100% pass rate) covering data integrity, target mapping, zero data leakage, pipeline loading, input validation, probability bounds, sensitivity, dataset explorer schema, and viva question loaders.

### Comprehensive 86-Item Diagnostic Test
Execute the comprehensive end-to-end test runner:
```bash
python src/run_tests.py
```
**Result:** 86/86 checks passing (100% pass rate) with machine-readable results saved to `test_results.log`.

---

## 18. Limitations
1. **Historical EHR Demographics:** The dataset covers US inpatient encounters from 1999–2008. Medical practices and diagnostic criteria have since evolved.
2. **Precision vs. Recall Trade-Off:** Optimizing for clinical sensitivity (Recall = 52.4%) leads to a false-positive rate with Precision around 17.4%.
3. **Observational Nature:** Feature importances indicate statistical associations within historical records; they do not prove that altering any specific feature will clinically prevent readmission.

---

## 19. Future Improvements
- Integration of modern natural language processing (ClinicalBERT) on discharge summary text notes.
- Exploration of calibrated cost-sensitive deep learning tabular architectures (e.g., TabNet, FT-Transformer).
- Integration with HL7/FHIR protocols for real-time electronic health record interoperability.

---

## 20. Clinical & Legal Disclaimer
> **IMPORTANT NOTICE:**  
> MedPredict is an educational machine learning demonstration and research prototype. It does **not** generate medical diagnoses, clinical treatment plans, or formal healthcare recommendations. Predictions generated by this system must **never** be used as a substitute for the clinical assessment, diagnosis, or judgment of a licensed healthcare professional.
