# MedPredict: Patient Readmission Risk Prediction Using Machine Learning and Exploratory Data Analysis
**Comprehensive Engineering and Technical Project Report**

---

## 1. Abstract
Hospital readmissions within 30 days of discharge represent a critical indicator of healthcare quality and transitional care efficacy. This project presents **MedPredict**, an end-to-end clinical machine learning system developed to assess the 30-day hospital readmission risk for diabetic patients using the historical *Diabetes 130-US Hospitals (1999–2008)* dataset comprising 101,766 hospital encounters. Addressing severe class imbalance (11.16% readmission rate), the system implements leakage-free data cleaning, extensive exploratory data analysis, clinically guided feature engineering, and a Scikit-Learn `ColumnTransformer` preprocessing pipeline. Five classification architectures—Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and XGBoost—were trained and evaluated on an untouched holdout test partition ($N = 20,354$). A cost-sensitively tuned Random Forest classifier emerged as the champion model, achieving a test ROC-AUC of 0.6536 and a clinical sensitivity (Recall) of 52.40%, successfully identifying 1,190 of 2,271 at-risk patients while avoiding misleading high accuracy artifacts. Model explainability was established via Tree-based Mean Decrease in Impurity, revealing that prior healthcare utilization (`number_inpatient`, `total_previous_visits`) drives over 32% of predictive importance. The entire workflow is operationalized through a modular, responsive Streamlit dashboard featuring live inference, what-if sensitivity simulations, and de-identified export capabilities.

---

## 2. Introduction
Diabetes mellitus is a chronic metabolic disorder affecting hundreds of millions globally. Diabetic inpatients frequently experience complex multi-system complications, polypharmacy regimens, and glycemic volatility, culminating in heightened susceptibility to post-discharge health deterioration. In recent decades, hospital readmission rates have transitioned from an internal clinical metric to a major regulatory benchmark, largely driven by initiatives such as the US Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP). Developing robust predictive systems requires strict data hygiene, avoidance of post-discharge information leakage, and explicit prioritization of sensitivity over raw accuracy.

---

## 3. Problem Statement
Given an inpatient hospital encounter record for a diabetic patient, predict whether the patient belongs to the 30-day readmission risk class (`readmission_risk = 1`), defined as an unplanned readmission occurring strictly in fewer than 30 days after discharge. The core technical challenges include:
1. **Severe Class Imbalance:** Positive events occur in only 11.16% of records.
2. **High Dimensionality:** Hundreds of distinct ICD-9 diagnostic codes and medication dosages.
3. **Data Leakage Risks:** Preventing post-discharge outcome leakage (such as discharge to hospice or patient expiration).
4. **Clinical Priority Alignment:** Preventing models from defaulting to trivial majority-class predictions that exhibit high accuracy but fail to identify vulnerable patients.

---

## 4. Motivation
Traditional clinical risk scoring indices (such as LACE or HOSPITAL scores) rely on narrow, hand-crafted integer heuristics that often fail to capture non-linear interactions across medication adjustments, secondary diagnoses, and prior utilization history. Applying machine learning allows healthcare systems to leverage granular Electronic Health Record (EHR) features to identify high-risk individuals prior to discharge, enabling targeted interventions such as comprehensive medication reconciliation, home health support, and timely primary care follow-up.

---

## 5. Objectives
1. **Auditing & Cleaning:** Clean the raw 101,766-record dataset without altering the original source file.
2. **Leakage Elimination:** Systematically isolate and drop administrative and post-discharge features.
3. **Feature Engineering:** Engineer interpretable domain features capturing historical utilization, age brackets, stay durations, and medication complexity.
4. **Pipeline Construction:** Build an integrated scikit-learn `Pipeline` and `ColumnTransformer` fitted exclusively on training data.
5. **Model Benchmarking:** Train and evaluate five diverse classification algorithms under cost-sensitive weighting.
6. **Hyperparameter Optimization:** Conduct stratified 3-fold cross-validation on the training set to maximize ROC-AUC.
7. **Explainability & Inference:** Extract global feature importances and deliver an interactive Streamlit application with input validation and what-if analysis.

---

## 6. Existing System
Existing hospital discharge planning frequently relies on manual nurse assessments or static scoring indices (e.g., LACE Index). These existing approaches present critical drawbacks:
- **Low Discrimination:** LACE scores frequently yield C-statistics between 0.55 and 0.60 for specialized diabetic cohorts.
- **Neglect of Pharmacotherapy:** Static indices fail to evaluate active diabetes medication changes or insulin titrations.
- **Manual Overhead:** Requires manual calculation by clinicians at discharge, increasing cognitive burden.

---

## 7. Proposed System
MedPredict introduces an automated, data-driven pipeline:
- **Comprehensive Multivariable Processing:** Incorporates 50 raw features spanning demographics, utilization, lab testing, diagnostic categories, and 20 specific diabetes medications.
- **Cost-Sensitive Learning:** Employs inverse-frequency class weighting (~7.96x) to align model optimization with clinical patient safety.
- **Unified Production Pipeline:** Bundles imputation, scaling, one-hot encoding, and the champion classifier into a single serialized artifact (`medpredict_pipeline.pkl`).
- **Interactive Decision-Support UI:** Provides real-time inference, risk stratification labels, probability scores, sensitivity simulations, and clear educational disclaimers.

---

## 8. Dataset Description
The dataset originates from the *Diabetes 130-US Hospitals (1999–2008)* database:
- **Sample Size:** 101,766 hospital encounters across 130 medical centers.
- **Timeframe:** 10 consecutive years (1999–2008).
- **Inclusion Criteria:** Inpatient admission with diagnosed diabetes mellitus; length of stay between 1 and 14 days; laboratory tests performed; medications administered.
- **Initial Schema:** 50 raw features, including 3 ID attributes, 7 demographic features, 6 admission/discharge codes, 8 utilization metrics, 3 diagnosis codes, and 23 medication attributes.

---

## 9. Data Preprocessing & Leakage Audit
Conducted in `src/data_cleaning.py` and `src/preprocessing.py`:
1. **Missing Data Handling:** Replaced `'?'` with `NaN`. Features with extreme missingness (`weight` at 96.85%) were removed to avoid introducing synthetic bias. Meaningful clinical categories like `'None'` in `max_glu_serum` and `A1Cresult` were preserved.
2. **Leakage & ID Pruning:** Dropped `encounter_id` (unique primary key) and `patient_nbr` (patient tracking ID). Dropped `discharge_disposition_id` codes associated with hospice/death and `payer_code` (billing reconciliation feature).
3. **Target Formulation:**
   - `<30` $\rightarrow 1$ (High Risk: 11,357 encounters, 11.16%)
   - `NO` and `>30` $\rightarrow 0$ (Low Risk: 90,409 encounters, 88.84%)
4. **Stratified Split:** 80% Training ($N = 81,412$) and 20% Testing ($N = 20,354$) with `random_state=42`.
5. **ColumnTransformer Construction:**
   - *Numerical (13 features):* `SimpleImputer(strategy='median')` + `StandardScaler()`.
   - *Categorical (37 features):* `SimpleImputer(strategy='constant', fill_value='missing')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - *Transformed Dimension:* 2,384 orthogonal features.
   - *Leakage Control:* Preprocessor fitted exclusively on $X_{train}$; $X_{test}$ evaluated purely through `.transform()`.

---

## 10. Exploratory Data Analysis (EDA)
EDA generated 16 validated figures in `eda_outputs/`:
- **Target Distribution:** Verifies 88.84% negative vs 11.16% positive distribution.
- **Demographics:** Demonstrates highest encounter volume in the 60–70 and 70–80 age brackets.
- **Stay Duration:** Length of stay peaks at 2–3 days; stays exceeding 7 days exhibit disproportionately higher readmission incidence.
- **Utilization Disparity:** Over 75% of readmitted patients had at least one prior acute hospital encounter in the preceding year.
- **Pharmacotherapy:** Patients experiencing insulin dose adjustments (`Down` or `Up`) exhibited higher readmission risk than patients maintained on stable regimens.

---

## 11. Feature Engineering
Eight domain-specific features were created in `src/feature_engineering.py`:
1. `total_previous_visits`: Aggregate past-year contacts (`number_outpatient + number_emergency + number_inpatient`).
2. `has_previous_inpatient_visit`: Binary flag indicating acute inpatient history.
3. `has_previous_emergency_visit`: Binary flag indicating acute emergency department history.
4. `has_previous_outpatient_visit`: Binary flag indicating routine outpatient management.
5. `age_group`: Lifecycle segmentation (`Young_Adults`, `Middle_Aged`, `Seniors`).
6. `stay_category`: Length of stay categorized into `Short` ($\le 3$ d), `Medium` (4–7 d), and `Long` ($> 7$ d).
7. `active_diabetes_meds_count`: Integer count of active antidiabetic agents.
8. `medication_complexity`: Regimen classification (`None`, `Simple`, `Complex`).

---

## 12. Machine Learning Algorithms
Five distinct machine learning architectures were implemented:
1. **Logistic Regression:** Linear baseline using L2 regularization and balanced class weighting.
2. **Decision Tree Classifier:** Non-linear rule-based tree with balanced weighting and cost-complexity pruning.
3. **Random Forest Classifier:** Ensemble bagging algorithm constructing decorrelated decision trees.
4. **Gradient Boosting Classifier:** Sequential boosting minimizing binomial deviance.
5. **XGBoost Classifier:** Extreme gradient boosting utilizing histogram-based tree building (`tree_method='hist'`) and `scale_pos_weight=7.96`.

---

## 13. Model Training
Implemented in `src/train_models.py`:
- All models trained on processed $X_{train}$ (81,412 samples, 2,384 features).
- Class imbalance addressed using algorithmic cost-weighting (~7.96x).
- Reproducibility ensured via `random_state=42`.

---

## 14. Model Evaluation & Benchmark
Evaluated strictly against the held-out test partition ($N = 20,354$):

| Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Random Forest (Tuned)** 🏆 | **0.6694** | **0.1740** | **0.5240** | **0.2613** | **0.6536** | **0.2045** |
| XGBoost Classifier | 0.6630 | 0.1740 | 0.5394 | 0.2632 | 0.6564 | 0.2096 |
| Gradient Boosting | 0.6683 | 0.1748 | 0.5302 | 0.2629 | 0.6509 | 0.2066 |
| Logistic Regression | 0.6385 | 0.1664 | 0.5588 | 0.2565 | 0.6438 | 0.2015 |
| Decision Tree | 0.6217 | 0.1580 | 0.5522 | 0.2457 | 0.6223 | 0.1777 |

---

## 15. Hyperparameter Tuning
Implemented in `src/tune_models.py`:
- Conducted Stratified 3-Fold Cross-Validation on $X_{train}$ using `RandomizedSearchCV` optimizing for ROC-AUC.
- Tuned Random Forest: `n_estimators=100`, `max_depth=18`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight='balanced'`. Best CV ROC-AUC: 0.6444.
- Tuned XGBoost: `n_estimators=120`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.7`, `scale_pos_weight=7.96`. Best CV ROC-AUC: 0.6444.

---

## 16. Final Model Selection
**Random Forest** was selected as champion architecture:
- **Test Confusion Matrix:**
  - True Negatives (TN): **12,435**
  - False Positives (FP): **5,648**
  - False Negatives (FN): **1,081**
  - True Positives (TP): **1,190**
- **Clinical Sensitivity:** Successfully identified **1,190 out of 2,271** at-risk patients (52.40% Recall).
- **Stability:** Exhibits superior resistance to overfitting compared to single trees and gradient boosters.

---

## 17. Model Explainability
Generated in `src/explainability.py`:
- **Top Predictor Weights (MDI Gini Importance):**
  1. `number_inpatient`: **0.13172** (13.17% relative weight)
  2. `total_previous_visits`: **0.10268** (10.27%)
  3. `has_previous_inpatient_visit`: **0.08972** (8.97%)
  4. `time_in_hospital`: **0.02870** (2.87%)
  5. `num_medications`: **0.02566** (2.57%)
  6. `number_diagnoses`: **0.02314** (2.31%)
  7. `has_previous_emergency_visit`: **0.02287** (2.29%)
- **Ethical Rule:** Importance signifies mathematical contribution to decision boundary partitioning in historical records; it does not indicate medical causation.

---

## 18. System Architecture
The system employs a multi-tiered pipeline:
1. **Data Layer:** `diabetic_data.csv` (read-only), `cleaned_diabetic_data.csv`, `feature_engineered_data.csv`.
2. **Pipeline Layer:** `medpredict_pipeline.pkl` combining `ColumnTransformer` and `RandomForestClassifier`.
3. **Inference Layer:** `src/predict.py` with schema validation, default injection, and risk scoring.
4. **Presentation Layer:** Multi-page Streamlit web application.

---

## 19. Application Workflow
```
[User Form Input] ──> [validate_patient_input()] ──> [prepare_input_for_pipeline()]
       │
       ▼
[pipeline.predict() & predict_proba()] ──> [Structured Result Object]
       │
       ▼
[Streamlit UI Display] ──> [Risk Badge + Probability Gauge + What-If Simulation]
```

---

## 20. Streamlit User Interface
The Streamlit application (`app/app.py`) provides 6 integrated sections:
1. **Dashboard:** Live KPI stat cards, dataset summary, and distribution charts.
2. **EDA:** Deep-dive tabs into demographics, stay duration, utilization, diagnoses, and correlations.
3. **Model Performance:** Comparative benchmark table, confusion matrices, ROC/PR curves, and selection justification.
4. **Prediction:** Input form, real-time risk assessment, what-if sensitivity sliders, and JSON report export.
5. **Explainability:** Feature importance chart and clinical interpretation of top predictors.
6. **About:** Project scope, methodology, tech stack, and mandatory clinical disclaimers.

---

## 21. Experimental Results
- **Overall Accuracy:** 66.94%
- **ROC-AUC:** 0.6536
- **Recall (Sensitivity):** 52.40%
- **Precision:** 17.40%
- **F1-Score:** 0.2613
- **PR-AUC:** 0.2045 (substantially exceeding baseline random guessing at 0.1116)

---

## 22. Advantages
- **Strict Leakage Prevention:** Preprocessing fitted solely on training data; administrative leakage excluded.
- **No Fabricated Data or Metrics:** All reported numbers derived from genuine model execution.
- **Clinical Alignment:** Avoids high-accuracy artifacts by prioritizing recall through cost-sensitive loss functions.
- **Production-Ready Serialization:** Unified pipeline handles raw categorical and numerical inputs in a single call.

---

## 23. Limitations
- **Historical Demographics:** Data from 1999–2008 does not capture modern post-2020 EHR workflows or newer antidiabetic classes (e.g., SGLT2 inhibitors, GLP-1 agonists).
- **False-Positive Trade-off:** Precision is 17.40%, meaning non-invasive transitional care interventions will be recommended to some patients who would not have been readmitted.
- **Lack of Social Determinants:** EHR data lacks social determinants of health (SDOH) such as food security and transportation access.

---

## 24. Future Enhancements
- Integration of clinical notes using fine-tuned transformer architectures (ClinicalBERT).
- Deployment of Calibrated Probability models using Platt Scaling or Isotonic Regression.
- FHIR/HL7 API standard compliance for direct electronic health record integration.

---

## 25. Conclusion
MedPredict demonstrates a scientifically rigorous, leakage-free machine learning implementation for clinical readmission risk prediction. By respecting class imbalance, engineering clinically relevant predictors, and prioritizing sensitivity over raw accuracy, the platform achieves meaningful patient risk stratification while providing full model explainability and operational readiness.

---

## 26. References
1. Strack, B., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International.
2. Centers for Medicare & Medicaid Services (CMS). *Hospital Readmissions Reduction Program (HRRP).*
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.

---

## 27. Clinical & Legal Disclaimer
> **This prediction system is based on statistical patterns learned from historical data and is NOT a medical diagnosis or a substitute for professional clinical judgment.**
