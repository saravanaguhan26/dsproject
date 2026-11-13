# MedPredict — Viva Voce, Defense & Oral Exam Comprehensive Guide
**55 Comprehensive Technical, Theoretical, and Clinical Questions & Answers**

---

### Category 1: Dataset & Problem Formulation

#### 1. What dataset was used in this project and what is its provenance?
**Answer:** The project utilizes the *Diabetes 130-US Hospitals (1999–2008)* dataset from the UCI Machine Learning Repository. It comprises 101,766 hospital encounters across 130 medical centers over a 10-year span, specifically representing diabetic inpatients who received medical care and medication management.

#### 2. Why was this specific dataset chosen?
**Answer:** It is the benchmark clinical dataset for studying inpatient diabetic outcomes and post-discharge hospital readmission. It contains a rich blend of administrative codes, laboratory results, diagnostic ICD-9 codes, and longitudinal medication tracking across 20+ distinct antidiabetic compounds.

#### 3. What is the original target variable and what values does it contain?
**Answer:** The original column is `readmitted`. It contains three distinct categorical values:
- `<30`: Readmission occurred within 30 days of discharge.
- `>30`: Readmission occurred more than 30 days after discharge.
- `NO`: No recorded subsequent readmission.

#### 4. How was the binary ML target `readmission_risk` formulated?
**Answer:** 
$$\text{readmission\_risk} = \begin{cases} 1, & \text{if } \text{readmitted} = \text{'<30'} \\ 0, & \text{if } \text{readmitted} \in \{\text{'NO'}, \text{'>30'}\} \end{cases}$$

#### 5. Why are `<30` days treated as Class 1 while `NO` and `>30` are grouped into Class 0?
**Answer:** Under clinical standards and health policy frameworks (such as the CMS Hospital Readmissions Reduction Program), a readmission within 30 days indicates acute post-discharge complications, premature discharge, or transitional care breakdown. Conversely, readmissions beyond 30 days reflect routine chronic disease progression rather than acute inpatient discharge quality failures.

#### 6. Did you modify or overwrite the original `diabetic_data.csv` file?
**Answer:** No. In accordance with strict data science reproducibility standards, `dataset/diabetic_data.csv` was treated as an immutable, read-only data artifact. All transformations, cleaning, and engineered features were saved to new files (`cleaned_diabetic_data.csv` and `feature_engineered_data.csv`).

---

### Category 2: Data Cleaning & Missing Values

#### 7. How were missing values encoded in the raw dataset?
**Answer:** Missing values were primarily encoded as the literal string `'?'` rather than standard empty or `NaN` representations. For example, features such as `weight`, `medical_specialty`, and `payer_code` contained large percentages of `'?'`.

#### 8. How did you handle the string `'None'` in lab assay columns like `max_glu_serum` and `A1Cresult`?
**Answer:** The string `'None'` in these columns was preserved as a valid clinical category. It signifies that the attending physician did not order the test during the inpatient stay, which is clinically informative regarding physician practice patterns and patient presentation.

#### 9. Why was the `weight` column dropped during cleaning?
**Answer:** The `weight` feature exhibited over 96.85% missingness. Imputing nearly 97% of values across 101,766 rows would introduce massive statistical noise and artificial bias into the model.

#### 10. Why were `encounter_id` and `patient_nbr` excluded from the predictor matrix?
**Answer:** `encounter_id` is an arbitrary unique primary key with 100% uniqueness, providing zero generalizable predictive signal. `patient_nbr` is an internal patient identifier; including it would lead to direct target memorization and severe overfitting for patients with multiple admissions.

#### 11. What is administrative data leakage, and which columns were removed because of it?
**Answer:** Administrative data leakage occurs when features contain information that would not be available or valid at the decision point (pre-discharge). We removed:
- `discharge_disposition_id`: Disposition codes 11, 13, 14, 19, 20, 21 indicate hospice transfer or patient expiration, trivially dictating that 30-day readmission is impossible.
- `payer_code`: Primarily an insurance billing feature finalized during hospital back-office billing reconciliation post-discharge.

#### 12. How were remaining numerical and categorical missing values imputed?
**Answer:** Numerical missing values were imputed using median values to remain robust against skewed distributions. Categorical missing values were imputed using a constant indicator string (`'Unknown'` or `'missing'`).

---

### Category 3: Feature Engineering & Preprocessing

#### 13. What engineered features were created and what was their clinical rationale?
**Answer:**
1. `total_previous_visits`: Sum of past-year outpatient, emergency, and inpatient encounters, capturing overall healthcare system reliance.
2. `has_previous_inpatient_visit`: Binary indicator for prior inpatient stay.
3. `has_previous_emergency_visit`: Binary indicator for emergency department utilization.
4. `age_group`: Lifecycle consolidation (`Young_Adults`, `Middle_Aged`, `Seniors`).
5. `stay_category`: Categorized stay duration (`Short` $\le 3$ d, `Medium` 4–7 d, `Long` $>7$ d).
6. `active_diabetes_meds_count` and `medication_complexity`: Quantifies antidiabetic polypharmacy.

#### 14. What is strict data leakage prevention in machine learning preprocessing?
**Answer:** It is the principle that information from the test dataset must never inform the training of preprocessing transformations. Transformers (e.g., `StandardScaler`, `SimpleImputer`, `OneHotEncoder`) must be fitted strictly on the training partition ($X_{train}$) and only applied (`.transform()`) to the test partition ($X_{test}$).

#### 15. What train-test split configuration was used?
**Answer:** An 80/20 train/test split (81,412 training records, 20,354 test records) with `random_state=42` and strict stratification on `readmission_risk`.

#### 16. Why is stratification essential when splitting this dataset?
**Answer:** The positive class represents only 11.16% of the records. Random unstratified sampling could lead to disparate class proportions between train and test partitions, distorting performance evaluation. Stratification guarantees identical class ratios (88.84% : 11.16%) across both folds.

#### 17. Why was `OneHotEncoder(handle_unknown='ignore')` used instead of pandas `get_dummies`?
**Answer:** `get_dummies` does not fit a persistent transformer object and fails if unseen categories appear during production inference. Scikit-learn's `OneHotEncoder(handle_unknown='ignore')` safely encodes unseen categories as all zeros without raising runtime errors.

#### 18. Into how many features did the `ColumnTransformer` expand the feature space?
**Answer:** The 50 input features were expanded into **2,384** orthogonal features, primarily due to one-hot encoding high-cardinality ICD-9 diagnostic codes.

---

### Category 4: Class Imbalance & Evaluation Metrics

#### 19. What is the class imbalance ratio in this dataset?
**Answer:** Approximately 8:1 (88.84% negative class vs. 11.16% positive class; imbalance multiplier of 7.96x).

#### 20. Why is accuracy a misleading and dangerous metric for this problem?
**Answer:** A naive dummy model that predicts "No Readmission" for every single patient would achieve **88.84% accuracy**, yet its clinical recall would be **0.0%**, failing to identify a single high-risk patient.

#### 21. What is Recall (Sensitivity) and why is it prioritized in MedPredict?
**Answer:** Recall is $\frac{\text{TP}}{\text{TP} + \text{FN}}$. It measures the proportion of actual 30-day readmissions correctly identified by the model. Missing an at-risk patient (False Negative) can lead to preventable complications or death, making sensitivity clinically paramount.

#### 22. What is Precision and what trade-off occurs when maximizing Recall?
**Answer:** Precision is $\frac{\text{TP}}{\text{TP} + \text{FP}}$. In an imbalanced setting, aggressive cost-weighting forces the model to capture more true positives, which inherently increases false positives, reducing precision (~17.4%).

#### 23. What is F1-score?
**Answer:** The harmonic mean of Precision and Recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
It balances the trade-off between false alarms and missed detections.

#### 24. What is ROC-AUC and what does our score (0.6536) indicate?
**Answer:** Receiver Operating Characteristic Area Under the Curve evaluates the model's ranking ability across all possible classification thresholds. A score of 0.6536 indicates that there is a 65.36% probability that the model assigns a higher risk score to a randomly chosen readmitted patient than to a non-readmitted patient.

#### 25. What is PR-AUC and why is it useful in imbalanced problems?
**Answer:** Precision-Recall Area Under the Curve focuses specifically on the minority positive class. While random guessing yields an ROC-AUC of 0.50, random guessing on PR-AUC equals the positive class prevalence (0.1116). MedPredict's PR-AUC of 0.2045 represents a substantial improvement over baseline chance.

#### 26. Did you use SMOTE? Why or why not?
**Answer:** SMOTE was evaluated but avoided in final training. Synthesizing artificial data points along k-nearest neighbors across 2,384 sparse one-hot dummy variables introduces severe synthetic noise and medically invalid categorical combinations. Instead, algorithmic cost-weighting (`class_weight='balanced'`) was employed.

---

### Category 5: Machine Learning Algorithms & Tuning

#### 27. What 5 classification algorithms were evaluated?
**Answer:**
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

#### 28. How does Logistic Regression work in this setting?
**Answer:** It models the log-odds of readmission as a linear combination of normalized input features using the sigmoid function:
$$P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta^T X)}}$$

#### 29. How does Random Forest differ from Decision Trees?
**Answer:** A Decision Tree constructs a single hierarchical rule tree that is prone to high variance and overfitting. Random Forest builds an ensemble of $B$ decorrelated trees using bootstrap aggregation (bagging) and random feature subspace sampling, drastically reducing variance.

#### 30. How does Gradient Boosting differ from Random Forest?
**Answer:** Random Forest builds independent trees in parallel. Gradient Boosting builds trees sequentially, where each successive shallow tree fits the negative gradient (pseudo-residuals) of the loss function from previous trees.

#### 31. What is XGBoost and what optimizations does it offer?
**Answer:** Extreme Gradient Boosting is an optimized distributed gradient boosting library featuring second-order Taylor expansion approximations, weighted quantile sketch tree learning, sparsity awareness, and histogram-based binning (`tree_method='hist'`).

#### 32. How was hyperparameter tuning executed?
**Answer:** Using `RandomizedSearchCV` with Stratified 3-Fold Cross-Validation conducted strictly on $X_{train}$. The optimization objective was ROC-AUC.

#### 33. Why must hyperparameter tuning never use the test set?
**Answer:** Using the test set during tuning causes data snooping and information leakage. The hyperparameters would overfit to the specific test partition, invalidating the test set as an unbiased benchmark of generalization error.

#### 34. What was the champion model and its configuration?
**Answer:** **Random Forest Classifier**:
- `n_estimators`: 100
- `max_depth`: 18
- `min_samples_split`: 20
- `min_samples_leaf`: 10
- `class_weight`: `'balanced'`

#### 35. What were the exact final test metrics of the champion Random Forest model?
**Answer:**
- **Accuracy:** 0.6694 (66.94%)
- **Precision:** 0.1740 (17.40%)
- **Recall:** 0.5240 (52.40% — intercepted 1,190 of 2,271 at-risk patients)
- **F1-Score:** 0.2613
- **ROC-AUC:** 0.6536
- **Confusion Matrix:** TN=12,435, FP=5,648, FN=1,081, TP=1,190

---

### Category 6: Model Explainability & Interpretability

#### 36. What method was used for model explainability?
**Answer:** Tree-based Mean Decrease in Impurity (Gini Importance), which calculates the total normalized reduction of Gini impurity brought by that feature across all split nodes in the ensemble.

#### 37. What were the top 3 most influential features in MedPredict?
**Answer:**
1. `number_inpatient` (13.17%)
2. `total_previous_visits` (10.27%)
3. `has_previous_inpatient_visit` (8.97%)

#### 38. What clinical insights do the top features reveal?
**Answer:** Historical healthcare utilization is the dominant predictor of readmission. Patients with frequent prior acute hospitalizations represent medically fragile individuals with severe chronic illness burdens and outpatient management barriers.

#### 39. Does high feature importance imply clinical causation?
**Answer:** **No.** Feature importance measures statistical association within the historical training data. A high importance score does not prove that modifying that feature will medically prevent readmission.

#### 40. Why was SHAP not the primary export in the real-time Streamlit app?
**Answer:** Computing TreeSHAP values across 2,384 one-hot encoded features for 20,000+ records introduces substantial latency on standard CPU environments. Precomputed Gini importance tables and cached visual artifacts provide instantaneous, deterministic interpretability.

---

### Category 7: Pipeline Serialization & Inference

#### 41. What is contained in `models/medpredict_pipeline.pkl`?
**Answer:** The complete, unified Scikit-Learn `Pipeline` containing both the fitted `ColumnTransformer` (preprocessor step) and the tuned `RandomForestClassifier` (classifier step).

#### 42. Why is saving the complete unified pipeline better than saving separate preprocessor and model files?
**Answer:** A unified pipeline eliminates manual feature alignment bugs. The end-to-end pipeline accepts a raw Pandas DataFrame, executes imputation, scaling, and one-hot encoding internally, and outputs final predictions and probabilities without intermediate file exports.

#### 43. How does `src/predict.py` validate incoming patient inputs?
**Answer:** It checks data types, ensures required features are present, validates numeric bounds (e.g., `time_in_hospital` between 1 and 14 days; non-negative visit counts), and supplies standard clinical defaults for unselected optional medications.

#### 44. What output format does `predict_readmission()` return?
**Answer:** A structured JSON-serializable dictionary containing: `prediction` (0 or 1), `risk_label` ("HIGH READMISSION RISK" or "LOW READMISSION RISK"), `probability` (float), `probability_percent` (string), class 0 and 1 probabilities, `model_name`, UTC `timestamp`, and mandatory clinical disclaimer.

---

### Category 8: Streamlit Application & Architecture

#### 45. What are the six pages in the MedPredict Streamlit dashboard?
**Answer:**
1. **Dashboard:** High-level KPIs, project summaries, and key distribution charts.
2. **EDA:** Interactive tabs covering demographics, stay durations, procedures, medications, and correlations.
3. **Model Performance:** Comparative benchmark table, confusion matrices, ROC/PR curves, and selection rationale.
4. **Prediction:** Interactive patient risk calculator with form validation.
5. **Explainability:** Feature importance chart and clinical interpretation.
6. **About:** Project scope, methodology, tech stack, and disclaimers.

#### 46. Are metrics hardcoded in the Streamlit application?
**Answer:** **No.** All metrics, tables, feature importances, and model configurations are dynamically loaded from verified artifacts (`models/best_model_info.json`, `models/model_comparison.csv`, `models/feature_importance.csv`, and `models/preprocessing_info.json`).

#### 47. What is the "What-If" sensitivity analysis feature in the prediction tab?
**Answer:** It allows users to dynamically modify select patient features (e.g., prior inpatient visits or length of stay) using sliders and immediately observe the model's updated readmission probability and delta change, holding all other features constant.

#### 48. Does the application protect patient privacy during export?
**Answer:** Yes. The export feature generates a de-identified JSON summary containing clinical features and prediction probabilities, strictly excluding personal identifiers (e.g., patient name, phone numbers, addresses, social security/Aadhaar numbers).

---

### Category 9: Software Engineering, Testing & Reproducibility

#### 49. How is reproducibility guaranteed across the project?
**Answer:** By setting explicit seeds (`random_state=42`) across all stochastic operations (train/test splitting, cross-validation folds, Random Forest bagging, and XGBoost tree construction).

#### 50. What automated test suites were built?
**Answer:** Three dedicated Pytest suites:
- `tests/test_data.py`: Tests original data preservation, clean missingness handling, target mapping, feature engineering logic, and zero data leakage.
- `tests/test_model.py`: Tests pipeline serialization, model benchmark completeness, metadata validity, and feature importance descending monotonicity.
- `tests/test_prediction.py`: Tests end-to-end inference, probability boundary compliance, output dictionary schema, and input validation exception handling.

#### 51. What was the outcome of the automated test suite?
**Answer:** 17 out of 17 tests passed (100% pass rate) with zero failures.

#### 52. What is `run.bat` and what does it do?
**Answer:** It is a Windows batch script that performs pre-flight checks (verifying Python installation, dataset presence, and model pipeline existence) and launches `streamlit run app/app.py` in the default web browser.

---

### Category 10: Clinical Limitations & Disclaimers

#### 53. What are the primary limitations of this project?
**Answer:**
1. **Temporal Discrepancy:** Historical data (1999–2008) does not reflect post-2020 EHR workflows or modern antidiabetic therapies (GLP-1 receptor agonists, SGLT2 inhibitors).
2. **Missing Social Determinants:** Lacks data on social determinants of health (SDOH), health literacy, and transportation access.
3. **Low Precision Trade-Off:** Achieving 52.4% sensitivity in an 11% prevalence setting results in a precision of 17.4% (false-positive rate).

#### 54. What is the difference between a clinical decision-support tool and an automated diagnostic tool?
**Answer:** A diagnostic tool attempts to independently determine medical conditions and treatment. MedPredict is strictly an educational clinical decision-support prototype that alerts care coordinators to potential statistical readmission vulnerability to prompt manual clinical review.

#### 55. What is the mandatory disclaimer displayed across MedPredict?
**Answer:** *"This prediction is based on statistical patterns learned from historical data and is NOT a medical diagnosis or a substitute for professional clinical judgment."*
