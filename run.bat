@echo off
TITLE MedPredict - Patient Readmission Risk Prediction
COLOR 0B

echo ===============================================================================
echo                MEDPREDICT - PATIENT READMISSION RISK PREDICTION
echo               Machine Learning and Exploratory Data Analysis System
echo ===============================================================================
echo.

:: 1. Check Python Availability
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ and ensure it is added to your system environment variables.
    pause
    exit /b 1
)

echo [OK] Python is available:
python --version
echo.

:: 2. Check Dataset and Trained Model Pipeline
IF NOT EXIST "dataset\diabetic_data.csv" (
    echo [ERROR] Original dataset not found at dataset\diabetic_data.csv!
    pause
    exit /b 1
)

IF NOT EXIST "models\medpredict_pipeline.pkl" (
    echo [WARNING] Trained model pipeline not found at models\medpredict_pipeline.pkl.
    echo Training pipeline now...
    python src\inspect_dataset.py
    python src\data_cleaning.py
    python src\eda.py
    python src\feature_engineering.py
    python src\preprocessing.py
    python src\train_models.py
    python src\evaluate_models.py
    python src\tune_models.py
    python src\explainability.py
)

echo [OK] Dataset and trained pipeline verified.
echo.

:: 3. Launch Streamlit Application
echo ===============================================================================
echo Launching MedPredict Streamlit Web Application...
echo The dashboard will automatically open in your default browser at http://localhost:8501
echo ===============================================================================
echo.

python -m streamlit run app\app.py

pause
