# Obesity Prevalence Prediction – BRFSS Dataset

This project uses the CDC BRFSS "Nutrition, Physical Activity and Obesity" dataset to train a machine-learning model that predicts obesity prevalence (%) based on demographic and survey features. The final model (Random Forest) is deployed through a Streamlit web application.

## Project Structure
project/
├── app.py
├── requirements.txt
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_features.pkl
└── data/
    └── data.csv

## Note on Model Files
All model artifacts (`best_model.pkl`, `scaler.pkl`, `model_features.pkl`) are **automatically created when the Jupyter Notebook is executed**.  
No manual creation is required.

## Installation
Create and activate a virtual environment (optional):

Mac/Linux:
    python3 -m venv venv
    source venv/bin/activate

Windows:
    python -m venv venv
    venv\Scripts\activate

Install dependencies:
    pip install -r requirements.txt

If requirements.txt is missing:
    pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn

## Running the App
Execute this command in the folder containing app.py:
    streamlit run app.py

The application will be available at:
    http://localhost:8501

## Requirements
- Python 3.8+
- Streamlit
- scikit-learn
- pandas, numpy
- joblib

The model files must remain inside the models/ directory for the app to function.

## Notes
- The model predicts obesity prevalence (%) after reversing the StandardScaler transformation.
- All input preprocessing must match the training pipeline.
- Maintain the folder structure exactly as shown above.
