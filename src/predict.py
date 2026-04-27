import os
import joblib
import pandas as pd

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl'))


def load_model(model_path=None):
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model artifact not found at {path}. Run python src/train.py first.')
    return joblib.load(path)


def normalize_flag(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['yes', 'y', 'true', '1']:
            return 'Yes'
        if value in ['no', 'n', 'false', '0']:
            return 'No'
    return 'No'


def build_input_row(data):
    row = {
        'Tenure Months': int(data.get('tenure', data.get('Tenure Months', 0))),
        'Monthly Charges': float(data.get('Monthly Charges', data.get('MonthlyCharges', data.get('MonthlyCharges', 0)))),
        'Contract': data.get('Contract', 'Month-to-month'),
        'Payment Method': data.get('Payment Method', data.get('PaymentMethod', 'Electronic check')),
        'Internet Service': data.get('Internet Service', data.get('InternetService', 'DSL')),
        'Senior Citizen': normalize_flag(data.get('Senior Citizen', data.get('SeniorCitizen', 'No'))),
        'Paperless Billing': normalize_flag(data.get('Paperless Billing', data.get('PaperlessBilling', 'Yes'))),
        'Partner': normalize_flag(data.get('Partner', 'No')),
        'Dependents': normalize_flag(data.get('Dependents', 'No')),
        'Phone Service': normalize_flag(data.get('Phone Service', 'Yes')),
        'Online Security': normalize_flag(data.get('Online Security', 'No')),
        'Tech Support': normalize_flag(data.get('Tech Support', 'No')),
        'Streaming TV': normalize_flag(data.get('Streaming TV', 'No')),
        'Streaming Movies': normalize_flag(data.get('Streaming Movies', 'No')),
        'Multiple Lines': normalize_flag(data.get('Multiple Lines', 'No')),
    }
    row['Total Charges'] = float(row['Monthly Charges'] * row['Tenure Months'])
    row['AvgMonthlySpend'] = row['Total Charges'] / (row['Tenure Months'] + 1)
    row['MonthlyTenureRatio'] = row['Monthly Charges'] / (row['Tenure Months'] + 1)
    return pd.DataFrame([row])


def calculate_risk(probability):
    if probability >= 0.75:
        return 'High'
    if probability >= 0.50:
        return 'Medium'
    return 'Low'


def predict_churn(request_data):
    model = load_model()
    row = build_input_row(request_data)
    proba = float(model.predict_proba(row)[0][1])
    prediction = int(model.predict(row)[0])
    return {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'probability': round(proba, 4),
        'risk_level': calculate_risk(proba),
    }
