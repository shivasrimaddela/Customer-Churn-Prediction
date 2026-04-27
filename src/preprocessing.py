import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC_COLUMNS = [
    'Tenure Months',
    'Monthly Charges',
    'Total Charges',
    'AvgMonthlySpend',
    'MonthlyTenureRatio',
]

CATEGORICAL_COLUMNS = [
    'Contract',
    'Payment Method',
    'Internet Service',
    'Senior Citizen',
    'Paperless Billing',
    'Partner',
    'Dependents',
    'Phone Service',
    'Online Security',
    'Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Multiple Lines',
]

TARGET_COLUMN = 'Churn Flag'


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found at {path}')
    if path.lower().endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def detect_columns(df):
    replacements = {
        'Churn Label': TARGET_COLUMN,
        'Churn': TARGET_COLUMN,
        'Total Charges': 'Total Charges',
        'TotalCharges': 'Total Charges',
        'Monthly Charges': 'Monthly Charges',
        'MonthlyCharges': 'Monthly Charges',
        'Tenure Months': 'Tenure Months',
        'tenure': 'Tenure Months',
        'Senior Citizen': 'Senior Citizen',
        'SeniorCitizen': 'Senior Citizen',
        'Payment Method': 'Payment Method',
        'PaymentMethod': 'Payment Method',
        'Internet Service': 'Internet Service',
        'InternetService': 'Internet Service',
    }
    for source, target in replacements.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
    return df


def clean_data(df):
    df = detect_columns(df)
    df = df.copy()
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)
    if 'Senior Citizen' in df.columns:
        df['Senior Citizen'] = df['Senior Citizen'].replace({0: 'No', 1: 'Yes'})
    return df


def feature_engineering(df):
    df = df.copy()
    df['AvgMonthlySpend'] = df['Total Charges'] / (df['Tenure Months'] + 1)
    df['MonthlyTenureRatio'] = df['Monthly Charges'] / (df['Tenure Months'] + 1)
    return df


def transform_target(df):
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
    return df


def build_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_COLUMNS),
        ('cat', categorical_transformer, CATEGORICAL_COLUMNS),
    ])
    return preprocessor


def prepare_data(df):
    df = clean_data(df)
    df = feature_engineering(df)
    df = transform_target(df)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN]
    return X, y


def get_feature_names(preprocessor):
    numeric_names = NUMERIC_COLUMNS
    categorical_names = []
    if hasattr(preprocessor.named_transformers_['cat']['onehot'], 'get_feature_names_out'):
        categorical_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(CATEGORICAL_COLUMNS).tolist()
    return numeric_names + categorical_names
