import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from preprocessing import load_data, prepare_data, build_preprocessor, get_feature_names


def build_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'Decision Tree (entropy)': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'Decision Tree (gini)': DecisionTreeClassifier(criterion='gini', random_state=42),
    }


def find_dataset_path():
    base = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(base, '..', 'Telco_customer_churn.xlsx'),
        os.path.join(base, '..', 'Telco_customer_churn.csv'),
        os.path.join(base, '..', 'data', 'Telco_customer_churn.xlsx'),
        os.path.join(base, '..', 'data', 'Telco_customer_churn.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError('Telco customer churn dataset not found. Place Telco_customer_churn.xlsx or Telco_customer_churn.csv in the project root or data folder.')


def build_feature_importances(pipeline):
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = []
    try:
        feature_names = get_feature_names(preprocessor)
    except Exception:
        feature_names = []

    if hasattr(classifier, 'feature_importances_'):
        values = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        values = np.abs(classifier.coef_[0])
    else:
        values = []

    importances = []
    for name, value in zip(feature_names, values):
        importances.append({'feature': name, 'importance': float(value)})
    importances.sort(key=lambda row: row['importance'], reverse=True)
    return importances[:12]


def train_and_export(dataset_path=None):
    if dataset_path is None:
        dataset_path = find_dataset_path()

    df = load_data(dataset_path)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []
    pipelines = {}
    preprocessor = build_preprocessor()

    for name, model in build_models().items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        results.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'probability_average': float(np.mean(y_proba)) if y_proba is not None else None,
        })
        pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by='f1_score', ascending=False)
    best_model_name = results_df.iloc[0]['model']
    best_pipeline = pipelines[best_model_name]

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(output_dir, 'best_model.pkl'))
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    churn_counts = df['Churn Label'].map({'Yes': 'Churn', 'No': 'No Churn'}).value_counts().to_dict() if 'Churn Label' in df.columns else df['Churn'].map({'Yes': 'Churn', 'No': 'No Churn'}).value_counts().to_dict()
    dashboard_payload = {
        'best_model': best_model_name,
        'churn_distribution': churn_counts,
        'model_comparison': results_df.to_dict(orient='records'),
        'feature_importance': build_feature_importances(best_pipeline),
    }

    with open(os.path.join(output_dir, 'dashboard_data.json'), 'w', encoding='utf-8') as file:
        json.dump(dashboard_payload, file, indent=2)

    print(f'Trained models and exported the best model: {best_model_name}')
    print(f'Model files saved to {output_dir}')


if __name__ == '__main__':
    train_and_export()
