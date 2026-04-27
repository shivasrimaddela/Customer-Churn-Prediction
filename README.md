# Customer Churn Prediction Web Project

This folder now contains a simple frontend/backend churn prediction project built from the existing customer churn notebook.

## What is included

- `app.py` - Flask web app and API endpoints
- `src/preprocessing.py` - data loading, cleaning, feature engineering, and preprocessing
- `src/train.py` - model training, comparison, and artifact export
- `src/predict.py` - sample input transformation and churn prediction
- `templates/index.html` - Bootstrap frontend form and dashboard
- `static/style.css` - custom UI styles
- `models/` - generated model artifacts after training
- `data/` - optional dataset location

## Setup

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Place the dataset in this folder:

- `Telco_customer_churn.xlsx`

3. Train the models:

```bash
python src/train.py
```

4. Start the web app:

```bash
python app.py
```

5. Open the browser at `http://localhost:5000`

## Notes

The project uses the dataset schema from the notebook, including `Churn Label`, `Tenure Months`, and `Monthly Charges`.
