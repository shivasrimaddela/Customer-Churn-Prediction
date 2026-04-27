import json
import os
from flask import Flask, render_template, request, jsonify
from src.predict import predict_churn

app = Flask(__name__, template_folder='templates', static_folder='static')

MODEL_DATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'dashboard_data.json')


def load_dashboard_data():
    if not os.path.exists(MODEL_DATA_PATH):
        return None
    with open(MODEL_DATA_PATH, 'r', encoding='utf-8') as file:
        return json.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json() if request.is_json else request.form.to_dict()
    try:
        result = predict_churn(payload)
        return jsonify({'success': True, 'result': result})
    except Exception as error:
        return jsonify({'success': False, 'error': str(error)}), 400


@app.route('/dashboard-data', methods=['GET'])
def dashboard_data():
    data = load_dashboard_data()
    if data is None:
        return jsonify({'success': False, 'error': 'Dashboard data not found. Run python src/train.py first.'}), 404
    return jsonify({'success': True, 'data': data})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
