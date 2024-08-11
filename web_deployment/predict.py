import pickle
from flask import Flask, request, jsonify
import configparser
import mlflow
import os
from sklearn.feature_extraction import DictVectorizer


def read_config():
    config = configparser.ConfigParser()
    config_path = './config.config'
    print(f"Trying to read config file from: {os.path.abspath(config_path)}")
    config.read('./web_deployment/config.config')

    best_run_id = config['DEFAULT']['best_run_id']
    dv_full_path = config['DEFAULT']['dv_full_path']
    return best_run_id, dv_full_path

best_run_id, dv_full_path = read_config()

with open(dv_full_path, 'rb') as f_in:
    dv = pickle.load(f_in)

model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

def prepare_features(mobile):
    features = {}
    features['battery_power'] = mobile['battery_power']
    features['clock_speed'] = mobile['clock_speed']
    features['fc'] = mobile['fc']
    features['int_memory'] = mobile['int_memory']
    features['m_dep'] = mobile['m_dep']
    features['mobile_wt'] = mobile['mobile_wt']
    features['n_cores'] = mobile['n_cores']
    features['pc'] = mobile['pc']
    features['px_height'] = mobile['px_height']
    features['px_width'] = mobile['px_width']
    features['ram'] = mobile['ram']
    features['sc_h'] = mobile['sc_h']
    features['sc_w'] = mobile['sc_w']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])

app = Flask('price-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    mobile = request.get_json()

    features = prepare_features(mobile)
    pred = predict(features)

    result = {
        'price_range': pred,
        'model_version': best_run_id
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
