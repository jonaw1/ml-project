import joblib
from dotenv import load_dotenv
import os
import pandas as pd
from src.preprocessing import handle_missing_data, encode_categories
import json

load_dotenv()

PRODUCTION_FOLDER_PATH = os.environ.get('PRODUCTION_FOLDER').replace('/', os.path.sep)
RAW_DATA_PATH = os.environ.get('RAW_DATA_PATH').replace('/', os.path.sep)
PREPROCESSED_DATA_PATH = os.environ.get('PREPROCESSED_DATA_PATH').replace('/', os.path.sep)
PREDICTIONS_PATH = os.environ.get('PREDICTIONS_FOLDER').replace('/', os.path.sep)

def load_model(file_path):
    model = joblib.load(file_path)
    print(f'Loaded model from {file_path}')
    return model

def load_production_data():
    files = [os.path.join(PRODUCTION_FOLDER_PATH, f) for f in os.listdir(PRODUCTION_FOLDER_PATH) if f.endswith('.csv')]
    data = {f.split(os.sep)[-1].rstrip('.csv'):pd.read_csv(f) for f in files}
    print(f"Loaded {len(files)} production dataset{'s' if len(files) > 1 else ''}")
    return data

def preprocess_data(data):
    try:
        data = {key:handle_missing_data(d, silent=True) for key, d in data.items()}
    except Exception as e:
        print('Error while handling missing data:', e, '❌')
        return
    try:
        data = {key:encode_categories(d, silent=True) for key, d in data.items()}
    except Exception as e:
        print('Error while encoding categories:', e, '❌')
        return
    # Add missing hot-encoded columns
    all_columns = pd.read_csv(PREPROCESSED_DATA_PATH).columns.tolist()
    all_columns.remove('Verkaufspreis')
    data = {key:d.reindex(columns=all_columns, fill_value=0) for key, d in data.items()}
    data = {key:d.sort_index(axis=1) for key, d in data.items()}
    return data

def validate_data(data):
    train_columns = pd.read_csv(RAW_DATA_PATH).columns.tolist()
    train_columns.remove('Verkaufspreis')

    for dataset in data.values():
        if set(train_columns) != set(dataset.columns):
            return False
    return True

def create_predictions(model):
    try:
        data = load_production_data()
    except Exception as e:
        print('Error while loading production data:', e, '❌')
        return
    predicted_files = [f.rstrip('_predicted.json') for f in os.listdir(PREDICTIONS_PATH) if f.endswith('.json')]
    matching_files = [key + '.csv' for key in data.keys() if key in predicted_files]
    if matching_files:
        print(', '.join(matching_files), 'already predicted')
    files_left = [key + '.csv' for key in data.keys() if key not in predicted_files]
    if not files_left:
        print('Aborting - all files already predicted ❌')
        return
    data = {key:d for key, d in data.items() if key + '.csv' in files_left}
    if not validate_data(data):
        print('Error: Dataset columns do not match required columns ❌')
        return
    data = preprocess_data(data)
    if not data:
        return
    print('Predicting', ', '.join(files_left) + '...')
    predictions = {key:model.predict(d) for key, d in data.items()}
    for key, prediction in predictions.items():
        json.dump(prediction.tolist(), open(os.path.join(PREDICTIONS_PATH, f'{key}_predicted.json'), 'w'))
    print('Predictions saved to', PREDICTIONS_PATH)