import joblib

def load_model(file_path):
    model = joblib.load(file_path)
    print(f'Loaded model from {file_path}')
    return model