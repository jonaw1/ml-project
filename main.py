from src.preprocessing import load_data, preprocess_data, save_data
from src.train_model import split_data, train_random_forest, save_model
from src.predict import load_model, create_predictions
import os
from dotenv import load_dotenv

load_dotenv()

TRAIN_FILE_PATH = os.environ.get('RAW_DATA_PATH').replace('/', os.path.sep)
PREPROCESSED_FILE_PATH = os.environ.get('PREPROCESSED_DATA_PATH').replace('/', os.path.sep)
TARGET_COLUMN = os.environ.get('TARGET_COLUMN')
MODEL_PATH = os.environ.get('MODEL_PATH').replace('/', os.path.sep)

def preprocess():
  data = load_data(TRAIN_FILE_PATH)
  data = preprocess_data(data)
  save_data(data)

def train_model():
  data = load_data(PREPROCESSED_FILE_PATH)
  X_train, X_test, y_train, y_test = split_data(data, TARGET_COLUMN)
  model = train_random_forest(X_train, y_train, X_test, y_test)
  save_model(model)

def predict():
  model = load_model(MODEL_PATH)
  create_predictions(model)

if __name__ == '__main__':
  # preprocess()
  # train_model()
  predict()
