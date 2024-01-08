from src.preprocessing import load_data, preprocess_data, save_data
from src.train_model import split_data, standardize_data, train_random_forest, save_model
from src.predict import load_model
import os

TRAIN_FILE_PATH = 'data/train.csv'.replace('/', os.path.sep)
PREPROCESSED_FILE_PATH = 'data/preprocessed.csv'.replace('/', os.path.sep)
TARGET_COLUMN = 'Verkaufspreis'

def preprocess():
  data = load_data(TRAIN_FILE_PATH)
  data = preprocess_data(data)
  save_data(data)

def train_model():
  data = load_data(PREPROCESSED_FILE_PATH)
  X_train, X_test, y_train, y_test = split_data(data, TARGET_COLUMN)
  X_train, X_test = standardize_data(X_train, X_test)
  model = train_random_forest(X_train, y_train)
  print(model.score(X_test, y_test))
  save_model(model, 'random_forest')

def predict(minimal=False):
  model = load_model('models/model.joblib')

if __name__ == '__main__':
  preprocess()
  train_model()
  # predict()
