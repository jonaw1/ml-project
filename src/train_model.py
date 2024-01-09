from sklearn.model_selection import train_test_split
from src.helpers import decimal_to_percentage
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv
import json

load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH').replace('/', os.path.sep)
BEST_PARAMS_PATH = os.getenv('BEST_PARAMS_PATH').replace('/', os.path.sep)
best_params = json.load(open(BEST_PARAMS_PATH))

def standardize_data(*data):
    scaler = StandardScaler()
    data = tuple(scaler.fit_transform(d) for d in data)
    print('Standardized the data')
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f'Split the data into training and testing sets ({decimal_to_percentage(1 - test_size)}/{decimal_to_percentage(test_size)})')
    return X_train, X_test, y_train, y_test

def show_metrics(model, X_test, y_test):
    mse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    r2 = model.score(X_test, y_test)
    print('Mean error:',  '{:.4f}'.format(mse).rstrip('0').rstrip('.'))
    print('R² score:', '{:.4f}'.format(r2).rstrip('0').rstrip('.'))
    
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(
        random_state=42,
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        bootstrap=best_params['bootstrap'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print('Trained a random forest model')
    show_metrics(model, X_test, y_test)
    return model

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('Trained a linear regression model')
    show_metrics(model, X_test, y_test)
    return model

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f'Saved the model to {MODEL_PATH}')
