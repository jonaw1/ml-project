from sklearn.model_selection import train_test_split
from src.helpers import decimal_to_percentage
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

def standardize_data(*data):
    scaler = StandardScaler()
    data = (scaler.fit_transform(d) for d in data)
    print('Standardized the data')
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f'Split the data into training and testing sets ({decimal_to_percentage(1 - test_size)}/{decimal_to_percentage(test_size)})')
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print('Trained a random forest regressor')
    return model

def save_model(model, type):
    joblib.dump(model, f'models/{type}_model.joblib')
    print('Saved the model to models/model.joblib')

