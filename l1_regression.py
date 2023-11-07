import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.replace('?', np.nan)
    data = data.drop('car name', axis=1)
    data = data.apply(lambda x: x.fillna(x.median()), axis=0)
    return data

def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[[target_column]]
    return X, y

def create_polynomial_features(X, degree=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, interaction_only=True)
    X_poly = poly.fit_transform(X_scaled)
    return X_poly

def train_ridge_regression(X_train, y_train, alpha=0.3):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    return ridge

def evaluate_model(ridge, X_train, y_train, X_test, y_test):
    train_r2 = ridge.score(X_train, y_train)
    test_r2 = ridge.score(X_test, y_test)
    return train_r2, test_r2

def plot_predictions(y_train, y_train_pred, y_test, y_test_pred):
    plt.figure(figsize=(10, 6))

    # Training set
    plt.scatter(y_train, y_train_pred, c='blue', marker='o', label='Training data')
    # Test set
    plt.scatter(y_test, y_test_pred, c='green', marker='s', label='Test data')

    plt.xlabel('True MPG')
    plt.ylabel('Predicted MPG')
    plt.title('True vs. Predicted MPG')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

def main():
    file_path = 'auto-mpg.csv'
    data = load_data(file_path)
    data = preprocess_data(data)
    X, y = split_data(data, 'mpg')
    X_poly = create_polynomial_features(X, degree=2)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.30, random_state=1)
    ridge = train_ridge_regression(X_train, y_train, alpha=0.3)
    
    train_r2, test_r2 = evaluate_model(ridge, X_train, y_train, X_test, y_test)
    print("Training R^2:", train_r2)
    print("Test R^2:", test_r2)
    
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    plot_predictions(y_train, y_train_pred, y_test, y_test_pred)

if __name__ == "__main__":
    main()