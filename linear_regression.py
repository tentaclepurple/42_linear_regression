import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from mse import ft_mse, ft_r2_score, calculate_accuracy


def split_train_test(data, test_ratio):
    """
    Split the data into training and test sets.
    returns
    X_train: training input features
    X_test: test input features
    y_train: training target variable
    y_test: test target variable
    """
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices]['km'], data.iloc[test_indices]['km'], data.iloc[train_indices]['price'], data.iloc[test_indices]['price']


def load_data(filename):
    """
    Load the data from a CSV file.
    """
    data = pd.read_csv(filename)
    X = data['km'].values
    y = data['price'].values
    return X, y


def predict(X, m, b):
    """
    Predict the output using the linear model.
    """
    return m * X + b


def gradient_descent(X, y, learning_rate, iterations):
    """
    Perform gradient descent to find the optimal values of m and b.
    parameters:
    X: array of input features
    y: target variable
    m: slope of the line
    b: intercept of the line
    learning_rate: step size for the update
    iterations: number of iterations

    how it works:
    1. Calculate the predicted values using
        the current values of m and b
    2. Compute the gradients of the loss with respect to m and b
    3. Update the values of m and b using the gradients
    4. Repeat the process for the specified number of iterations

    """
    m = 0
    b = 0
    n = len(X)
    for _ in range(iterations):
        y_pred = predict(X, m, b)
        m_gradient = -(2/n) * np.sum(X * (y - y_pred))
        b_gradient = -(2/n) * np.sum(y - y_pred)
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient
    return m, b


def plot_data(X, y, m, b):
    """
    Plot the data and the linear regression line.
    """
    plt.scatter(X, y, color='blue')
    plt.plot(X, m*X + b, color='red')
    plt.xlabel('Kms')
    plt.ylabel('Price')
    plt.savefig('linear_regression.png')


def ft_linear_regression():
    """
    Perform linear regression on the data.
    """
    try:
        df = pd.read_csv('data.csv')
        print("Training the model...")
    except Exception as e:
        print("An error occured: ", e)
        return		

    X = df['km'].values
    y = df['price'].values

    # divide the data into training and test sets
    X_train, X_test, y_train, y_test = split_train_test(df, 0.2)
    
    # Normalize X_train y X_test
    X_train_normalized = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test_normalized = (X_test - np.mean(X_train)) / np.std(X_train)

    # Hyperparameters
    learning_rate = 0.01
    iterations = 10000

    # Train the model
    m_norm, b_norm = gradient_descent(X_train_normalized, y_train, learning_rate, iterations)

    # Convert the normalized coefficients back to the original scale
    m = m_norm / np.std(X_train)
    b = b_norm - m_norm * np.mean(X_train) / np.std(X_train)


    # Evaluate the model on the test set
    y_pred = predict(X_test, m, b)

    # calculate the mean squared error
    
    plot_data(X, y, m, b)
    print("Plot saved as linear_regression.png")
    
    print(f"Slope/theta1 (m): {m}")
    print(f"Intercept/theta0  (b): {b:.2f}")

    accuracy = calculate_accuracy(X_test, y_test, m, b)
    mse = ft_mse(y_test, y_pred)
    r2 = ft_r2_score(y_test, y_pred)
    
    #theta1 = m = slope
    #theta0 = b = intercept

    print(f"Mean squared error: {mse:.2f}")
    print(f"R2 score: {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


    params = (m, b)
    try:
        with open('params.pkl', 'wb') as f:
            pickle.dump(params, f)
    except Exception as e:
        print("An error occured: ", e)
        return

    return m, b


if __name__ == '__main__':
    ft_linear_regression()
