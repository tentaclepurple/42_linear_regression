import numpy as np
import pandas as pd


def predict(X, m, b):
    """
    Predict the output using the linear model.
    """
    return m * X + b


def ft_mse(y, y_pred):
    """
    Calculate the mean squared error.
    """
    mse = np.mean((y - y_pred) ** 2)
    return mse


def ft_r2_score(y, y_pred):
    """
    Calculate the R2 score.
    """
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def calculate_accuracy(X_train, y_train, m, b, tolerance=0.15):
    """
    Calculate the accuracy of the model on the training set.
    Parameters:
    - X_train: array of input features for training set
    - y_train: array of actual target values for training set
    - m: slope of the model
    - b: intercept of the model
    - tolerance: the acceptable error margin within which a prediction is considered correct
    Returns:
    - accuracy: percentage of correct predictions
    """

    # Convert X_train and y_train to numpy arrays if they are pandas Series
    X_train = X_train.values if isinstance(X_train, pd.Series) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train

    # Make predictions on the training set
    y_pred_train = predict(X_train, m, b)
    
    # Calculate the number of correct predictions
    correct_predictions = 0
    
    print("Mileage\tActual Price\tPredicted Price")
    for i in range(len(y_train)):
        print(f"{X_train[i]}\t{y_train[i]}\t\t{y_pred_train[i]:.2f}")
        if abs(y_pred_train[i] - y_train[i]) <= tolerance * y_train[i]:
            correct_predictions += 1
    
    # Calculate the accuracy as a percentage
    accuracy = (correct_predictions / len(y_train)) * 100
    return accuracy

