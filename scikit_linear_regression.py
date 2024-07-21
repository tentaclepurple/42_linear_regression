import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


file_path = './data.csv'
data = pd.read_csv(file_path)

# Paso 1: Preparar los datos
X = data[['km']]
y = data['price']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 2: Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 3: Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

intercept = model.intercept_
slope = model.coef_[0]

print("Linear Regression Model with scikit-learn")
print("Intercept:", intercept)
print("Slope:", slope)
print(f"Mean squared error: {mse:.2f}, R2 score: {r2:.2f}")

# Predict the price for 100000 km
predicted_price = intercept + slope * 100000
print("Predicted price for 100000 km:", predicted_price)