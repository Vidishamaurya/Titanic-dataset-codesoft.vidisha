# -----------------------------
# 1. Import necessary libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 2. Load the dataset
# -----------------------------
df = pd.read_csv("advertising.csv.csv")  # change to your uploaded filename
print("First 5 rows of dataset:")
print(df.head())

# -----------------------------
# 3. Check for missing values
# -----------------------------
print("\nMissing values in each column:")
print(df.isnull().sum())

# -----------------------------
# 4. Define features (X) and target (y)
# -----------------------------
X = df[['TV', 'Radio', 'Newspaper']]   # independent variables
y = df['Sales']                        # dependent variable

# -----------------------------
# 5. Split dataset into train and test sets
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Train Linear Regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 7. Predict on test data
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 8. Evaluate model performance
# -----------------------------
print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# -----------------------------
# 9. Visualize Actual vs Predicted Sales
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# -----------------------------
# 10. Display coefficients
# -----------------------------
print("\nModel Coefficients:")
print("Intercept:", model.intercept_)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
