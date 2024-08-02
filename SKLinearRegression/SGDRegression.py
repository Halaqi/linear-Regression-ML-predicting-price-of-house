import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv(r"D:\programming\Machine learning\findHousePrice\houseDataSet.txt")

# Handle missing values
missing_values = SimpleImputer(missing_values=np.nan, strategy='mean')
missing_values.fit(data)
data = missing_values.transform(data)
print(data.shape)

# Split features and target variable
col = data.shape[1]
X = data[:, 0:col-1]
y = data[:, col-1]

# Standardize features
sc = StandardScaler()
X = sc.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Initialize the SGD Regressor
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01)

# Fit the model and collect the loss (cost) at each iteration
loss_history = []
for _ in range(1000):
    sgd_regressor.partial_fit(X_train, y_train)
    y_train_pred = sgd_regressor.predict(X_train)
    loss = mean_squared_error(y_train, y_train_pred)
    loss_history.append(loss)

# Plotting the loss over iterations
plt.plot(range(1, 1001), loss_history, color='blue', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Loss (MSE)')
plt.title('Loss Reduction over Iterations')
plt.show()

# Predict on the test set
y_pred = sgd_regressor.predict(X_test)

# Calculate mean squared error on test set
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {test_mse}")

# Plot the data points and the best fit line
plt.scatter(X_test[:, 0], y_test, color='green', label='Data Points')  # Assuming the first feature for simplicity
plt.plot(X_test[:, 0], y_pred, color='blue', linewidth=2, label='Best Fit Line')
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend()
plt.title('Linear Regression Best Fit Line')
plt.show()
