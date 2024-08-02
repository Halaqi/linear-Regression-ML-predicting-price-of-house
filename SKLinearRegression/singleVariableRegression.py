import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# loading the data
data = pd.read_csv(r"D:\programming\Machine learning\findHousePrice\houseDataSet.txt")

# testing the missing values
missing_values = SimpleImputer(missing_values=np.nan, strategy='mean')
missing_values.fit(data)
data = missing_values.transform(data)

# preparing x and y
col = data.shape[1]
X = data[:,0:col-1]
y = data[:,col-1:col]

print(X)
# scaling the data
sc = StandardScaler()
X = sc.fit_transform(X)

# spliting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# calling the linear regression class
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# printing the accuracy between training the testing data
score_train = regressor.score(X_train, y_train)
print('training accuracy: ',score_train*100 ,'%')

score_test = regressor.score(X_test, y_test)
print('testing accuracy: ',score_test*100 ,'%')

# predicting the testing data
y_pre1 = regressor.predict(X)
y_pre = regressor.predict(X_test)

print('the testing values:\n ',np.round(y_test[:5],2))
print('the predicted values:\n ',np.round(y_pre[:5],2))


# printing t the error between actual y and predicted y 
error = mean_absolute_error(y_test,y_pre )
print(error)


error2 = mean_squared_error(y_test,y_pre )
print(error2)

# plot the whole data
plt.scatter(X, y,c='green')
plt.plot(X,y_pre1 )
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend()
plt.title('Linear Regression Best Fit Line')
plt.show()


# plot the testing data
plt.scatter(X_test, y_test,c='green')
plt.plot(X_test,y_pre )
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend()
plt.title('Linear Regression Best Fit Line')
plt.show()


# Parameters for gradient descent
learning_rate = 0.01
n_iterations = 10000
m = X_train.shape[0] 

# Initialize weights
theta = np.zeros((X_train.shape[1], 1))

# Store the cost at each iteration
cost_history = []

# Gradient Descent
for i in range(n_iterations):
    gradients = (1/m) * X_train.T.dot(X_train.dot(theta) - y_train)
    theta = theta - learning_rate * gradients
    
    # Compute the cost (mean squared error)
    cost = (1/m) * np.sum((X_train.dot(theta) - y_train) ** 2)
    cost_history.append(cost)

# Plotting the cost over iterations
plt.plot(range(n_iterations), cost_history, color='blue', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction over Iterations')
plt.show()
