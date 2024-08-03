import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the file
path = 'houseDataSet.txt'
data = pd.read_csv(path, header=None, names=['area', 'price'])

# Insert extra column
data.insert(0, "ones", 1)
print('data= \n', data.head(5))

# Split x and y
cols = data.shape[1]
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
print('x= \n', x.head(5))
print('y= \n', y.head(5))

# Convert x and y to matrices
x = np.matrix(x.values)
y = np.matrix(y.values)

data.plot(kind='scatter', x='area', y='price', figsize=(5, 5))
plt.show()

# # Compute the cost
# def compute_cost(x, y, theta):
#     hypo = x * theta.T
#     cost = np.sum(np.power(hypo - y, 2))
#     return cost / (2 * len(x))

# print("The initial cost=\n", compute_cost(x, y, theta))  

# # Find the gradient descent    
# def GD(x, y, theta, alpha, iters):
#     m = len(y)
#     cost_list = []
    
#     for i in range(iters):
#         hypo = x * theta.T
#         d_theta = (1 / m) * (x.T * (hypo - y))
#         theta = theta - (alpha * d_theta).T
#         cost = compute_cost(x, y, theta)
#         cost_list.append(cost)
        
#     return theta, cost_list
def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((2, 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1/(2*m))*np.sum(np.square(y_pred - Y))
        d_theta = (1/m)*np.dot(X.T, y_pred - Y)
        theta = theta - learning_rate*d_theta
        cost_list.append(cost)
    return theta, cost_list  
iteration = 100
learning_rate = 0.00000005
theta, cost_list = model(x, y, learning_rate = learning_rate,
iteration = iteration)
print("theta=\n",theta)
print("the cost is =\n",cost_list[0:10])

new_houses = np.array([[1, 1547], [1, 1896], [1, 1934], [1,2800], [1, 3400], [1, 5000]])
for house in new_houses :
    print("Our model predicts the price of house with",house[1], 
          "sq. ft. area as : $", np.round(np.dot(house, theta)))

# get best fit line
x = np.linspace(data.area.min(), data.area.max(), 100)
# print('x \n',x)
# print('g \n',theta)

f = theta[0, 0] + (theta[1, 0] * x)
# print('f \n',f)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.area, data.price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('area')
ax.set_ylabel('price')
ax.set_title('Predicted price vs. square area')

#  get the cost error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iteration), cost_list, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
