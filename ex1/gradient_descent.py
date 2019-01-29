import numpy as np
import pandas as pds
from sklearn import linear_model
import matplotlib.pyplot as plt

def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) **2
    return totalError / float(len(points)) 

def step_gradient(b_current, m_current, points, learning_rate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]

def run():
    points = np.genfromtxt('data.csv', delimiter=',')

    # hyperparameters
    learning_rate = 0.0001
    
    # y = mx + b 
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)
    
    # for visualization
    line = []
    x_values = []
    y_values = []

    for i in range(len(points)):
        y = m * points[i, 0] + b
        line.append(y)
        x_values.append(points[i,0])
        y_values.append(points[i,1])

    plt.scatter(x_values, y_values)
    plt.plot(x_values, line)
    plt.show()

if __name__ == '__main__':
    run()