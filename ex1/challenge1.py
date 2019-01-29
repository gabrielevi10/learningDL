import pandas as pds
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy import array

# reading data
dataframe = pds.read_csv('challenge_dataset.txt', names =['first', 'second'])
x_values = dataframe[['first']]
y_values = dataframe[['second']]

# train the model
reg = linear_model.LinearRegression()
reg.fit(x_values, y_values)

# visualize results
# plt.scatter(x_values, y_values)
# plt.plot(x_values, reg.predict(x_values))
# plt.show()

# Predicting with 22.203 and expecting 24.147
arr = array([22.203]).reshape(-1, 1)
print("Expected value: " + str(24.147))
print("Model predict: " + str(reg.predict(arr)[0][0]))
print("Error: " + str(abs(reg.predict(arr)[0][0] - 24.147)))
