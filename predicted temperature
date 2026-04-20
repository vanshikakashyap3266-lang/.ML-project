import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Input: Day number
X = [[1], [2], [3], [4], [5]]

# Output: Temperature
y = [20, 22, 23, 25, 27]

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict temperature for day 6
predicted_temp = model.predict([[6]])
print("Predicted Temperature on Day 6:", predicted_temp[0])

# Predict temperature for all given days for plotting
y_pred = model.predict(X)

# Plot actual vs predicted
plt.scatter(X, y, color='blue', label='Actual Temperature')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Day vs Temperature')
plt.legend()
plt.show()
