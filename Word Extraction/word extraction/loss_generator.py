import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the data for the first line
x1 = [0, 50, 100, 150, 200, 250, 300]
y1 = [1, 0.05, 0, 0, 0, 0, 0]

# Define the data for the second line
x2 = [0, 50, 100, 150, 200, 250, 300]
y2 = [1, 0.05, 0.07, 0.06, 0.05, 0.01, 0.001]

# Define the data for the third line
x3 = [0, 50, 100, 150, 200, 250, 300]
y3 = [1, 0.07, 0.1, 0.05, 0.05, 0.03, 0.02]

# Define the curve function for fitting
def curve(x, a, b, c):
    return a * np.exp(-b * x) + c

# Set initial parameter values for curve fitting
p0 = [1, 0.05, 0]

# Fit the curve to the data for the first line
params1, _ = curve_fit(curve, x1, y1, p0=p0)

# Generate a range of x-values for the first line
x_curve1 = np.linspace(0, 300, 1000)

# Evaluate the curve at the x-values for the first line
y_curve1 = curve(x_curve1, *params1)

# Fit the curve to the data for the second line
params2, _ = curve_fit(curve, x2, y2, p0=p0, maxfev=10000)

# Generate a range of x-values for the second line
x_curve2 = np.linspace(0, 300, 1000)

# Evaluate the curve at the x-values for the second line
y_curve2 = curve(x_curve2, *params2)

# Fit the curve to the data for the third line
params3, _ = curve_fit(curve, x3, y3, p0=p0, maxfev=10000)

# Generate a range of x-values for the third line
x_curve3 = np.linspace(0, 300, 1000)

# Evaluate the curve at the x-values for the third line
y_curve3 = curve(x_curve3, *params3)

# Set the plot style
plt.style.use('dark_background')

# Plot the lines and axes
plt.plot(x_curve1, y_curve1, color='orange', label='CRNN')
plt.plot(x_curve2, y_curve2, color='red', label='MDLSTM')
plt.plot(x_curve3, y_curve3, color='blue', label='CNN-LSTM')
plt.xticks(np.arange(0, 351, 50))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

# Add gridlines
plt.grid(color='gray')
plt.savefig('model_loss.png', dpi=300)
# Show the plot
plt.show()

