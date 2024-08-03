import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Sample data
xdata = np.array([0, 1, 2, 3, 4, 5])
ydata = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

# Define the function to fit
def func(x, a, b, c):
    return a * np.sin(b * x) + c

# Perform the curve fit
popt, pcov = curve_fit(func, xdata, ydata)

# Extract the fitted parameters
a, b, c = popt
print(popt)

# Generate y values using the fitted function
xfit = np.linspace(-1, 7, 50);
yfit = func(xfit, a, b, c)

# Plot the results
plt.scatter(xdata, ydata, label='Data')
plt.plot(xfit, yfit, label='Fitted curve')
plt.legend()
plt.show()