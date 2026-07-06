import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data
porosity = np.array([0.75, 0.40, 0.15])
min_dist = np.array([0.26, 0.27, 0.05])

# Define a decreasing sigmoid:
# y = a + b / (1 + exp(k * (x - x0)))
# For k > 0 and b > 0, this decreases with x.
def dec_sigmoid(x, a, b, k, x0):
    return a + b / (1 + np.exp(k * (x - x0)))

# Initial guess for parameters: [a, b, k, x0]
p0 = [0.0, 0.3, 10.0, 0.4]

# Fit the sigmoid to the three points
params, _ = curve_fit(dec_sigmoid, porosity, min_dist, p0=p0, maxfev=10000)
a_fit, b_fit, k_fit, x0_fit = params

# Generate smooth curve for visualization
x_fit = np.linspace(0.1, 0.8, 200)
y_fit = dec_sigmoid(x_fit, a_fit, b_fit, k_fit, x0_fit)

# Plot
plt.figure(figsize=(6, 4))
plt.scatter(porosity, min_dist, color='black', zorder=3, label='Simulated scenarios')
plt.plot(x_fit, y_fit, color='tab:blue', linewidth=2, label='Decreasing sigmoid fit')

for x, y in zip(porosity, min_dist):
    plt.text(x, y + 0.01, f"({x:.2f}, {y:.2f})", ha='center', fontsize=8)

plt.xlabel('Porosity')
plt.ylabel('Minimum signed distance from bounds')
plt.title('Porosity sweep and viability distance (sigmoid interpolation)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
