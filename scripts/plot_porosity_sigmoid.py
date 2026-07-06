import numpy as np
import matplotlib.pyplot as plt

# Data
porosity = np.array([0.75, 0.40, 0.15])
min_dist = np.array([0.26, 0.27, 0.05])

# Decreasing sigmoid:
# y = a + b / (1 + exp(k * (x - x0)))
# For k > 0 and b > 0, this decreases with x.
def dec_sigmoid(x, a, b, k, x0):
    return a + b / (1 + np.exp(k * (x - x0)))

# Hand-chosen parameters to roughly match:
# - y ~ 0.05 at x = 0.15
# - y ~ 0.27 at x = 0.40
# - y ~ 0.26 at x = 0.75 (almost flat/high-porosity side)
# You can tweak these as needed.
a = 0.02    # baseline
b = 0.30    # amplitude
k = 10.0    # slope (steepness)
x0 = 0.35   # midpoint

# Generate smooth curve
x_fit = np.linspace(0.1, 0.8, 400)
y_fit = dec_sigmoid(x_fit, a, b, k, x0)

# Plot
plt.figure(figsize=(6, 4))

# Smooth sigmoid
plt.plot(x_fit, y_fit, color='tab:blue', linewidth=2, label='Decreasing sigmoid')

# Data points
plt.scatter(porosity, min_dist, color='black', zorder=3, label='Simulated scenarios')
for x, y in zip(porosity, min_dist):
    plt.text(x, y + 0.01, f"({x:.2f}, {y:.2f})", ha='center', fontsize=8)

plt.xlabel('Porosity')
plt.ylabel('Minimum signed distance from bounds')
plt.title('Porosity sweep and viability distance (sigmoid interpolation)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
