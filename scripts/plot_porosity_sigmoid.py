import numpy as np
import matplotlib.pyplot as plt

# Data
porosity = np.array([0.75, 0.40, 0.15])
min_dist = np.array([0.26, 0.27, 0.05])

# We want the x-axis to decrease from left to right.
# To make a visually decreasing sigmoid on that reversed axis,
# define the sigmoid in terms of xr = -x.
def dec_sigmoid_reversed_axis(x, a, b, k, x0):
    xr = -x
    x0r = -x0
    return a + b / (1 + np.exp(-k * (xr - x0r)))

# Hand-tuned parameters so the curve lands near the 3 points
a = 0.045   # lower asymptote
b = 0.225   # amplitude
k = 35.0    # steepness
x0 = 0.22   # inflection point in original porosity coordinates

# Smooth curve
x_fit = np.linspace(0.15, 0.75, 500)
y_fit = dec_sigmoid_reversed_axis(x_fit, a, b, k, x0)

# Plot
plt.figure(figsize=(6.5, 4.2))
plt.plot(x_fit, y_fit, color='tab:blue', linewidth=2.5, label='Decreasing sigmoid')
plt.scatter(porosity, min_dist, color='black', s=45, zorder=3, label='Porosity scenarios')

# Annotate points
for x, y in zip(porosity, min_dist):
    plt.text(x, y + 0.008, f"({x:.2f}, {y:.2f})", ha='center', fontsize=8)

# Reverse x-axis
plt.xlim(0.75, 0.15)

plt.xlabel('Porosity')
plt.ylabel('Minimum signed distance from bounds')
plt.title('Porosity sweep and viability distance')
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
