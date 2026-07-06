import numpy as np
import matplotlib.pyplot as plt

# Data
porosity = np.array([0.75, 0.40, 0.15])
min_dist = np.array([0.26, 0.27, 0.05])

# Increasing logistic in mathematical x;
# after inverting the x-axis, it appears decreasing visually.
def logistic(x, a, b, k, x0):
    return a + b / (1 + np.exp(-k * (x - x0)))

# Tuned so the curve passes near the three points
a = 0.045   # lower asymptote
b = 0.225   # amplitude
k = 30.0    # steepness
x0 = 0.28   # inflection point

x_fit = np.linspace(0.15, 0.75, 500)
y_fit = logistic(x_fit, a, b, k, x0)

fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Curve and points
ax.plot(x_fit, y_fit, color='tab:blue', linewidth=2.5, label='Sigmoid interpolation')
ax.scatter(porosity, min_dist, color='black', s=45, zorder=3, label='Porosity values')

# Optional point labels
for x, y in zip(porosity, min_dist):
    ax.text(x, y + 0.008, f"({x:.2f}, {y:.2f})", ha='center', fontsize=8)

# Linear axes
ax.set_xscale('linear')
ax.set_yscale('linear')

# Reverse x-axis only
ax.invert_xaxis()

ax.set_xlabel('Porosity')
ax.set_ylabel('Minimum signed distance from bounds')
ax.set_title('Porosity sweep and viability distance')
ax.grid(alpha=0.3)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
