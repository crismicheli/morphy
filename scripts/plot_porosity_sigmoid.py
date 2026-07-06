from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
out_dir = script_dir / "../output/porosity_sweep"
out_dir.mkdir(parents=True, exist_ok=True)

porosity = np.array([0.75, 0.40, 0.15])
min_dist = np.array([0.26, 0.27, 0.05])

def logistic(x, a, b, k, x0):
    return a + b / (1 + np.exp(-k * (x - x0)))

a, b, k, x0 = 0.045, 0.225, 30.0, 0.28
x_fit = np.linspace(0.15, 0.75, 500)
y_fit = logistic(x_fit, a, b, k, x0)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(x_fit, y_fit, color='tab:blue', linewidth=2.5)
ax.scatter(porosity, min_dist, color='black', s=45, zorder=3)

ax.invert_xaxis()
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.set_xlabel('Porosity')
ax.set_ylabel('Minimum signed distance from bounds')
ax.set_title('Porosity sweep and viability distance')
ax.grid(alpha=0.3)
plt.tight_layout()

fig.savefig(out_dir / "porosity_sweep_sigmoid.png", dpi=300)
fig.savefig(out_dir / "porosity_sweep_sigmoid.pdf")
