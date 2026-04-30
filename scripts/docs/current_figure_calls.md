# Current figure generation calls

This note lists a few script calls that match the current animation workflow for the 2D and 3D phenotype figures.

## 2D figures

```bash
python scripts/animate_scenario_2d.py
```

```bash
python scripts/animate_scenario_2d.py --filter "High porosity"
```

```bash
python scripts/animate_scenario_2d.py --filter "Near-critical asymmetric regime" --fps 12 --max-frames 220 --n-traj 30 --shift-T 1.02 --shift-E 0.92
```

```bash
python scripts/animate_scenario_2d.py --filter "Hypoxic environment" --fps 12 --max-frames 220 --n-traj 28 --shift-E 0.95
```

## 3D figures

```bash
python scripts/animate_scenario_3d.py --filter "Intermediate porosity" --show-box
```

```bash
python scripts/animate_scenario_3d.py --filter "High porosity" --show-box
```

```bash
python scripts/animate_scenario_3d.py --filter "Near-critical asymmetric regime" --show-box --fps 12 --max-frames 240 --n-traj 32 --shift-O 0.92 --shift-E 0.92 --shift-T 1.02
```

```bash
python scripts/animate_scenario_3d.py --filter "Hypoxic environment" --show-box --fps 12 --max-frames 220 --n-traj 28 --shift-O 0.92 --shift-E 0.95
```
## 2D vs 3D viability plots differences

The 2D phase-plane plots show ensembles in the \((E, T)\) projection with trajectories started from a Gaussian cloud around a central initial condition, which may lie inside or outside the admissible bounds; trajectories are then colored blue or red depending on whether they remain viable over the simulation window. The 3D \((E, T, O)\) plots instead sample initial conditions by random perturbations around a (possibly shifted) center but **reject** any samples that start outside the admissible \((C, T, E, O)\) box, so all trajectories begin inside the viability region; each trajectory is then integrated in full 4D state space, with its 3D projection animated and recolored when it exits via any violated variable.  

For the **2D case**, the main exposed controls for the initial cloud are the center and ensemble size, for example `x0_center = DEFAULT_SIM["x0_center"]`, `--n-traj 30`, `--shift-T 1.15`, and `--shift-E 1.20`, which move the cloud center in the hidden full state and change how many trajectories appear in the \((E,T)\) projection. In that setup, the cloud dimension shown on the figure is 2D because only \(E\) and \(T\) are plotted, but each trajectory still comes from a 4-variable initial point \([C,T,E,O]\) perturbed by the default noise scale `(0.03, 0.03, 0.03, 0.05)` inside `sample_initial_conditions(...)`.

For the **3D case**, the script exposes both the center shifts and the visible trajectory-cloud size more explicitly, for example `--n-traj 30 --shift-T 1.5 --shift-E 1.7 --shift-O 1.0`, with scenario-dependent noise scales such as `(0.04, 0.08, 0.08, 0.06)` for `borderline` and `(0.05, 0.10, 0.10, 0.07)` for `unstable` scenarios. Here the plotted cloud is genuinely 3D in the displayed coordinates \((E,T,O)\), but the admissibility check is still performed in the full 4D state \((C,T,E,O)\), so changing `n_traj`, the shifted center, or the noise scale changes both the spread of the visible cloud and the probability that enough viable starting points can be sampled.

## Notes

- The 2D script keeps the background quiver arrows.
- Borderline and unstable scenarios are the best choices when the goal is to show boundary-hugging or non-viable trajectories.
- The near-critical asymmetric regime is the best candidate for visually diverse trajectories.
