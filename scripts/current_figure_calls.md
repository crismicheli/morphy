# Current figure generation calls

This note lists a few script calls that match the current animation workflow for the 2D and 3D phenotype figures.

## 2D figures

```bash
python scripts/animate_scenario.py
```

```bash
python scripts/animate_scenario.py --filter "High porosity"
```

```bash
python scripts/animate_scenario.py --filter "Near-critical asymmetric regime" --fps 12 --max-frames 220 --n-traj 30 --shift-T 1.02 --shift-E 0.92
```

```bash
python scripts/animate_scenario.py --filter "Hypoxic environment" --fps 12 --max-frames 220 --n-traj 28 --shift-E 0.95
```

## 3D figures

```bash
python scripts/animate_scenario_3d_eto_box.py --filter "Intermediate porosity" --show-box
```

```bash
python scripts/animate_scenario_3d_eto_box.py --filter "High porosity" --show-box
```

```bash
python scripts/animate_scenario_3d_eto_box.py --filter "Near-critical asymmetric regime" --show-box --fps 12 --max-frames 240 --n-traj 32 --shift-O 0.92 --shift-E 0.92 --shift-T 1.02
```

```bash
python scripts/animate_scenario_3d_eto_box.py --filter "Hypoxic environment" --show-box --fps 12 --max-frames 220 --n-traj 28 --shift-O 0.92 --shift-E 0.95
```

## Notes

- The 2D script keeps the background quiver arrows.
- Borderline and unstable scenarios are the best choices when the goal is to show boundary-hugging or non-viable trajectories.
- The near-critical asymmetric regime is the best candidate for visually diverse trajectories.
