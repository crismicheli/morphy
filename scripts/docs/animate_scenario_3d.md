# animate_scenario_3d Script

## Example calls

```bash
python animate_scenario_3d.py --filter "Intermediate porosity"
python animate_scenario_3d.py --filter "Intermediate porosity" --show-box --c-stat median
python animate_scenario_3d.py --filter "Unstable" --fps 12 --max-frames 120 --elev 30 --azim -45
```

This script generates a 3D animated GIF for one selected scenario in the E-T-O phenotype space. It selects the scenario, runs a single-scenario ensemble simulation through the shared scenario helper, and sends the result to `plotting.plot_helpers.save_trajectory_animation`.

The animation helper renders the moving 3D trajectories, colors trajectory segments according to viability-box membership, and can display a numerical summary of the C coordinate across the ensemble.

This script focuses on dynamic trajectories rather than taxonomy labels. It is therefore the dynamic-only 3D companion to the static taxonomy plot script.

For sweep workflows, the 3D logic is broader than the 2D one: 3D sweeps are intended to show both estimated taxonomy states and dynamic trajectories, whereas 2D sweeps show only the dynamic trajectories.
