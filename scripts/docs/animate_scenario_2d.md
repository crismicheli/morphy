# animate_scenario_2d Script

## Example calls

```bash
python animate_scenario_2d.py --filter "Intermediate porosity"
python animate_scenario_2d.py --filter "Intermediate porosity" --fps 12 --max-frames 200
python animate_scenario_2d.py --filter "Unstable" --shift-T 1.2 --shift-E 1.1 --hide-box
```

This script generates a 2D animated GIF for one selected scenario in the E-T phase plane. It uses the shared scenario helper to choose the scenario, build the initial conditions, and run the ensemble simulation.

The actual rendering is delegated to `plotting.plot2d_helpers.save_et_animation`. That helper draws the phase-plane vector field, overlays the viability rectangle when requested, and animates the simulated trajectories over time.

The animation shows dynamic behavior only. It does not estimate taxonomy labels or classifier states.

This makes the script suitable for visualizing how one scenario moves in the reduced 2D phase plane, as opposed to the 3D taxonomy scripts that can add classifier-based phenotype interpretation.
