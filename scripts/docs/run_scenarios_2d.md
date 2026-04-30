# run_scenarios_2d.py

## Example calls

```bash
python run_scenarios_2d.py
python run_scenarios_2d.py --filter stable
python run_scenarios_2d.py --save figures/all_scenarios_2d.png --n-traj 40
```

This script runs a sweep over multiple predefined scenarios and produces a 2D phase-plane summary based on the simulated dynamics. It filters the scenario list optionally, runs one ensemble simulation per selected scenario, prints a summary table, and optionally saves a multi-panel figure.

The script uses `plotting.plot2d_helpers` for reusable presentation logic, including the summary table formatter and the wrapper used to save or show the all-scenarios 2D figure.

Important 2D versus 3D distinction: the 2D sweep only shows dynamic trajectories in the E-T phase plane. It does not estimate or display taxonomy labels for the sweep view.

In contrast, 3D sweeps combine both dynamic trajectories and estimated taxonomy information. The 3D sweep workflow therefore produces dynamic E-T-O trajectory animations together with classifier-colored taxonomy plots.
