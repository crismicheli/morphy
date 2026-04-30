# Scenarios Sweep Plots

This note describes the dependencies and execution logic of `run_scenarios_3d.py`, which runs a batch sweep over all scenarios belonging to one expected class and produces both taxonomy-colored 3D scatter plots and E-T-O trajectory animations.

## Example calls

Typical calls look like this:

```bash
python run_scenarios_3d.py unstable

python run_scenarios_3d.py boundary   --classifier-type temporal   --show-box

python run_scenarios_3d.py stable   --classifier-type state_machine   --n-traj 40   --stride 4   --fps 12   --max-frames 160
```

These runs all follow the same sweep workflow, but they can change the classifier backend, trajectory count, taxonomy-point subsampling, animation settings, and output location.

## Main dependencies

The script depends on shared plotting helpers from `plotting.plot_helpers`, scenario utilities from `plotting.scenario_helpers`, and classifier selection through `classifiers.classifier_dispatch`. Together, these modules remove the need to duplicate the plotting, animation, and classifier-wiring logic inside the sweep script itself.

It also depends on simulation configuration from `config` and on `viabilitykernels.simulation` for `runscenario` and `sampleinitialconditions`. These are used to generate the trajectory ensembles for every scenario-regime combination in the sweep.

## Sweep logic

The script begins by selecting all scenarios whose `expected` field matches the requested scenario class, such as `unstable`, `boundary`, or `stable`. It then iterates over three predefined initialization regimes: `inside`, `inside_near_boundary`, and `outside_near_boundary`.

For each scenario-regime pair, the script creates a regime-specific initial-condition center and noise scale, samples the initial conditions, checks whether any start outside the viability box, and runs the simulation once. The resulting ensemble is then reused for both the taxonomy plot and the trajectory animation.

## Classification logic

The taxonomy plot does not import a classifier implementation directly. Instead, the script calls `classifiers.classifier_dispatch` to obtain the classifier function, the optional memory-reset function, and the shared state-color mapping associated with the selected classifier type.

This means the same sweep script can produce static, temporal, or state-machine taxonomy views without changing the plotting code. Stateful classifiers can also reset their internal memory cleanly between trajectories through the reset function returned by the dispatcher.

## Plot outputs

The taxonomy figure is generated through `plotting.plot_helpers.save_taxonomy_plot`. This helper samples points from each simulated trajectory, computes local derivatives, applies the classifier returned by the dispatcher, and writes the classifier-colored 3D scatter plot as a PNG.

The animation is generated through `plotting.plot_helpers.save_trajectory_animation`. This helper builds 3D E-T-O segments from the simulated trajectories, colors them according to viability-box membership, animates the moving trajectory heads, and saves the result as a GIF.

## Practical design rationale

The workflow is organized into reusable layers: sweep-specific orchestration in `run_scenarios_3d.py`, scenario and viability utilities in `scenario_helpers`, classifier selection in `classifier_dispatch`, and visualization in `plot_helpers`. This keeps the sweep script independent as a runnable batch entry point while allowing it to share the same implementation backbone as the single-scenario plotting scripts.
