# Single Scenario Plots

This note describes the main dependencies and execution logic used to generate a 3D trajectory animation and a classifier-colored 3D taxonomy plot for one scenario.

## Example calls

The script is designed to accept a classifier type as a CLI argument, along with scenario selection and output settings. Typical calls would look like this:

```bash
python single_scenario_3d.py   --filter "Intermediate porosity"   --classifier-type temporal   --show-box

python single_scenario_3d.py   --filter "Intermediate porosity"   --classifier-type static   --n-traj 40   --stride 4   --out-dir figures/test_static

python single_scenario_3d.py   --filter "Unstable"   --classifier-type state_machine   --fps 12   --max-frames 200   --shift-T 1.2   --shift-E 1.1   --show-box
```

These examples all run the same single-scenario workflow, but they swap the classifier backend and optionally tune sampling density, animation settings, or the initial-condition center through the shift arguments.

## Main dependencies

The script depends on standard Python utilities for argument parsing and path setup, along with the shared project modules `plotting.plot_helpers` and `plotting.scenario_helpers`. These two helper modules now provide the reusable plotting, animation, scenario selection, initial-condition generation, and simulation-execution logic.

On the classification side, the script also depends on `classifiers.classifier_dispatch`, which acts as the selector layer for the available classifier backends. Instead of importing a single classifier implementation directly, the script requests the classifier function, optional reset function, and state-color mapping from the dispatcher.

## Execution flow

The script begins by parsing CLI arguments such as scenario filter, classifier type, output directory, number of trajectories, axis view angles, animation settings, and point subsampling stride. It then uses `scenario_helpers` to select the matching scenario, construct initial conditions, check viability-box membership of the starting points, and run the scenario simulation once.

The resulting simulation bundle is then reused for both outputs. The script sends the simulation result to the plotting helpers and asks the classifier dispatcher for the appropriate classifier backend before building the taxonomy plot.

## Classification logic

The classifier dispatcher centralizes backend selection for the static, temporal, and state-machine classifiers. It returns a common set of components: the pointwise classification function, a reset hook when the classifier maintains memory across calls, and the shared label-to-color mapping used by the taxonomy plot.

This keeps the plotting code independent from the internals of the classifier implementations. The taxonomy plotting helper simply receives a classifier function and applies it to sampled trajectory points, while memory reset behavior remains encapsulated in the dispatcher layer.

## Plot outputs

The animation is produced through `plotting.plot_helpers.save_trajectory_animation`, which uses the simulated trajectories directly rather than classifier output. It constructs the 3D E-T-O line segments, colors trajectory segments according to viability-box membership, draws the moving point markers, and saves the final GIF.

The taxonomy figure is produced through `plotting.plot_helpers.save_taxonomy_plot`, which samples points from the trajectories, computes local derivatives, applies the classifier returned by the dispatcher, and renders the classifier-colored 3D scatter plot. The taxonomy plot is saved as a PNG, while the animation is saved as a GIF in the selected output directory.

## Practical design rationale

The workflow is now split into three reusable layers: scenario preparation in `scenario_helpers`, classifier selection in `classifier_dispatch`, and visualization in `plot_helpers`. This keeps `single_scenario_plots.py`, `animate_scenario_3d_eto_box.py`, and `plot_scenario_3d_taxonomy_3d.py` independent as runnable scripts while allowing them to share the same underlying implementation logic.
