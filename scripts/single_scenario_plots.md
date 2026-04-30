# Single Scenario Plots

This note describes the main dependencies and execution logic used to generate a 3D trajectory animation and a classifier-colored 3D taxonomy plot for one scenario.

## Example calls

The script is designed to accept a classifier type as a CLI argument, along with scenario selection and output settings. Typical calls would look like this:

```bash
python single_scenario_plots.py   --filter "Intermediate porosity"   --classifier-type temporal   --show-box

python single_scenario_plots.py   --filter "Intermediate porosity"   --classifier-type static   --n-traj 40   --stride 4   --out-dir figures/test_static

python single_scenario_plots.py   --filter "Unstable"   --classifier-type state_machine   --fps 12   --max-frames 200   --shift-T 1.2   --shift-E 1.1   --show-box
```

These examples all run the same single-scenario workflow, but they swap the classifier backend and optionally tune sampling density, animation settings, or the initial-condition center through the shift arguments.

## Main dependencies

The script depends on standard Python utilities for argument parsing and path setup, NumPy for array handling, Matplotlib for static plotting, and `FuncAnimation` plus `PillowWriter` for GIF generation. It also uses `Line3DCollection` and `Poly3DCollection` from `mpl_toolkits.mplot3d.art3d` to render trajectory segments and the translucent viability box in 3D space.

On the domain side, the script imports simulation configuration from `config`, scenario definitions from `SCENARIOS`, and trajectory generation functions such as `run_scenario` and `sample_initial_conditions` from the simulation package. Classifier selection is delegated to a dispatcher layer, which routes calls to the static, temporal, or state-machine classifier while preserving shared state colors and reset behavior.

## Execution flow

The script begins by parsing CLI arguments such as scenario filter, classifier type, output directory, number of trajectories, axis view angles, animation settings, and point subsampling stride. It then selects a scenario by matching the user-provided filter string against scenario labels and constructs initial conditions using scenario-specific shifts and noise scales.

After the initial conditions are sampled, the script warns if any start outside the viability box and then runs the selected scenario simulation to obtain a bundle of solution trajectories. These solutions are then passed to the two plotting scripts mentioned earlier: `plot_scenario_3d_taxonomy_3d.py` for the classifier-colored taxonomy scatter plot, and `animate_scenario_3d_eto_box.py` for the E-T-O trajectory animation.

## Classification logic

For the classifier-colored plot, the workflow computes numerical derivatives from each solution, iterates through timepoints with a configurable stride, and calls the dispatcher with `classifier_type` to obtain a label for each sampled point. The dispatcher can route to the static taxonomy classifier, the temporal wrapper, or the state-machine classifier, and it is also the right place to reset classifier memory before each solution when stateful modes are used.

This separation keeps the plotting scripts focused on visualization rather than classifier internals. It also avoids duplicating import and reset logic across multiple plotting scripts, which is especially important because the temporal and state-machine classifiers maintain internal caches across calls.

## Plot outputs

The animation is handled by `animate_scenario_3d_eto_box.py`, which uses the simulated trajectories directly rather than classifier output. For each trajectory, it builds 3D line segments in E-T-O space and colors them according to whether the trajectory is inside or outside the viability region, while a moving point marks the current position at each frame.

The classifier plot is handled by `plot_scenario_3d_taxonomy_3d.py`, which samples points from the trajectories and colors them according to the selected classifier labels. A text overlay in the animation updates the current simulation time and an aggregated value of `C` across trajectories, using a selectable statistic such as mean, median, minimum, or maximum. The animation is saved as a GIF, while the taxonomy plot is saved as a PNG in the chosen output directory.

## Practical design rationale

The workflow combines two complementary views of the same simulation output: geometric trajectory evolution in the animation, and phenotype-like state labeling in the classifier plot. Keeping simulation, classification dispatch, and plotting as separate layers makes the code easier to extend, test, and reuse when additional classifiers or plotting modes are added later.
