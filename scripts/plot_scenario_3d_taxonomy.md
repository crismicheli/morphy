# plot_scenario_3d_taxonomy Script

## Example calls

```bash
python plot_scenario_3d_taxonomy.py --filter "Intermediate porosity"
python plot_scenario_3d_taxonomy.py --filter "Intermediate porosity" --classifier-type temporal --show-box
python plot_scenario_3d_taxonomy.py --filter "Unstable" --classifier-type state_machine --stride 4 --output figures/unstable_taxonomy_3d.png
```

This script generates a static 3D scatter plot in the E-T-O phenotype space for one selected scenario. The plotted points are sampled from simulated trajectories and colored by the taxonomy label estimated by the chosen classifier backend.

The script first selects one scenario through a label substring filter and runs a single-scenario ensemble simulation using the shared scenario helper. It then asks `classifier_dispatch` for the requested classifier function, reset hook, and shared label-to-color mapping.

The final figure is produced by `plotting.plot_helpers.save_taxonomy_plot`. That helper computes local derivatives along the trajectories, applies the selected classifier pointwise, and renders the resulting 3D taxonomy view.

This script shows estimated taxonomy states, not only raw dynamics. It is therefore the 3D taxonomy companion to the dynamic-only animation scripts.
