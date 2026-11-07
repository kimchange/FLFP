# Data

## Synthetic beads

| File in`demo/`  | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| `lf.tiff`       | Raw noisy light-field observations,`[view, y, x] = [81, 135, 135]` |
| `lf_clean.tiff` | Clean light-field reference in original optical intensity units      |
| `wf_true.txt`   | 21 reference wavefront coefficients in waves                         |
| `noise.json`    | Simulation metadata and intensity scale                              |

The reconstruction grid is `[z, y, x] = [101, 675, 675]`, with voxel spacing
`[5, 0.47898089, 0.47898089]` micrometers.

The observations follow `raw = Poisson(c * clean_LF) + Normal(105, 9)`, where
`c = 0.5829866912756507` and `9` is the Gaussian variance (standard deviation `3`).
Keep the sensor bias and negative values after bias subtraction when loading data.
The reference wavefront and clean light field are used only for evaluation after
optimization.

## Measured samples

Set `SAMPLE` in `autograd/demo_measured.py`. Each folder contains `lf.tiff`
(`[view, y, x]`) and `settings.json`. The observations include the surrounding
pixels needed for reconstruction; outputs contain the central region.

| Sample | Folder / `SAMPLE` | Figure | Objective | LF shape |
| --- | --- | --- | --- | --- |
| Beads | `beads` | 5a | Air | 81 x 357 x 357 |
| NeuN cells, left region | `neun_cells_left` | 5c | Oil | 81 x 375 x 375 |
| NeuN cells, right region | `neun_cells_right` | 5d | Oil | 81 x 375 x 375 |
| Worm slice, upper region | `worm_slice_upper` | 6d | Water | 81 x 339 x 339 |
| Worm slice, lower region | `worm_slice_lower` | 6e | Water | 81 x 339 x 339 |

