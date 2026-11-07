# Demo data


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
