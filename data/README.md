# Demo data

Only the files needed for the synthetic bead demo are included. The full datasets
will be released after the paper is accepted.

| File in `demo/` | Description |
| --- | --- |
| `lf.tiff` | Raw noisy light-field observations, `[view, y, x] = [81, 135, 135]` |
| `lf_clean.tiff` | Clean light-field reference in original optical intensity units |
| `wf_true.txt` | 21 reference wavefront coefficients in waves |
| `noise.json` | Simulation metadata and intensity scale |

The demo uses a peak expected signal of 100 photons and follows the configuration
in paper Fig. 3(a-e).
The reconstruction grid is `[z, y, x] = [101, 675, 675]`, with voxel spacing
`[5, 0.47898089, 0.47898089]` micrometers.

The observations follow `raw = Poisson(c * clean_LF) + Normal(105, 9)`, where
`c = 0.5829866912756507` and `9` is the Gaussian variance (standard deviation `3`).
Keep the sensor bias and negative values after bias subtraction when loading data.
The reference wavefront and clean light field are used only for evaluation after
optimization.

The script generates the paper's view schedule with CPU seed 0; no additional
schedule or initialization files are required.
