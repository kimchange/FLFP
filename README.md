# FLFP

A fast, fully differentiable light-field PSF model for joint 3D reconstruction and
optical calibration.

This release contains one synthetic bead demo. The full datasets will be released
after the paper is accepted.

## Quick start

Use Python 3.10 and a CUDA-enabled PyTorch installation. From this directory, run:

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python autograd/demo_wavefront.py
```

The demo jointly estimates a 3D bead volume and wavefront from 81 noisy light-field
views at a peak signal of 100 photons. It runs 2,400 optimization steps
and saves the reconstructed volume, wavefront, reprojected images, and error metrics
to `outputs/demo_wavefront_sample02/`. Running it again overwrites these outputs.

The demo follows the paper's joint estimation method for Fig. 3(a-e): nominal-PSF
refocusing for initialization, nonnegative sensitivity-scaled object updates, and
Adam wavefront updates. The fit uses the Poisson-Gaussian quasi-deviance, an XY
curvature prior, and Gaussian blur stages of 4, 2, 1, and 0 pixels. Step sizes stay
fixed throughout. A fixed seed reproduces the paper's view order.

For this sample (`points_p0100_s01`), a full run reproduces the paper's archived
wavefront RMS error of approximately **0.013328 waves** and clean light-field
relative L2 error of **7.79218%**. Validated with the pinned dependencies on an
RTX 3090: about 6.3 minutes for the full run. Small floating-point differences
may occur across environments.

## Files

- [autograd/demo_wavefront.py](autograd/demo_wavefront.py): configuration, forward
  model, loss, optimization, and output in one script.
- [autograd/lfpsf_torch_shift_blur_batched.py](autograd/lfpsf_torch_shift_blur_batched.py):
  differentiable PSF model.
- [data/demo/](data/demo/): input and reference files for this demo only.

To use your own data, edit the settings at the top of `demo_wavefront.py`.
Optical lengths are in meters; wavefront coefficients are in waves. Set `TRUE_WF`
and `CLEAN_LF` to `None` when reference data are unavailable.

## Data

See [data/README.md](data/README.md) for array shapes, units, and noise parameters.
Reference data are used only for evaluation after reconstruction.
