# FLFP

A fast, fully differentiable light-field PSF model for joint 3D reconstruction and
optical calibration.

This release contains one synthetic bead demo for joint estimation of scene and wavefront. The full project will be released
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
to `outputs/demo_wavefront/`. Running it again overwrites these outputs.

For the bead demo, a full run reproduces wavefront RMS error of 
approximately **0.013328 waves** and clean light-field
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
