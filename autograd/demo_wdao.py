"""Main Fig. 3: one calibrated staged wDAO comparison.

Run: python autograd/demo_wdao.py
The default input is the same P=100, sample 2 measurement as demo_wavefront.py.
Shared optical/noise/object settings live near the top of demo_wavefront.py;
change the input paths below to apply this procedure to another measurement.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import tifffile
import torch

import demo_wavefront as common
from wdao.estimate import estimate_wavefront
from wdao.reconstruct import reconstruct_fixed
from wdao.phase_integration import physical_view_coordinates


# Raw sensor LF in [view, y, x]; keep bias and signed observations intact.
DATA = common.DATA
TRUE_WF = common.TRUE_WF                 # Evaluation only; set to None if unknown.
CLEAN_LF = common.CLEAN_LF               # Evaluation only; set to None if unknown.
INTENSITY_SCALE = common.INTENSITY_SCALE # Set to 1 for outputs in photon units.
OUTPUT = common.ROOT / 'outputs/demo_wdao'
DEVICE = common.DEVICE
OUTER_ITERATIONS = 10
OBJECT_STEPS = 2300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--intensity-scale', type=float,
                        help='Known intensity-to-photon coefficient; custom --data defaults to 1')
    parser.add_argument('--no-evaluation', action='store_true',
                        help='Do not load the optional true wavefront or clean LF')
    args = parser.parse_args()
    custom_input = args.data.resolve() != Path(DATA).resolve()
    intensity_scale = args.intensity_scale if args.intensity_scale is not None else (
        1. if custom_input else INTENSITY_SCALE)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use a CUDA environment for the full example')
        torch.cuda.set_device(device)
    if common.MODES != 21 or common.FIXED_MODES != [0, 1, 2, 4]:
        raise ValueError('The calibrated wDAO fit requires 21 OSA slots and gauge 0,1,2,4')
    if not (0 < common.WARMUP < common.STEPS and np.isfinite(intensity_scale) and intensity_scale > 0
            and common.READ_VARIANCE > 0):
        raise ValueError('Invalid optimization schedule, intensity scale, or read-noise variance')
    if not args.no_evaluation and custom_input:
        # A command-line input change must not silently compare against paper truth.
        args.no_evaluation = True
        print('Custom input: optional paper ground-truth evaluation is disabled.', flush=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    raw = torch.as_tensor(tifffile.imread(args.data), dtype=torch.float32, device=device)
    expected = (len(common.PHYSICAL_VIEWS),
        *(len(range(0, n, common.SAMPLE_STRIDE)) for n in common.VOLUME_SHAPE[-2:]))
    if tuple(raw.shape) != expected or not torch.isfinite(raw).all():
        raise ValueError(f'Expected finite LF data with shape {expected}; got {tuple(raw.shape)}')
    args.output.mkdir(parents=True, exist_ok=True)
    optical_settings = {key: common.OPTICS[key] for key in
        ('M', 'MLPitch', 'Nnum', 'fml', 'na_detection', 'lam_detection')}
    if common.OPTICS['Nnum'] % common.SAMPLE_STRIDE:
        raise ValueError('SAMPLE_STRIDE must divide the microlens angular-grid size')
    optical_settings['lf_samples_per_microlens'] = common.OPTICS['Nnum']//common.SAMPLE_STRIDE
    # The paper calibration deliberately supports the complete radius-5 disk
    # with 3 spatial samples per microlens; check this before expensive setup.
    physical_view_coordinates(common.PHYSICAL_VIEWS, optical_settings)
    common.synchronize(device)
    started = time.perf_counter()
    generator = common.PsfGenerator5D(**common.OPTICS, device=device,
                                      input_views=common.PHYSICAL_VIEWS)
    projection = common.Projection(generator)
    common.synchronize(device)
    setup_s = time.perf_counter()-started

    # Stage 1: ten ISRA/NCC/integration cycles, using raw observations only.
    started = time.perf_counter()
    wf, staged_history = estimate_wavefront(generator, projection, raw,
        common.PHYSICAL_VIEWS, optical_settings, bias=common.BIAS,
        outer_iterations=OUTER_ITERATIONS, output=args.output/'staged')
    common.synchronize(device)
    staged_s = time.perf_counter()-started
    np.savetxt(args.output/'staged_wavefront_waves.txt', wf.cpu().numpy(), fmt='%.12g')

    # Stage 2: a fresh uniform object; hold the staged estimate fixed for all updates.
    # The solver uses the original signed raw-BIAS data for its noise objective.
    started = time.perf_counter()
    volume, wf, fit = reconstruct_fixed(projection, raw[None], wf, common, steps=OBJECT_STEPS)
    common.synchronize(device)
    object_s = time.perf_counter()-started
    if not torch.isfinite(volume).all() or not torch.isfinite(wf).all():
        raise FloatingPointError('The final reconstruction contains nonfinite values')

    # Reuse the common demo's float32 output and optional post-fit evaluation.
    common.OUTPUT = args.output
    common.INTENSITY_SCALE = intensity_scale
    common.TRUE_WF = None if args.no_evaluation else TRUE_WF
    common.CLEAN_LF = None if args.no_evaluation else CLEAN_LF
    summary = common.save_results(generator, projection, volume, wf)
    (args.output/'object_history.json').write_text(json.dumps(fit.pop('history'))+'\n')
    summary.update(status='complete', method='Calibrated RUSH3D-derived wDAO adaptation',
        input=common.relative_path(args.data),
        outer_iterations=OUTER_ITERATIONS, object_steps=OBJECT_STEPS,
        formal_iteration_budget=(OUTER_ITERATIONS == 10 and OBJECT_STEPS == 2300),
        selected_outer=len(staged_history), fixed_wavefront=True,
        setup_s=setup_s, staged_estimation_s=staged_s, object_solve_s=object_s,
        device=device.type, intensity_scale=intensity_scale,
        volume_shape=common.VOLUME_SHAPE, psf_size=common.PSF_SIZE,
        sample_stride=common.SAMPLE_STRIDE, physical_views=common.PHYSICAL_VIEWS,
        optics=common.OPTICS, staged_calibration=optical_settings,
        bias=common.BIAS, read_variance=common.READ_VARIANCE,
        stages_access_ground_truth=False,
        lf_metric_domain='All clean channels, including fitted channels',
        **fit)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'Results: {common.relative_path(args.output)}')


if __name__ == '__main__':
    main()
