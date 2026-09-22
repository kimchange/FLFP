"""Staged wDAO estimation, without archived catalogs or ground-truth access.

The numerical loop is ported from the manuscript's run_wdao.py. Registration,
phase integration, ordered views, and outer-iteration object resets are retained.
"""
from pathlib import Path
import json
import time

import numpy as np
import torch

from .optical_operator import CachedOptics, isra_epoch
from .rush_registration import estimate_shifts, remove_center_and_defocus
from .phase_integration import fit_shifts_to_wavefront


@torch.no_grad()
def estimate_wavefront(generator, projection, raw, physical_views, optical_settings,
                       *, bias=105., outer_iterations=10, output=None):
    """Estimate 21 OSA coefficients from raw [view, y, x] observations only.

    Optical lengths in optical_settings are in metres. Gradients are calibrated
    on the measured physical pupil support; gauge slots 0,1,2,4 remain zero.
    The final iteration is always returned, without selecting trajectory minima.
    An optional output directory receives per-iteration diagnostics.
    """
    if outer_iterations < 1:
        raise ValueError('outer_iterations must be positive')
    if raw.ndim != 3 or len(raw) != len(physical_views):
        raise ValueError('raw must have shape [len(physical_views), y, x]')
    if not torch.isfinite(raw).all():
        raise ValueError('raw contains nonfinite values')
    if len(projection.basis) != 21:
        raise ValueError('This calibrated wDAO adaptation requires 21 OSA slots')
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    device = raw.device

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    data = raw.sub(bias).clamp_min(0)
    weight = data.sum((-2, -1))
    if weight.max() <= 0:
        raise ValueError('No positive observed signal remains after bias subtraction')
    nnum = optical_settings['Nnum']
    coords = torch.tensor([(v//nnum-nnum//2, v % nnum-nnum//2)
                           for v in physical_views], device=device)
    op = CachedOptics(generator, projection)
    wf = torch.zeros(21, device=device)
    history = []
    synchronize()
    started = time.perf_counter()
    for outer in range(outer_iterations):
        iteration_started = time.perf_counter()
        op.set_wavefront(wf)
        volume = torch.ones((op.depths, op.height, op.width), device=device)
        volume *= data.sum()/op.forward_all(volume).sum().clamp_min(1e-20)
        shifts = torch.zeros((len(physical_views), 2), device=device)
        for inner in range(5 if outer == 0 else 3):
            ncc_summary = None
            if inner:
                predicted = op.forward_all(volume)
                raw_shifts, ncc = estimate_shifts(predicted, data, max_shift=5.,
                                                  upsample=5, batch_size=8)
                shifts, removed = remove_center_and_defocus(
                    raw_shifts, coords, valid=ncc['valid'], clip=5.)
                ncc_summary = dict(mean_ncc=float(ncc['peak_ncc'].mean()),
                    min_ncc=float(ncc['peak_ncc'].min()),
                    invalid_views=int((~ncc['valid']).sum()),
                    search_boundary_views=int(ncc['at_search_boundary'].sum()),
                    shift_rms_native_pixel=float(shifts.square().mean().sqrt()),
                    shift_max_native_pixel=float(shifts.abs().max()),
                    center_shift_yx=removed['center_shift_yx'].cpu().tolist(),
                    removed_defocus_slope=float(removed['removed_defocus_slope']))
            isra_epoch(op, volume, data, shifts, weight=weight)
            if not torch.isfinite(volume).all():
                raise FloatingPointError('ISRA volume became nonfinite')
            synchronize()
            print(f'wDAO outer {outer+1}/{outer_iterations}, pass {inner+1}: '
                  f'{time.perf_counter()-started:.1f} s', flush=True)
        if not bool(ncc['valid'].all()):
            raise RuntimeError('Final NCC has invalid channels; missing shifts cannot be integrated as zeros')
        increment, diag = fit_shifts_to_wavefront(shifts.cpu().numpy(),
            physical_views, optical_settings, return_diagnostics=True)
        wf += torch.tensor(increment, device=device, dtype=wf.dtype)
        wf[[0, 1, 2, 4]] = 0
        if not torch.isfinite(wf).all():
            raise FloatingPointError('Estimated wavefront became nonfinite')
        predicted = op.forward_all(volume)
        observed_error = float((predicted-data).norm()/data.norm())
        if output is not None:
            np.savez_compressed(output/f'outer_{outer+1:02d}.npz',
                raw_shifts_yx=raw_shifts.cpu().numpy(),
                corrected_shifts_yx=shifts.cpu().numpy(),
                peak_ncc=ncc['peak_ncc'].cpu().numpy(),
                valid_views=ncc['valid'].cpu().numpy(),
                integrated_wavefront_waves=diag['integrated_wavefront_waves'],
                observed_support_mask=diag['observed_support_mask'],
                grid_coordinates_yx=diag['grid_coordinates_yx'],
                increment_waves=increment, cumulative_wavefront_waves=wf.cpu().numpy())
        history.append(dict(outer=outer+1,
            increment_coefficient_l2_waves=float(np.linalg.norm(increment)),
            cumulative_coefficient_l2_waves=float(wf.norm()), ncc=ncc_summary,
            unaligned_observed_lf_relative_l2_before_phase_update=observed_error,
            integration_fit_rms_waves=diag['fit_rms_on_observed_support_waves'],
            elapsed_s=time.perf_counter()-started,
            iteration_s=time.perf_counter()-iteration_started))
        if output is not None:
            (output/'history.json').write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
        del volume, predicted
    synchronize()
    return wf, history
