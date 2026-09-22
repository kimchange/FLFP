"""Calibrated RUSH3D wDAO adaptation: shifts -> path mean -> OSA wavefront.

This module does not load experiment observations or wavefront ground truth.
Input shifts are measured LF translations in (row, column) pixel units, with
positive shifts meaning positive observed displacements relative to the
nominal forward image. They are NOT alignment-correction translations.

Historical provenance: RUSH3D v0.1 reconstruction/Loop_Phase_Reconstruction
calls intercircle_zy_v2 and SH.
It averages 1000 uniformly random, shortest, center-origin Manhattan paths,
using the gradient at each departing node. Dynamic programming here computes
that same estimator's exact expectation, not its finite-sample realization.

Declared adaptations: physical normalized-pupil coordinates and SI units
replace the historical ratio=4 and rescaled angular disc; linear interpolation
on the measured convex hull replaces zero padding/bicubic/nearest resizing;
21 OSA modes replace the historical 45 unnormalized SH modes. All 21 modes
are fit before removing gauge indices 0,1,2,4, matching Figure 5's 17 free
coefficients. OSA 0 in this nuisance fit is true constant piston. The PSF
code's slot 0 is instead an unused axial-propagation mode; returned slot 0
is always zero. No gradient extrapolation is performed. The fitted polynomial
can be evaluated beyond the measured hull, up to the unit pupil; that is
polynomial extrapolation and is not additional observed wavefront support.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping

import numpy as np
from scipy.interpolate import griddata

EXCLUDED_OSA = (0, 1, 2, 4)
PROVENANCE = {
    'integration_helper': 'intercircle_zy_v2.m',
    'historical_caller': 'RUSH3D v0.1 reconstruction/Loop_Phase_Reconstruction',
    'implementation': 'physical-coordinates-exact-path-expectation-osa21-v1',
}


def _array(value):
    if hasattr(value, 'detach'):
        if value.is_cuda:
            raise ValueError('This CPU calibration module requires a CPU tensor or NumPy array')
        value = value.detach().numpy()
    return np.asarray(value, dtype=np.float64)


def integrate_path_mean(gradient_yx, spacing):
    """Integrate an odd square grid, returning center-zero path means.

    gradient_yx has shape (N,N,2), with derivatives with respect to physical
    row/column coordinates. A CPU Torch tensor is also accepted. spacing is
    a positive scalar or (row_spacing,column_spacing). Both-NaN nodes mark
    unobserved support; outside the inscribed circle is also unobserved.
    No zeros or extrapolated gradients are supplied for missing nodes.
    Returns (N,N), NaN outside observed support, and exactly zero at center.

    For quadrant directions s_y,s_x, at offsets i,j the probability that a
    uniformly random shortest path arrived through its row predecessor is
    i/(i+j), and through its column predecessor j/(i+j). Each arrival term
    adds that edge's departure-node gradient times signed physical spacing.
    The valid support must contain both predecessors whenever they exist.
    """
    g = _array(gradient_yx)
    if g.ndim != 3 or g.shape[-1] != 2 or g.shape[0] != g.shape[1]:
        raise ValueError('gradient_yx must have shape (N,N,2)')
    n = g.shape[0]
    if n < 3 or n % 2 != 1:
        raise ValueError('The integration grid must be odd and at least 3x3')
    step = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    if not np.all(np.isfinite(step)) or np.any(step <= 0):
        raise ValueError('spacing must be positive and finite')
    finite = np.isfinite(g)
    if np.any(finite[..., 0] != finite[..., 1]) or np.any(np.isinf(g)):
        raise ValueError('Missing support must have NaN in both gradient channels')
    radius = n // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    valid = finite.all(axis=-1) & (yy * yy + xx * xx <= radius * radius)
    if not valid[radius, radius]:
        raise ValueError('The pupil center must be observed')
    phase = np.full((n, n), np.nan)
    phase[radius, radius] = 0.0
    for sy in (-1, 1):
        for sx in (-1, 1):
            for i in range(radius + 1):
                for j in range(radius + 1):
                    row, col = radius + sy * i, radius + sx * j
                    if i + j == 0 or not valid[row, col]:
                        continue
                    value = 0.0
                    if i:
                        prev = (row - sy, col)
                        if not valid[prev] or not np.isfinite(phase[prev]):
                            raise ValueError('Observed support is not coordinate-monotone from center')
                        value += i / (i + j) * (phase[prev] + sy * step[0] * g[prev][0])
                    if j:
                        prev = (row, col - sx)
                        if not valid[prev] or not np.isfinite(phase[prev]):
                            raise ValueError('Observed support is not coordinate-monotone from center')
                        value += j / (i + j) * (phase[prev] + sx * step[1] * g[prev][1])
                    phase[row, col] = value
    return phase


def osa_to_nm(index):
    if index < 0 or int(index) != index:
        raise ValueError('OSA index must be a nonnegative integer')
    n = int((math.sqrt(8 * int(index) + 1) - 1) / 2)
    return n, 2 * int(index) - n * (n + 2)


@lru_cache(maxsize=None)
def _osa_terms(index):
    """Cartesian polynomial coefficients ((y_power,x_power),coefficient)."""
    n, m = osa_to_nm(index)
    q = abs(m)
    norm = math.sqrt((n + 1) * (2 if m else 1))
    terms = {}
    for k in range((n - q) // 2 + 1):
        radial = ((-1) ** k * math.factorial(n - k) /
                  (math.factorial(k) * math.factorial((n + q) // 2 - k) *
                   math.factorial((n - q) // 2 - k)))
        p = (n - 2 * k - q) // 2
        for y_harmonic in range(q + 1):
            # Real/imaginary parts of (x+i*y)^|m|, avoiding polar singularities.
            if m >= 0:
                trig = (1, 0, -1, 0)[y_harmonic % 4]
            else:
                trig = (0, 1, 0, -1)[y_harmonic % 4]
            if trig == 0:
                continue
            angular = math.comb(q, y_harmonic) * trig
            for y_radial in range(p + 1):
                key = (y_harmonic + 2 * y_radial,
                       q - y_harmonic + 2 * (p - y_radial))
                terms[key] = terms.get(key, 0.0) + norm * radial * angular * math.comb(p, y_radial)
    return tuple(terms.items())


def osa_basis(coordinates_yx, count=21):
    """Unit-RMS OSA/ANSI basis at actual normalized-pupil coordinates."""
    coordinates = _array(coordinates_yx)
    if coordinates.shape[-1] != 2:
        raise ValueError('Coordinates must end in (y,x)')
    y, x = coordinates[..., 0], coordinates[..., 1]
    basis = np.zeros(y.shape + (count,))
    for j in range(count):
        for (py, px), coefficient in _osa_terms(j):
            basis[..., j] += coefficient * y ** py * x ** px
    return basis


def osa_gradient_basis(coordinates_yx, count=21):
    """Analytic (d/d rho_y,d/d rho_x); shape (...,count,2)."""
    coordinates = _array(coordinates_yx)
    y, x = coordinates[..., 0], coordinates[..., 1]
    derivatives = np.zeros(y.shape + (count, 2))
    for j in range(count):
        for (py, px), coefficient in _osa_terms(j):
            if py:
                derivatives[..., j, 0] += coefficient * py * y ** (py - 1) * x ** px
            if px:
                derivatives[..., j, 1] += coefficient * px * y ** py * x ** (px - 1)
    return derivatives


@dataclass(frozen=True)
class OpticalSettings:
    M: float
    MLPitch: float
    Nnum: int
    fml: float
    na_detection: float
    lam_detection: float
    lf_samples_per_microlens: int = 3

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            result = value
        else:
            get = value.get if isinstance(value, Mapping) else lambda name, default=None: getattr(value, name, default)
            result = cls(M=float(get('M')), MLPitch=float(get('MLPitch')), Nnum=int(get('Nnum')),
                         fml=float(get('fml')), na_detection=float(get('na_detection', get('NA'))),
                         lam_detection=float(get('lam_detection', get('wavelength_m'))),
                         lf_samples_per_microlens=int(get('lf_samples_per_microlens', 3)))
        values = (result.M, result.MLPitch, result.fml, result.na_detection, result.lam_detection)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError('Optical lengths, magnification and NA must be positive SI values')
        if result.Nnum < 11 or result.Nnum % 2 != 1 or result.lf_samples_per_microlens != 3:
            raise ValueError('This Figure 5 adaptation requires an odd angular grid and 3 LF samples per microlens')
        return result

    @property
    def angular_step(self):
        return self.M * (self.MLPitch / self.Nnum) / (self.fml * self.na_detection)

    @property
    def lf_pixel_pitch_object_m(self):
        return self.MLPitch / (3 * self.M)

    @property
    def shift_to_gradient_waves(self):
        # dW/d rho = (NA/lambda) * delta_x_object; no radians or historical ratio=4.
        return self.na_detection / self.lam_detection * self.lf_pixel_pitch_object_m


def physical_view_coordinates(physical_views, opticalsettings):
    """81 zero-based flattened angular IDs -> physical normalized pupil (y,x)."""
    settings = OpticalSettings.from_value(opticalsettings)
    view_ids = np.asarray(physical_views)
    if view_ids.shape != (81,) or not np.issubdtype(view_ids.dtype, np.integer):
        raise ValueError('physical_views must be 81 integer, zero-based flattened angular IDs')
    if np.any(view_ids < 0) or np.any(view_ids >= settings.Nnum ** 2):
        raise ValueError('A physical view ID is outside the angular grid')
    offsets = np.column_stack(np.unravel_index(view_ids, (settings.Nnum, settings.Nnum))) - settings.Nnum // 2
    expected = {(y, x) for y in range(-5, 6) for x in range(-5, 6) if y * y + x * x <= 25}
    if {tuple(offset) for offset in offsets} != expected:
        raise ValueError('Expected all 81 distinct views in the radius-5 angular disc')
    return offsets * settings.angular_step


def fit_shifts_to_wavefront(shifts_yx_LF81, physical_views, opticalsettings, *,
                            grid_size=135, return_diagnostics=False):
    """Return 21 OSA coefficients in waves, with indices 0,1,2,4 set to zero.

    Input order must match physical_views. All modes (including nuisance
    piston/tip/tilt/defocus) participate in the least-squares fit BEFORE gauge
    removal. A constant gradient offset, e.g. from central-view referencing,
    is therefore absorbed as tilt without forcing coma coefficients to fit it.

    Set return_diagnostics=True to receive (coefficients, diagnostics). The
    diagnostics include observed_support_mask and grids for archival; this
    module deliberately performs no file I/O and writes no case estimates.
    """
    settings = OpticalSettings.from_value(opticalsettings)
    shifts = _array(shifts_yx_LF81)
    if shifts.shape != (81, 2) or not np.isfinite(shifts).all():
        raise ValueError('shifts_yx_LF81 must contain 81 finite (row,column) pairs')
    if grid_size < 7 or grid_size % 2 != 1:
        raise ValueError('grid_size must be odd and >=7')
    coordinates = physical_view_coordinates(physical_views, settings)
    support_radius = 5 * settings.angular_step
    if support_radius > 1 + 1e-12:
        raise ValueError('The measured angular disc extends outside the unit pupil')
    axis = np.linspace(-support_radius, support_radius, grid_size)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    grid_coordinates = np.stack((yy, xx), axis=-1)
    measured_gradients = shifts * settings.shift_to_gradient_waves
    gradient_grid = griddata(coordinates, measured_gradients, grid_coordinates, method='linear', fill_value=np.nan)
    circle = yy * yy + xx * xx <= support_radius ** 2 * (1 + 1e-12)
    gradient_grid[~circle] = np.nan
    observed = np.isfinite(gradient_grid).all(axis=-1)
    spacing = 2 * support_radius / (grid_size - 1)
    phase = integrate_path_mean(gradient_grid, spacing)
    observed &= np.isfinite(phase)
    basis = osa_basis(grid_coordinates[observed])
    coefficients_all, _, rank, singular = np.linalg.lstsq(basis, phase[observed], rcond=None)
    if rank != 21:
        raise ValueError(f'The observed support is insufficient for the 21-mode fit: rank={rank}')
    coefficients = coefficients_all.copy()
    coefficients[list(EXCLUDED_OSA)] = 0.0
    if not return_diagnostics:
        return coefficients
    residual = basis @ coefficients_all - phase[observed]
    diagnostics = dict(
        provenance=PROVENANCE.copy(), coefficient_units='waves; unit-RMS OSA/ANSI j=0..20',
        excluded_osa=list(EXCLUDED_OSA), fit_all_modes_before_gauge_removal=True,
        nuisance_piston_is_true_osa0_not_psf_axial_slot0=True,
        input_shift_convention='observed positive LF displacement (row,column), not alignment correction',
        angular_step_rho=settings.angular_step, support_radius_rho=support_radius,
        lf_pixel_pitch_object_m=settings.lf_pixel_pitch_object_m,
        shift_to_gradient_waves=settings.shift_to_gradient_waves,
        integration_spacing_rho=spacing, interpolation='linear griddata; valid convex hull only',
        grid_size=grid_size, observed_pixel_count=int(observed.sum()), circle_pixel_count=int(circle.sum()),
        observed_support_mask=observed, grid_coordinates_yx=grid_coordinates,
        interpolated_gradients_yx_waves=gradient_grid, integrated_wavefront_waves=phase,
        coefficients_all_before_gauge_removal=coefficients_all, fit_rank=int(rank),
        design_condition_number=float(singular[0] / singular[-1]),
        fit_rms_on_observed_support_waves=float(np.sqrt(np.mean(residual ** 2))),
        extrapolation='Only the fitted OSA polynomial is extended beyond measured convex hull; gradients are not extrapolated',
    )
    return coefficients, diagnostics
