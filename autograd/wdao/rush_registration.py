"""NCC registration adapted from the public RUSH3D reconstruction routine.

Reference: yuanlong-o/RUSH3D, commit
4e78e6cbaba3245ad82516526fed329e2a6f3d49,
va.31/reconstruction_module/recon_module.m, lines 93--138.
The historical Loop_Phase_Reconstruction (commit 13923b3...) first resizes
LF measurements by Nnum/Nshift. Here that factor is 5 by default.

This is normalized cross-correlation, not phase correlation. Peaks are integer
on the upsampled grid. Shift coordinates are (row, column), measured minus
predicted, expressed in ORIGINAL LF pixels. The +/-5-pixel default search
margin is 25 pixels after 5x enlargement, retargeted to the original example's
clip=25. It is not the original search margin=9, which was smaller than its
clipping bound. That example also uses 3x3 spatial tiles. This common-pupil
experiment uses one central template per angular view, retaining the original
~88% template fraction when the requested search margin fits. No blur filter
is applied; reg_gSig=2
in main_global.m belongs to the separate temporal motion-correction stage.

All operations retain the input device. The Keys bicubic upsampler uses
a=-0.5, half-pixel coordinates and symmetric extension as MATLAB imresize;
torch's built-in bicubic mode instead uses a=-0.75 and is not used here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _cubic(x: torch.Tensor) -> torch.Tensor:
    """Keys cubic-convolution kernel with a=-1/2."""
    a = x.abs()
    return torch.where(
        a <= 1,
        1.5 * a**3 - 2.5 * a**2 + 1,
        torch.where(a < 2, -.5 * a**3 + 2.5 * a**2 - 4 * a + 2, 0),
    )


def _resize_weights(n: int, scale: int, like: torch.Tensor):
    positions = (torch.arange(n * scale, device=like.device, dtype=like.dtype)
                 + .5) / scale - .5
    indices = positions.floor().long()[:, None] + torch.arange(
        -1, 3, device=like.device
    )[None, :]
    weights = _cubic(positions[:, None] - indices)
    weights = weights / weights.sum(dim=1, keepdim=True)
    # Symmetric extension repeats the edge sample: ... b a | a b ... .
    indices = indices.remainder(2 * n)
    indices = torch.where(indices < n, indices, 2 * n - 1 - indices)
    return indices, weights


def matlab_bicubic_upsample(images: torch.Tensor, scale: int = 5) -> torch.Tensor:
    """Separable MATLAB-style bicubic enlargement of (..., H, W) arrays.

    Enlargement needs no antialiasing prefilter. This function never clips
    overshoots or renormalizes intensities; both would alter the registration.
    """
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
        raise ValueError("scale must be a positive integer")
    if images.ndim < 2 or min(images.shape[-2:]) < 1:
        raise ValueError("images must have nonempty spatial axes")
    if not images.is_floating_point() or images.dtype in (torch.float16, torch.bfloat16):
        images = images.float()
    if scale == 1:
        return images
    iy, wy = _resize_weights(images.shape[-2], scale, images)
    ix, wx = _resize_weights(images.shape[-1], scale, images)
    enlarged_y = (images[..., iy, :] * wy[..., None]).sum(dim=-2)
    return (enlarged_y[..., ix] * wx).sum(dim=-1)


def _window_sum(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    integral = F.pad(images.cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    return (integral[..., height:, width:] - integral[..., :-height, width:]
            - integral[..., height:, :-width] + integral[..., :-height, :-width])


def estimate_shifts(
    predicted_LF81: torch.Tensor,
    measured_LF81: torch.Tensor,
    upsample: int = 5,
    max_shift: float = 5.0,
    batch_size: int = 8,
    variance_rtol: float = 1e-7,
    template_fraction: float = .88,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate independent per-view shifts and NCC diagnostics.

    Inputs have shape (views, H, W); the view count need not be exactly 81.
    Search displacements are bounded by floor(max_shift * upsample) pixels
    on the enlarged grid. A central prediction crop is correlated with every
    fully overlapping window of the measured image, matching MATLAB's valid
    template-window NCC. The measured search crop surrounds the template
    position by the specified margin; it need not occupy the whole image.
    Positive dy means features moved down in measured.

    Invalid (constant/non-finite) views return shift=(0,0), peak_ncc=0 and
    valid=False. There is no quality-based rejection or subpixel peak fitting.
    ``remove_center_and_defocus`` applies the separate angular gauge step.
    """
    predicted = torch.as_tensor(predicted_LF81)
    measured = torch.as_tensor(measured_LF81, device=predicted.device)
    if predicted.ndim != 3 or predicted.shape != measured.shape:
        raise ValueError("predicted and measured must have equal (views,H,W) shapes")
    if not isinstance(upsample, int) or isinstance(upsample, bool) or upsample < 1:
        raise ValueError("upsample must be a positive integer")
    if max_shift < 0 or not math.isfinite(max_shift):
        raise ValueError("max_shift must be finite and nonnegative")
    if batch_size < 1 or variance_rtol < 0:
        raise ValueError("batch_size must be positive and variance_rtol nonnegative")
    if not 0 < template_fraction <= 1:
        raise ValueError("template_fraction must lie in (0,1]")
    dtype = torch.float64 if predicted.dtype == torch.float64 else torch.float32
    predicted, measured = predicted.to(dtype), measured.to(dtype)
    views, h0, w0 = predicted.shape
    if views < 1:
        raise ValueError("at least one view is required")
    margin = math.floor(max_shift * upsample + 1e-8)
    h, w = h0 * upsample, w0 * upsample
    # MATLAB round for positive values is floor(x+.5), unlike Python's ties
    # to even. Match N3=round(.88*H/2)*2+1, but permit smaller test arrays.
    th, tw = [min(math.floor(template_fraction*n/2+.5)*2+1, n-2*margin)
              for n in (h, w)]
    if min(th, tw) < 3:
        raise ValueError("search margin leaves less than a 3x3 template")
    top, left = (h-th)//2, (w-tw)//2
    search_h, search_w = th+2*margin, tw+2*margin
    fft_shape = (search_h + th - 1, search_w + tw - 1)
    # Smooth powers of two avoid large-prime FFTs and improve GPU throughput.
    fft_shape = tuple(1 << (n - 1).bit_length() for n in fft_shape)
    all_shifts, all_peaks, all_valid, all_boundary = [], [], [], []
    all_template_std, all_window_std = [], []
    for start in range(0, views, batch_size):
        p0, m0 = predicted[start:start + batch_size], measured[start:start + batch_size]
        finite = torch.isfinite(p0).all(dim=(-2, -1)) & torch.isfinite(m0).all(dim=(-2, -1))
        p0 = torch.where(finite[:, None, None], p0, 0)
        m0 = torch.where(finite[:, None, None], m0, 0)
        measured_energy = (m0-m0.mean(dim=(-2,-1),keepdim=True)).square().sum(dim=(-2,-1))
        measured_nonconstant = measured_energy > variance_rtol*m0.square().sum(dim=(-2,-1))
        p = matlab_bicubic_upsample(p0, upsample)
        m = matlab_bicubic_upsample(m0, upsample)
        template_raw = p[:, top:top+th, left:left+tw]
        m = m[:, top-margin:top+th+margin, left-margin:left+tw+margin]
        template = template_raw - template_raw.mean(dim=(-2, -1), keepdim=True)
        template_energy = template.square().sum(dim=(-2, -1))
        raw_energy = template_raw.square().sum(dim=(-2, -1))
        valid_template = template_energy > variance_rtol * raw_energy
        # A global DC removal is NCC-invariant and stabilizes integral sums.
        m = m - m.mean(dim=(-2, -1), keepdim=True)
        numerator = torch.fft.irfft2(
            torch.fft.rfft2(m, s=fft_shape)
            * torch.fft.rfft2(template, s=fft_shape).conj(), s=fft_shape
        )[:, :2 * margin + 1, :2 * margin + 1]
        window_sum = _window_sum(m, th, tw)
        window_sum_sq = _window_sum(m.square(), th, tw)
        window_energy = (window_sum_sq - window_sum.square() / (th * tw)).clamp_min(0)
        valid_windows = window_energy > variance_rtol * window_sum_sq
        denominator = (template_energy[:, None, None] * window_energy).sqrt()
        valid = finite & valid_template & measured_nonconstant & valid_windows.any(dim=(-2, -1))
        ncc = torch.where(
            valid[:, None, None] & valid_windows,
            numerator / denominator.clamp_min(torch.finfo(dtype).tiny),
            -torch.inf,
        ).clamp(max=1)
        # MATLAB find selects the first maximum in COLUMN-major order.
        flat = ncc.transpose(-1, -2).contiguous().flatten(1)
        peaks, indices = flat.max(dim=1)
        peak_y = indices.remainder(2 * margin + 1)
        peak_x = indices.div(2 * margin + 1, rounding_mode='floor')
        shifts = torch.stack((peak_y - margin, peak_x - margin), dim=1).to(dtype) / upsample
        shifts = torch.where(valid[:, None], shifts, 0)
        peaks = torch.where(valid, peaks.clamp_min(-1), 0)
        boundary = valid & ((peak_y == 0) | (peak_y == 2 * margin)
                            | (peak_x == 0) | (peak_x == 2 * margin))
        selected_energy = window_energy[torch.arange(len(p), device=p.device), peak_y, peak_x]
        all_shifts.append(shifts)
        all_peaks.append(peaks)
        all_valid.append(valid)
        all_boundary.append(boundary)
        all_template_std.append((template_energy / (th * tw)).sqrt())
        all_window_std.append(torch.where(valid, (selected_energy / (th * tw)).sqrt(), 0))
    return torch.cat(all_shifts), {
        'peak_ncc': torch.cat(all_peaks), 'valid': torch.cat(all_valid),
        'at_search_boundary': torch.cat(all_boundary),
        'template_std': torch.cat(all_template_std),
        'matched_window_std': torch.cat(all_window_std),
        'upsample': upsample, 'search_margin_original_pixels': margin / upsample,
        'shift_quantization_original_pixels': 1 / upsample,
        'upsampled_shape': (h, w), 'template_shape': (th, tw),
        'template_fraction_requested': template_fraction,
        'upsampler': 'Keys bicubic a=-0.5; half-pixel; symmetric boundary',
        'sign': 'measured-minus-predicted; (row,column); original LF pixels',
    }


def remove_center_and_defocus(
    shifts_yx: torch.Tensor,
    views_yx: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    remove_defocus: bool = True,
    clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the center-reference and radial-defocus projection separately.

    ``views_yx`` are centered integer angular coordinates, e.g. the 81 points
    with y*y+x*x<=25. The official formula is
    k=sum(y*shift_y+x*shift_x)/sum(y*y+x*x), then shift -= k*(y,x).
    It is applied after subtraction of the center shift and optional clipping;
    the result is clipped again. Invalid channels stay zero and are excluded
    from the projection denominator (an explicit robustness adaptation).
    If the center is invalid, its reference is zero and this is reported.
    """
    shifts = torch.as_tensor(shifts_yx)
    if not shifts.is_floating_point():
        shifts = shifts.float()
    coords = torch.as_tensor(views_yx, device=shifts.device, dtype=shifts.dtype)
    if shifts.ndim != 2 or shifts.shape[-1] != 2 or coords.shape != shifts.shape:
        raise ValueError("shifts and angular coordinates must both have shape (views,2)")
    centers = (coords == 0).all(dim=1).nonzero().flatten()
    if centers.numel() != 1:
        raise ValueError("exactly one (0,0) angular channel is required")
    if clip is not None and (clip < 0 or not math.isfinite(clip)):
        raise ValueError("clip must be finite and nonnegative")
    mask = torch.isfinite(shifts).all(dim=1)
    if valid is not None:
        supplied = torch.as_tensor(valid, device=shifts.device, dtype=torch.bool)
        if supplied.shape != mask.shape:
            raise ValueError("valid must have shape (views,)")
        mask = mask & supplied
    shifts = torch.where(mask[:, None], shifts, 0)
    center = centers[0]
    center_shift = shifts[center].clone()
    corrected = torch.where(mask[:, None], shifts - center_shift, 0)
    if clip is not None:
        corrected = corrected.clamp(-clip, clip)
    denominator = coords.square().sum(dim=1)[mask].sum()
    k = (corrected * coords).sum() / denominator.clamp_min(torch.finfo(shifts.dtype).tiny)
    if not remove_defocus:
        k = k * 0
    corrected = torch.where(mask[:, None], corrected - k * coords, 0)
    if clip is not None:
        corrected = corrected.clamp(-clip, clip)
    return corrected, {'center_shift_yx': center_shift, 'center_valid': mask[center],
                       'removed_defocus_slope': k, 'valid': mask}
