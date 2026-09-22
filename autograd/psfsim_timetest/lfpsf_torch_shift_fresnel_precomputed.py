"""Shift--Fresnel with all angle kernels prepared by the constructor.

This is an explicit variant of the unchanged reference implementation beside
this file. It retains its Fourier grid, sampling, normalization and FFT batch
sizes. To fit the 765-pixel case plus 225 full-grid kernels on a 24-GiB GPU,
the unused kprop array is released and full-depth, zero-aberration requests
reuse kbase directly instead of copying the entire 101-plane spectrum.
"""

import importlib.util
from pathlib import Path

import torch

_path = Path(__file__).with_name("lfpsf_torch_shift_fresnel.py")
_spec = importlib.util.spec_from_file_location("_precomputed_fresnel_reference", _path)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


class PsfGenerator5D(_reference.PsfGenerator5D):
    def __init__(self, *args, precompute_view_batch_size=3, **kwargs):
        if precompute_view_batch_size < 1:
            raise ValueError("precompute_view_batch_size must be positive")
        kwargs["cache_angle_kernels"] = True
        super().__init__(*args, **kwargs)
        # kprop is only used by the base constructor to form kbase. No base
        # evaluation method reads it; the complete depth physics remains in kbase.
        del self.kprop
        self._full_depth_indices = torch.arange(self.Nz, device=self.device)
        self._prepared_views = {}
        for start in range(0, self.Nnum**2, precompute_view_batch_size):
            stop = min(start + precompute_view_batch_size, self.Nnum**2)
            flat = torch.arange(start, stop, device=self.device)
            u = flat // self.Nnum
            v = flat % self.Nnum
            block = super()._angle_kernels(u, v)
            for local, view in enumerate(range(start, stop)):
                self._prepared_views[view] = block[local]
        self.precomputed_view_count = len(self._prepared_views)
        self.precomputed_kernel_bytes = sum(t.numel() * t.element_size()
                                            for t in self._angle_kernel_cache.values())
        self.angle_batch_cache_hits = 0
        self.angle_subset_cache_hits = 0
        self.full_depth_spectrum_reuses = 0

    def _angle_kernels(self, u, v):
        key = (tuple(u.detach().cpu().tolist()), tuple(v.detach().cpu().tolist()))
        cached = self._angle_kernel_cache.get(key)
        if cached is not None:
            self.angle_batch_cache_hits += 1
            return cached
        # Requests for one view or a different subset also use prepared kernels;
        # they never perform another angle-kernel FFT. Standard benchmark
        # batches take the cached-block path above, which does not copy kernels.
        indices = [a * self.Nnum + b for a, b in zip(*key)]
        self.angle_subset_cache_hits += 1
        return torch.stack([self._prepared_views[index] for index in indices])

    def _depth_spectrum(self, depth_indices, phi, normalized, piston_tip_tilt):
        if (phi is None and depth_indices.numel() == self.Nz
                and torch.equal(depth_indices, self._full_depth_indices)):
            self.full_depth_spectrum_reuses += 1
            return self.kbase
        return super()._depth_spectrum(depth_indices, phi, normalized, piston_tip_tilt)

    def clear_angle_cache(self):
        # Clear the request-block lookup while retaining the precomputed views.
        # Subsequent requests take the no-FFT subset path.
        self._angle_kernel_cache.clear()
