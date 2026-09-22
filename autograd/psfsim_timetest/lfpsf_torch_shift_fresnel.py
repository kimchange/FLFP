"""GPU-vectorized shifted-Fresnel light-field PSF generator.

This module preserves the public API and numerical model of
``lfpsf_torch_batch.py`` while removing its main CPU/Python bottlenecks:

* no global ``torch.cuda.FloatTensor`` dependency;
* no depth-sized 3-D coordinate meshgrids during initialization;
* no Python loop or repeated ``torch.cat`` for angular kernel shifts;
* no nested Python loops for the periodic microlens array;
* lazily cached, GPU-resident Zernike modes;
* batched FFTs over views and depth chunks;
* an intensity-only path that avoids retaining a complex output tensor.

The class returns the same layout as the original implementation:
``[number_of_views, number_of_depths, Ny, Nx]``.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch


IndexLike = Union[int, slice, Sequence[int], np.ndarray, torch.Tensor, None]
ViewLike = Union[int, Sequence[int], np.ndarray, torch.Tensor]


class PsfGenerator5D:
    """Vectorized implementation compatible with ``lfpsf_torch_batch.py``.

    Parameters added to the original constructor
    --------------------------------------------
    depth_batch_size:
        Number of axial planes transformed together. ``None`` processes all
        requested depths in one batch. A moderate value (for example 4--16)
        reduces peak GPU memory without moving computation to the CPU.
    cache_angle_kernels:
        Cache frequency-domain shifted-Fresnel kernels on the selected device.
        This is useful when the same views are evaluated repeatedly during
        optimization, but consumes approximately one complex image per view.
    circle_aperture:
        Retains the circular microlens aperture used by the original PyTorch
        file. Set to ``False`` for a square microlens cell.
    input_views:
        Optional flattened view identifiers in row-major ``Nnum x Nnum``
        order, matching ``lfpsf_torch_batch4.py``. The new
        ``incoherent_psf(a, d, phi)`` interface uses ``a`` to select entries
        from this view catalogue. ``None`` exposes all ``Nnum**2`` views.
    """

    def __init__(
        self,
        lfpsf_shape: Tuple[int, int, int, int, int] = (13, 13, 101, 351, 351),
        MLPitch: float = 100e-6,
        dz: float = 0.2e-6,
        M: float = 63,
        lam_detection: float = 525e-9,
        n: float = 1.515,
        na_detection: float = 1.4,
        fml: float = 2100e-6,
        Nnum: int = 13,
        OSR: int = 3,
        n_threads: int = 4,
        device: Union[str, torch.device] = "cuda:0",
        zernike_coef_in_lambda: bool = True,
        depth_batch_size: Optional[int] = 8,
        cache_angle_kernels: bool = False,
        circle_aperture: bool = True,
        input_views: Optional[ViewLike] = None,
    ) -> None:
        del n_threads  # Preserved for call-site compatibility.

        lfpsf_shape = tuple(int(value) for value in lfpsf_shape)
        if len(lfpsf_shape) != 5:
            raise ValueError("lfpsf_shape must be (Nnum, Nnum, Nz, Ny, Nx)")
        if lfpsf_shape[:2] != (Nnum, Nnum):
            raise ValueError(
                "lfpsf_shape[:2] must equal (Nnum, Nnum); "
                f"received {lfpsf_shape[:2]} and Nnum={Nnum}"
            )
        if OSR < 1:
            raise ValueError("OSR must be a positive integer")
        if depth_batch_size is not None and depth_batch_size < 1:
            raise ValueError("depth_batch_size must be positive or None")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is unavailable"
            )

        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.lfpsf_shape = lfpsf_shape
        self.Nnum = int(Nnum)
        self.OSR = int(OSR)
        self.Nz = lfpsf_shape[-3]
        self.out_Ny = lfpsf_shape[-2]
        self.out_Nx = lfpsf_shape[-1]
        self.Ny = self.out_Ny * self.OSR
        self.Nx = self.out_Nx * self.OSR

        self.MLPitch = float(MLPitch)
        self.pixelPitch = self.MLPitch / self.Nnum
        self.M = float(M)
        self.dz = float(dz)
        self.lam_detection = float(lam_detection)
        self.n = float(n)
        self.na_detection = float(na_detection)
        self.fml = float(fml)
        self.zernike_coef_in_lambda = bool(zernike_coef_in_lambda)
        self.depth_batch_size = depth_batch_size
        self.cache_angle_kernels = bool(cache_angle_kernels)
        self.circle_aperture = bool(circle_aperture)

        self.dx = self.pixelPitch / self.M / self.OSR
        self.dy = self.dx
        self.k = torch.tensor(
            2 * torch.pi / self.lam_detection,
            device=self.device,
            dtype=self.real_dtype,
        )

        # A single 2-D objective-frequency grid is broadcast over depth. This
        # replaces the original KZ3/KY3/KX3 tensors.
        kx = torch.fft.fftfreq(
            self.Nx, d=self.dx, device=self.device, dtype=self.real_dtype
        )
        ky = torch.fft.fftfreq(
            self.Ny, d=self.dy, device=self.device, dtype=self.real_dtype
        )
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
        radial_frequency = torch.hypot(kx_grid, ky_grid)

        self.kcut = self.na_detection / self.lam_detection
        self.kmask2 = radial_frequency <= self.kcut
        self.kmask3 = self.kmask2.unsqueeze(0).expand(self.Nz, -1, -1)
        self.krho = radial_frequency / self.kcut
        self.kphi = torch.atan2(ky_grid, kx_grid)

        axial_argument = (
            self.n**2 - radial_frequency.square() * self.lam_detection**2
        )
        self._H = torch.sqrt(torch.clamp(axial_argument, min=0.0))
        z = self.dz * (
            torch.arange(self.Nz, device=self.device, dtype=self.real_dtype)
            - self.Nz // 2
        )
        self.z = z
        self.kprop = torch.exp(
            2j
            * torch.pi
            / self.lam_detection
            * z[:, None, None]
            * self._H[None, :, :]
        )
        self.kbase = self.kmask2.to(self.real_dtype).unsqueeze(0) * self.kprop
        self.energy_ratio_complex = torch.sqrt(
            self.kmask2.to(self.real_dtype).mean()
        )
        phase_angular_spectrum = -self._H[self.kmask2]
        self.rms_H = torch.sqrt(
            torch.mean(
                (
                    phase_angular_spectrum
                    - phase_angular_spectrum.mean()
                ).square()
            )
        )

        # Image-plane coordinates used for the periodic modulation and sensor
        # propagation, matching the original implementation.
        modulation_dx = self.pixelPitch / self.OSR
        x1space = modulation_dx * (
            torch.arange(self.Ny, device=self.device, dtype=self.real_dtype)
            - self.Ny // 2
        )
        x2space = modulation_dx * (
            torch.arange(self.Nx, device=self.device, dtype=self.real_dtype)
            - self.Nx // 2
        )
        cell_samples = self.Nnum * self.OSR
        x1MLspace = modulation_dx * (
            torch.arange(
                -(cell_samples // 2),
                cell_samples // 2 + 1,
                device=self.device,
                dtype=self.real_dtype,
            )
        )
        x2MLspace = modulation_dx * (
            torch.arange(
                -(cell_samples // 2),
                cell_samples // 2 + 1,
                device=self.device,
                dtype=self.real_dtype,
            )
        )
        # For odd cell_samples this is exactly cell_samples values. Preserve
        # the original arange convention for even values as well.
        if x1MLspace.numel() > cell_samples:
            x1MLspace = x1MLspace[:-1]
            x2MLspace = x2MLspace[:-1]

        self.MLARRAY = self.calcML(
            self.fml,
            self.k,
            x1MLspace,
            x2MLspace,
            x1space,
            x2space,
            circle_aperture=self.circle_aperture,
        ).to(device=self.device, dtype=self.complex_dtype)

        self.fresnel_2dkernel = self.get_fresnel2dkernel(
            self.Nx,
            self.Ny,
            modulation_dx,
            self.fml,
            self.k,
            device=self.device,
        )
        self.myzifftn = lambda value: torch.fft.ifftn(
            value, dim=(-2, -1)
        )

        self._spatial_y = torch.arange(self.Ny, device=self.device)
        self._spatial_x = torch.arange(self.Nx, device=self.device)
        self._zernike_cache: Dict[Tuple[int, bool], torch.Tensor] = {}
        self._angle_kernel_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...]], torch.Tensor
        ] = {}

        if input_views is None:
            self.input_views = torch.arange(
                self.Nnum**2, device=self.device, dtype=torch.long
            )
        else:
            self.input_views = torch.as_tensor(
                input_views, device=self.device, dtype=torch.long
            ).reshape(-1)
            if self.input_views.numel() == 0:
                raise ValueError("input_views must contain at least one view")
            if torch.any(
                (self.input_views < 0)
                | (self.input_views >= self.Nnum**2)
            ):
                raise IndexError(
                    f"input_views must lie in [0, {self.Nnum**2})"
                )

    # ------------------------------------------------------------------
    # Zernike functions
    # ------------------------------------------------------------------
    @staticmethod
    def _osa_to_nm(index: int) -> Tuple[int, int]:
        n_order = int((math.sqrt(8 * index + 1) - 1) / 2)
        m_order = 2 * index - n_order * (n_order + 2)
        return n_order, m_order

    def zernike_polynomial(
        self, idx: int, normalized: bool = True
    ) -> torch.Tensor:
        """Evaluate one OSA/ANSI Zernike mode on the GPU-resident pupil grid."""

        cache_key = (int(idx), bool(normalized))
        cached = self._zernike_cache.get(cache_key)
        if cached is not None:
            return cached

        n_order, m_order = self._osa_to_nm(int(idx))
        abs_m = abs(m_order)
        radial = torch.zeros_like(self.krho)

        if (n_order - abs_m) % 2 == 0:
            for s_index in range((n_order - abs_m) // 2 + 1):
                coefficient = (
                    (-1) ** s_index
                    * math.factorial(n_order - s_index)
                    / (
                        math.factorial(s_index)
                        * math.factorial((n_order + abs_m) // 2 - s_index)
                        * math.factorial((n_order - abs_m) // 2 - s_index)
                    )
                )
                radial = radial + coefficient * self.krho ** (
                    n_order - 2 * s_index
                )

        if m_order < 0:
            polynomial = radial * torch.sin(abs_m * self.kphi)
        elif m_order > 0:
            polynomial = radial * torch.cos(m_order * self.kphi)
        else:
            polynomial = radial

        if normalized:
            norm = math.sqrt(n_order + 1)
            if m_order != 0:
                norm *= math.sqrt(2)
            polynomial = polynomial * norm

        self._zernike_cache[cache_key] = polynomial
        return polynomial

    def _zernike_basis(
        self, mode_indices: Sequence[int], normalized: bool
    ) -> torch.Tensor:
        return torch.stack(
            [
                self.zernike_polynomial(index, normalized=normalized)
                for index in mode_indices
            ],
            dim=0,
        )

    def masked_phase_array(
        self,
        phi: Union[Sequence[float], torch.Tensor],
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        """Convert Zernike coefficients to a pupil phase without CPU loops."""

        coefficients = torch.as_tensor(
            phi, device=self.device, dtype=self.real_dtype
        ).reshape(-1)
        start_index = 0 if piston_tip_tilt else 3
        mode_indices = range(start_index, start_index + coefficients.numel())
        basis = self._zernike_basis(mode_indices, normalized=normalized)
        phase = torch.einsum("c,cyx->yx", coefficients, basis)
        return self.kmask2.to(self.real_dtype) * phase

    def calcRMS(
        self,
        phi: Union[Sequence[float], torch.Tensor],
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        phase = self.masked_phase_array(
            phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt
        )
        phase = phase[self.kmask2]
        return torch.sqrt(torch.mean((phase - phase.mean()).square()))

    # ------------------------------------------------------------------
    # View kernels and batched PSF evaluation
    # ------------------------------------------------------------------
    def _normalize_views(
        self, u: ViewLike, v: ViewLike
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_tensor = torch.as_tensor(u, device=self.device, dtype=torch.long)
        v_tensor = torch.as_tensor(v, device=self.device, dtype=torch.long)
        u_tensor = u_tensor.reshape(-1)
        v_tensor = v_tensor.reshape(-1)
        if u_tensor.numel() != v_tensor.numel():
            raise ValueError("u and v must contain the same number of views")
        if u_tensor.numel() == 0:
            raise ValueError("At least one angular view must be requested")
        if torch.any((u_tensor < 0) | (u_tensor >= self.Nnum)):
            raise IndexError(f"u must lie in [0, {self.Nnum})")
        if torch.any((v_tensor < 0) | (v_tensor >= self.Nnum)):
            raise IndexError(f"v must lie in [0, {self.Nnum})")
        return u_tensor, v_tensor

    def _views_from_catalogue(
        self, a: IndexLike
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve batch4-style catalogue positions into ``(u, v)`` pairs."""

        catalogue_positions = torch.arange(
            self.input_views.numel(), device=self.device
        )
        if a is None:
            selected_positions = catalogue_positions
        elif isinstance(a, slice):
            selected_positions = catalogue_positions[a]
        else:
            if isinstance(a, range):
                a = list(a)
            selected_positions = torch.as_tensor(a, device=self.device)
            if selected_positions.dtype == torch.bool:
                selected_positions = selected_positions.reshape(-1)
                if (
                    selected_positions.numel()
                    != self.input_views.numel()
                ):
                    raise IndexError(
                        "A boolean view mask must match len(input_views)"
                    )
                selected_positions = torch.nonzero(
                    selected_positions, as_tuple=False
                ).reshape(-1)
            else:
                selected_positions = selected_positions.to(
                    dtype=torch.long
                ).reshape(-1)
                selected_positions = torch.where(
                    selected_positions < 0,
                    selected_positions + self.input_views.numel(),
                    selected_positions,
                )

        if selected_positions.numel() == 0:
            raise ValueError("At least one angular view must be requested")
        if torch.any(
            (selected_positions < 0)
            | (selected_positions >= self.input_views.numel())
        ):
            raise IndexError(
                f"a must index the configured view catalogue with "
                f"length {self.input_views.numel()}"
            )

        flat_views = self.input_views.index_select(
            0, selected_positions.to(dtype=torch.long)
        )
        u = torch.div(flat_views, self.Nnum, rounding_mode="floor")
        v = torch.remainder(flat_views, self.Nnum)
        return u, v

    def _normalize_depth_indices(self, d: IndexLike) -> torch.Tensor:
        all_indices = torch.arange(self.Nz, device=self.device)
        if d is None:
            return all_indices
        if isinstance(d, slice):
            return all_indices[d]
        indices = torch.as_tensor(d, device=self.device, dtype=torch.long)
        indices = indices.reshape(-1)
        indices = torch.where(indices < 0, indices + self.Nz, indices)
        if torch.any((indices < 0) | (indices >= self.Nz)):
            raise IndexError(f"depth index is outside [0, {self.Nz})")
        return indices

    def _normalize_depth_coordinates(
        self, d: IndexLike
    ) -> torch.Tensor:
        """Return batch4-style axial coordinates measured in units of ``dz``."""

        stored_coordinates = torch.arange(
            self.Nz, device=self.device, dtype=self.real_dtype
        ) - self.Nz // 2
        if d is None:
            return stored_coordinates
        if isinstance(d, slice):
            return stored_coordinates[d]
        if isinstance(d, range):
            d = list(d)
        coordinates = torch.as_tensor(
            d, device=self.device, dtype=self.real_dtype
        ).reshape(-1)
        if coordinates.numel() == 0:
            raise ValueError("At least one depth coordinate is required")
        if not torch.isfinite(coordinates).all():
            raise ValueError("Depth coordinates must be finite")
        return coordinates

    def _batched_roll_fresnel(
        self, shifts_y: torch.Tensor, shifts_x: torch.Tensor
    ) -> torch.Tensor:
        """Apply independent circular 2-D shifts to one kernel on the GPU."""

        rows = (
            self._spatial_y.unsqueeze(0) - shifts_y.unsqueeze(1)
        ) % self.Ny
        cols = (
            self._spatial_x.unsqueeze(0) - shifts_x.unsqueeze(1)
        ) % self.Nx
        return self.fresnel_2dkernel[
            rows[:, :, None], cols[:, None, :]
        ]

    def _angle_kernels(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        cache_key = None
        if self.cache_angle_kernels:
            # The host transfer is performed only when caching is requested;
            # the default path remains entirely device resident.
            cache_key = (
                tuple(int(value) for value in u.detach().cpu().tolist()),
                tuple(int(value) for value in v.detach().cpu().tolist()),
            )
            cached = self._angle_kernel_cache.get(cache_key)
            if cached is not None:
                return cached

        center = self.Nnum // 2
        shifts_y = -(u - center) * self.OSR
        shifts_x = -(v - center) * self.OSR
        shifted_fresnel = self._batched_roll_fresnel(shifts_y, shifts_x)
        spatial_kernel = shifted_fresnel * self.MLARRAY.unsqueeze(0)
        kernel_frequency = torch.fft.fft2(
            torch.fft.ifftshift(spatial_kernel, dim=(-2, -1)),
            dim=(-2, -1),
        )

        if self.cache_angle_kernels:
            assert cache_key is not None
            self._angle_kernel_cache[cache_key] = kernel_frequency
        return kernel_frequency

    def clear_angle_cache(self) -> None:
        """Release cached view kernels without modifying optical parameters."""

        self._angle_kernel_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _depth_spectrum(
        self,
        depth_indices: torch.Tensor,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool,
        piston_tip_tilt: bool,
    ) -> torch.Tensor:
        spectrum = self.kbase.index_select(0, depth_indices)
        if phi is None:
            return spectrum

        phase = self.masked_phase_array(
            phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt
        )
        if self.zernike_coef_in_lambda:
            aberration = torch.exp(-2j * torch.pi * phase)
        else:
            aberration = torch.exp(
                -2j * torch.pi * phase / self.lam_detection
            )
        return spectrum * aberration.unsqueeze(0)

    def _coordinate_depth_spectrum(
        self,
        depth_coordinates: torch.Tensor,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool,
        piston_tip_tilt: bool,
    ) -> torch.Tensor:
        """Generate propagation spectra for arbitrary batch4-style depths."""

        axial_positions = depth_coordinates * self.dz
        spectrum = self.kmask2.to(self.real_dtype).unsqueeze(0) * torch.exp(
            2j
            * torch.pi
            / self.lam_detection
            * axial_positions[:, None, None]
            * self._H[None, :, :]
        )
        if phi is None:
            return spectrum

        phase = self.masked_phase_array(
            phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt
        )
        if self.zernike_coef_in_lambda:
            aberration = torch.exp(-2j * torch.pi * phase)
        else:
            aberration = torch.exp(
                -2j * torch.pi * phase / self.lam_detection
            )
        return spectrum * aberration.unsqueeze(0)

    def _compute_psf(
        self,
        u: ViewLike,
        v: ViewLike,
        d: IndexLike,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool,
        piston_tip_tilt: bool,
        intensity: bool,
        depth_coordinates: bool = False,
    ) -> torch.Tensor:
        view_u, view_v = self._normalize_views(u, v)
        angle_kernels = self._angle_kernels(view_u, view_v)
        if depth_coordinates:
            coordinates = self._normalize_depth_coordinates(d)
            depth_spectrum = self._coordinate_depth_spectrum(
                coordinates, phi, normalized, piston_tip_tilt
            )
        else:
            depth_indices = self._normalize_depth_indices(d)
            depth_spectrum = self._depth_spectrum(
                depth_indices, phi, normalized, piston_tip_tilt
            )

        depth_count = depth_spectrum.shape[0]
        chunk_size = self.depth_batch_size or max(depth_count, 1)
        output_chunks = []
        sample_offset = self.OSR // 2

        for start in range(0, depth_count, chunk_size):
            stop = min(start + chunk_size, depth_count)
            modulated_spectrum = (
                angle_kernels[:, None, :, :]
                * depth_spectrum[None, start:stop, :, :]
            )
            field = torch.fft.ifft2(modulated_spectrum, dim=(-2, -1))
            field = torch.fft.fftshift(field, dim=(-2, -1))
            field = field[
                ...,
                sample_offset
                : sample_offset + self.out_Ny * self.OSR
                : self.OSR,
                sample_offset
                : sample_offset + self.out_Nx * self.OSR
                : self.OSR,
            ]
            field = field * (self.OSR / self.energy_ratio_complex)
            if intensity:
                field = field.abs().square()
            output_chunks.append(field)

        if not output_chunks:
            output_dtype = self.real_dtype if intensity else self.complex_dtype
            return torch.empty(
                (view_u.numel(), 0, self.out_Ny, self.out_Nx),
                device=self.device,
                dtype=output_dtype,
            )
        return torch.cat(output_chunks, dim=1)

    def coherent_psf(
        self,
        u: ViewLike,
        v: ViewLike,
        d: IndexLike,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        return self._compute_psf(
            u,
            v,
            d,
            phi,
            normalized,
            piston_tip_tilt,
            intensity=False,
        )

    def _catalogue_incoherent_psf(
        self,
        a: IndexLike,
        d: IndexLike,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool,
        piston_tip_tilt: bool,
    ) -> torch.Tensor:
        u, v = self._views_from_catalogue(a)
        return self._compute_psf(
            u,
            v,
            d,
            phi,
            normalized,
            piston_tip_tilt,
            intensity=True,
            depth_coordinates=True,
        )

    @staticmethod
    def _bind_psf_arguments(
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
        names: Tuple[str, ...],
        defaults: Dict[str, object],
    ) -> Dict[str, object]:
        """Bind a small compatibility signature with Python-like errors."""

        if len(args) > len(names):
            raise TypeError(
                f"Expected at most {len(names)} positional arguments, "
                f"received {len(args)}"
            )
        values = dict(defaults)
        for name, value in zip(names, args):
            if name in kwargs:
                raise TypeError(f"Multiple values were supplied for '{name}'")
            values[name] = value
        for name, value in kwargs.items():
            if name not in names:
                raise TypeError(f"Unexpected keyword argument '{name}'")
            values[name] = value
        return values

    def incoherent_psf(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Compute selected incoherent PSFs with old and batch4-style calls.

        Preferred ``lfpsf_torch_batch4.py``-style interface::

            psf = generator.incoherent_psf(
                a=view_selection,
                d=depth_coordinates,
                phi=zernike_coefficients,
                normalized=True,
                piston_tip_tilt=True,
            )

        ``a`` selects positions in the constructor's ``input_views`` catalogue.
        If ``input_views`` was omitted, it directly selects flattened row-major
        view identifiers in ``[0, Nnum**2)``. A scalar, sequence, tensor,
        boolean mask, slice, or ``None`` (all configured views) is accepted.

        ``d`` contains axial coordinates relative to the focal plane, measured
        in units of ``dz`` as in batch4. Thus ``torch.arange(-50, 51)`` requests
        101 planes from ``-50*dz`` through ``50*dz``. ``None`` or a slice uses
        the centered depth grid implied by ``lfpsf_shape``.

        ``phi`` (alias ``zernike``) contains the differentiable Zernike
        coefficients. The returned layout is ``[view, depth, y, x]``.

        The original interface remains supported::

            psf = generator.incoherent_psf(u, v, d, phi)

        In that form, ``u`` and ``v`` are angular coordinates and ``d`` selects
        indices from the precomputed depth stack.
        """

        call_kwargs = dict(kwargs)
        original_keys = set(call_kwargs)
        legacy_keywords = bool({"u", "v"} & original_keys)
        catalogue_keywords = bool(
            {"a", "views", "depths", "zernike"} & original_keys
        )
        if legacy_keywords and catalogue_keywords:
            raise TypeError(
                "Do not mix (u, v) arguments with catalogue-style arguments"
            )

        # Positional calls are distinguished without changing existing code:
        # old: (u, v, d, phi[, normalized[, piston_tip_tilt]])
        # new: (a, d, phi[, normalized[, piston_tip_tilt]])
        if legacy_keywords:
            legacy_call = True
        elif catalogue_keywords:
            legacy_call = False
        elif len(args) >= 6:
            legacy_call = True
        elif len(args) == 5:
            legacy_call = not (
                isinstance(args[3], (bool, np.bool_))
                and isinstance(args[4], (bool, np.bool_))
            )
        elif len(args) == 4:
            legacy_call = not isinstance(args[3], (bool, np.bool_))
        elif len(args) == 3 and "phi" in original_keys:
            legacy_call = True
        elif len(args) == 2 and "d" in original_keys:
            legacy_call = True
        else:
            legacy_call = False

        if legacy_call:
            values = self._bind_psf_arguments(
                args,
                call_kwargs,
                ("u", "v", "d", "phi", "normalized", "piston_tip_tilt"),
                {
                    "d": None,
                    "phi": None,
                    "normalized": False,
                    "piston_tip_tilt": False,
                },
            )
            if "u" not in values or "v" not in values:
                raise TypeError("The original interface requires both u and v")
            return self._compute_psf(
                values["u"],
                values["v"],
                values["d"],
                values["phi"],
                bool(values["normalized"]),
                bool(values["piston_tip_tilt"]),
                intensity=True,
            )

        for alias, canonical in (
            ("views", "a"),
            ("depths", "d"),
            ("zernike", "phi"),
        ):
            if alias in call_kwargs:
                if canonical in call_kwargs:
                    raise TypeError(
                        f"Use either '{canonical}' or its alias '{alias}', "
                        "not both"
                    )
                call_kwargs[canonical] = call_kwargs.pop(alias)

        values = self._bind_psf_arguments(
            args,
            call_kwargs,
            ("a", "d", "phi", "normalized", "piston_tip_tilt"),
            {
                "a": None,
                "d": None,
                "phi": None,
                "normalized": False,
                "piston_tip_tilt": False,
            },
        )
        return self._catalogue_incoherent_psf(
            values["a"],
            values["d"],
            values["phi"],
            bool(values["normalized"]),
            bool(values["piston_tip_tilt"]),
        )

    # ------------------------------------------------------------------
    # Static optical helpers retained from lfpsf_torch_batch.py
    # ------------------------------------------------------------------
    @staticmethod
    def calcML(
        fml: float,
        k: Union[float, torch.Tensor],
        x1MLspace: torch.Tensor,
        x2MLspace: torch.Tensor,
        x1space: torch.Tensor,
        x2space: torch.Tensor,
        circle_aperture: bool,
    ) -> torch.Tensor:
        """Construct a periodic MLA without Python loops or sparse convolution."""

        device = x1space.device
        real_dtype = x1space.dtype
        wave_number = torch.as_tensor(
            k, device=device, dtype=real_dtype
        ).reshape(())

        local_y, local_x = torch.meshgrid(
            x1MLspace.to(device=device, dtype=real_dtype),
            x2MLspace.to(device=device, dtype=real_dtype),
            indexing="ij",
        )
        cell = torch.exp(
            -1j
            * wave_number
            / (2 * fml)
            * (local_y.square() + local_x.square())
        )

        if circle_aperture:
            iy = torch.arange(
                x1MLspace.numel(), device=device, dtype=real_dtype
            )
            ix = torch.arange(
                x2MLspace.numel(), device=device, dtype=real_dtype
            )
            denominator_y = max(x1MLspace.numel() // 2, 1)
            denominator_x = max(x2MLspace.numel() // 2, 1)
            normalized_y = iy / denominator_y - 1
            normalized_x = ix / denominator_x - 1
            aperture_y, aperture_x = torch.meshgrid(
                normalized_y, normalized_x, indexing="ij"
            )
            cell = cell * (
                aperture_y.square() + aperture_x.square() <= 1
            )

        center_y = int(torch.argmin(x1space.abs()))
        center_x = int(torch.argmin(x2space.abs()))
        cell_center_y = x1MLspace.numel() // 2
        cell_center_x = x2MLspace.numel() // 2

        indices_y = (
            torch.arange(x1space.numel(), device=device)
            - center_y
            + cell_center_y
        ) % x1MLspace.numel()
        indices_x = (
            torch.arange(x2space.numel(), device=device)
            - center_x
            + cell_center_x
        ) % x2MLspace.numel()
        return cell[indices_y[:, None], indices_x[None, :]]

    @staticmethod
    def get_fresnel2dkernel(
        Nx: int,
        Ny: int,
        dx0: float,
        z: float,
        k: Union[float, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        device = torch.device(device)
        wave_number = torch.as_tensor(
            k, device=device, dtype=torch.float32
        ).reshape(())
        u = torch.fft.fftfreq(Nx, d=dx0, device=device)
        v = torch.fft.fftfreq(Ny, d=dx0, device=device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        transfer = torch.exp(1j * wave_number * z) * torch.exp(
            -1j
            * 2
            * torch.pi**2
            * (u_grid.square() + v_grid.square())
            * z
            / wave_number
        )
        return torch.fft.fftshift(torch.fft.ifft2(transfer))

    @staticmethod
    def fresnel2d(
        x: torch.Tensor,
        dx0: float,
        z: float,
        k: Union[float, torch.Tensor],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Retain the original helper's frequency-domain return convention."""

        device = torch.device(device)
        wave_number = torch.as_tensor(
            k, device=device, dtype=torch.float32
        ).reshape(())
        u = torch.fft.fftfreq(x.shape[-1], d=dx0, device=device)
        v = torch.fft.fftfreq(x.shape[-2], d=dx0, device=device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        return x * torch.exp(1j * wave_number * z) * torch.exp(
            -1j
            * 2
            * torch.pi**2
            * (u_grid.square() + v_grid.square()).unsqueeze(0)
            * z
            / wave_number
        )


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # parameter 1
    Nnum = 15
    lfpsf_shape = (Nnum, Nnum, 101, 375, 375)
    OSR=3
    M=20
    n=1.406
    na_detection=1.05
    MLPitch=56.4e-6
    fml=536.4e-6
    lam_detection=525*1e-9
    dz=0.6e-6
    device = 'cuda:0'

    # parameter 2
    lfpsf_shape = (Nnum, Nnum, 101, 765, 765)
    OSR= 3
    M=7.85
    n=1
    na_detection=0.5
    MLPitch=56.4e-6
    fml=444.15e-6
    lam_detection=525*1e-9
    dz=5e-6
    device = 'cuda:0'


    start = time.perf_counter()
    generator = PsfGenerator5D(
        lfpsf_shape=lfpsf_shape,
        MLPitch=MLPitch,
        dz=dz,
        M=M,
        lam_detection=lam_detection,
        n=n,
        na_detection=na_detection,
        fml=fml,
        Nnum=Nnum,
        OSR=OSR,
        device=device,
        depth_batch_size=lfpsf_shape[-3] // 8,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    initialized = time.perf_counter()

    u, v = np.unravel_index(np.arange(Nnum**2), (Nnum, Nnum))
    # result = generator.incoherent_psf(u, v, slice(None), None)
    blksize = 3
    single_point = generator.incoherent_psf(u[0:blksize],v[0:blksize], slice(None,None,None), None, normalized=True, piston_tip_tilt=False)# .cpu()
    for blk in range(1, (len(u)-1) // blksize + 1):
        single_point_tmp = generator.incoherent_psf(u[blk*blksize:(blk+1)*blksize],v[blk*blksize:(blk+1)*blksize], slice(None,None,None), None, normalized=True, piston_tip_tilt=False)
        # single_point = torch.cat((single_point, single_point_tmp.to(single_point.device) ), dim=0)
    import tifffile
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    finished = time.perf_counter()

    print("device:", device)
    # print("output shape:", tuple(single_point.shape))
    print("initialization time [s]:", initialized - start)
    print("PSF time [s]:", finished - initialized)
