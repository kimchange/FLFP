"""PyTorch implementation of the conventional light-field PSF model.

The propagation and phase-space rearrangement follow the MATLAB implementation
released with:

    Wu et al., Cell 184, 3318--3332 (2021).

Unlike ``lfpsf_torch_batch.py``, a requested angular PSF is not generated
directly.  All ``Nnum**2`` intra-cell source displacements are propagated
through the complete microlens array and then rearranged into phase space.

The public class and method signatures intentionally follow
``lfpsf_torch_batch.py`` so the two formulations can be benchmarked with the
same calling code.  The objective pupil model, Zernike convention, sampling
grid, and default microlens aperture also match that file.  Options are
provided for the zero-padded shifts and detector-pixel summation used by the
original MATLAB code.  The MATLAB-only adaptive support search and 0.5% hard
threshold are deliberately omitted: the requested ``lfpsf_shape`` fixes the
support, while hard thresholding would change the physical response and bias a
numerical-equivalence or runtime comparison.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


IndexLike = Union[int, slice, Sequence[int], np.ndarray, torch.Tensor, None]
ViewLike = Union[int, Sequence[int], np.ndarray, torch.Tensor]


class PsfGenerator5D:
    """Conventional Cell-2021-style light-field PSF generator.

    Parameters are compatible with :class:`lfpsf_torch_batch.PsfGenerator5D`.

    Additional parameters
    ---------------------
    detector_binning:
        ``"sample"`` matches ``lfpsf_torch_batch.py`` by sampling the coherent
        field at the center of every OSR block and multiplying it by ``OSR``.
        ``"sum"`` reproduces the intensity summation of MATLAB
        ``pixelBinning.m`` and is available through :meth:`incoherent_psf`.
    shift_mode:
        ``"zero"`` reproduces MATLAB ``im_shift2/3``. ``"circular"`` is useful
        for checking the exact periodic-boundary equivalence to the
        shift--Fresnel formulation.
    normalize_raw:
        Normalize every depth of every raw source-displacement intensity PSF
        before phase-space rearrangement, as in the MATLAB implementation.
        It is disabled by default for a fair amplitude comparison with
        ``lfpsf_torch_batch.py``.
    circle_aperture:
        ``True`` matches the current PyTorch shifted-Fresnel implementation.
        Set to ``False`` for the square microlens cell in the original MATLAB
        ``calcML.m``.
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
        detector_binning: str = "sample",
        shift_mode: str = "zero",
        normalize_raw: bool = False,
        circle_aperture: bool = True,
    ) -> None:
        del n_threads  # Kept only for API compatibility.

        lfpsf_shape = tuple(int(value) for value in lfpsf_shape)
        if len(lfpsf_shape) != 5:
            raise ValueError("lfpsf_shape must be (Nnum, Nnum, Nz, Ny, Nx)")
        if lfpsf_shape[:2] != (Nnum, Nnum):
            raise ValueError(
                "lfpsf_shape[:2] must equal (Nnum, Nnum); "
                f"received {lfpsf_shape[:2]} and Nnum={Nnum}"
            )
        if Nnum % 2 != 1:
            raise ValueError("Nnum must be odd, as required by the MATLAB model")
        if OSR < 1:
            raise ValueError("OSR must be a positive integer")
        if lfpsf_shape[-2] % Nnum or lfpsf_shape[-1] % Nnum:
            raise ValueError(
                "The output Ny and Nx must be divisible by Nnum for exact "
                "phase-space rearrangement"
            )
        if detector_binning not in {"sample", "sum"}:
            raise ValueError("detector_binning must be 'sample' or 'sum'")
        if shift_mode not in {"zero", "circular"}:
            raise ValueError("shift_mode must be 'zero' or 'circular'")

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
        self.device = torch.device(device)
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.zernike_coef_in_lambda = bool(zernike_coef_in_lambda)
        self.detector_binning = detector_binning
        self.shift_mode = shift_mode
        self.normalize_raw = bool(normalize_raw)
        self.circle_aperture = bool(circle_aperture)

        # The objective-space frequency grid matches lfpsf_torch_batch.py.
        self.dx = self.pixelPitch / self.M / self.OSR
        self.dy = self.dx
        kx = torch.fft.fftfreq(
            self.Nx, d=self.dx, device=self.device, dtype=self.real_dtype
        )
        ky = torch.fft.fftfreq(
            self.Ny, d=self.dy, device=self.device, dtype=self.real_dtype
        )
        z = self.dz * (
            torch.arange(self.Nz, device=self.device, dtype=self.real_dtype)
            - self.Nz // 2
        )

        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
        radial_frequency = torch.hypot(kx_grid, ky_grid)
        self.kcut = self.na_detection / self.lam_detection
        self.kmask2 = radial_frequency <= self.kcut
        self.kmask3 = self.kmask2.unsqueeze(0).expand(self.Nz, -1, -1)

        axial_argument = (
            self.n**2 - radial_frequency.square() * self.lam_detection**2
        )
        axial_root = torch.sqrt(torch.clamp(axial_argument, min=0.0))
        kprop = torch.exp(
            2j
            * torch.pi
            / self.lam_detection
            * z[:, None, None]
            * axial_root[None, :, :]
        )
        self.kbase = self.kmask2.to(self.real_dtype).unsqueeze(0) * kprop

        self.krho = radial_frequency / self.kcut
        self.kphi = torch.atan2(ky_grid, kx_grid)

        pupil_fraction = self.kmask2.to(self.real_dtype).mean()
        self.energy_ratio_complex = torch.sqrt(pupil_fraction)

        # The MLA and sensor-propagation grids are in image-space units.
        self.modulation_dx = self.pixelPitch / self.OSR
        self.MLARRAY = self._make_periodic_mla().to(self.complex_dtype)
        self.fresnel_transfer = self._make_fresnel_transfer()

    # ------------------------------------------------------------------
    # Pupil and aberration model
    # ------------------------------------------------------------------
    @staticmethod
    def _osa_to_nm(index: int) -> Tuple[int, int]:
        """Convert the zero-based OSA/ANSI index to radial orders (n, m)."""

        n_order = int((math.sqrt(8 * index + 1) - 1) / 2)
        m_order = 2 * index - n_order * (n_order + 2)
        return n_order, m_order

    def zernike_polynomial(
        self, idx: int, normalized: bool = True
    ) -> torch.Tensor:
        """Evaluate the OSA/ANSI Zernike mode used by lfpsf_torch_batch.py."""

        n_order, m_order = self._osa_to_nm(int(idx))
        abs_m = abs(m_order)
        radial = torch.zeros_like(self.krho)

        if (n_order - abs_m) % 2:
            return radial

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
            polynomial = norm * polynomial
        return polynomial

    def masked_phase_array(
        self,
        phi: Union[Sequence[float], torch.Tensor],
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        """Build a pupil phase map from Zernike coefficients."""

        coefficients = torch.as_tensor(
            phi, dtype=self.real_dtype, device=self.device
        ).reshape(-1)
        phase = torch.zeros_like(self.krho)
        index_offset = 0 if piston_tip_tilt else 3
        for coefficient_index, coefficient in enumerate(coefficients):
            phase = phase + coefficient * self.zernike_polynomial(
                coefficient_index + index_offset, normalized=normalized
            )
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
        values = phase[self.kmask2]
        return torch.sqrt(torch.mean((values - values.mean()).square()))

    def _pupil_spectrum(
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

    def _objective_field(
        self,
        depth_indices: torch.Tensor,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool,
        piston_tip_tilt: bool,
    ) -> torch.Tensor:
        spectrum = self._pupil_spectrum(
            depth_indices, phi, normalized, piston_tip_tilt
        )
        field = torch.fft.ifft2(spectrum, dim=(-2, -1))
        field = torch.fft.fftshift(field, dim=(-2, -1))
        return field / self.energy_ratio_complex

    # ------------------------------------------------------------------
    # Periodic modulation and Fresnel propagation
    # ------------------------------------------------------------------
    def _make_periodic_mla(self) -> torch.Tensor:
        """Construct the ideal periodic MLA on the oversampled image grid."""

        y = self.modulation_dx * (
            torch.arange(self.Ny, device=self.device, dtype=self.real_dtype)
            - self.Ny // 2
        )
        x = self.modulation_dx * (
            torch.arange(self.Nx, device=self.device, dtype=self.real_dtype)
            - self.Nx // 2
        )

        # Local coordinates in the centered elementary cell [-pitch/2,pitch/2).
        local_y = torch.remainder(y + self.MLPitch / 2, self.MLPitch)
        local_y = local_y - self.MLPitch / 2
        local_x = torch.remainder(x + self.MLPitch / 2, self.MLPitch)
        local_x = local_x - self.MLPitch / 2
        local_y_grid, local_x_grid = torch.meshgrid(
            local_y, local_x, indexing="ij"
        )

        wave_number = 2 * torch.pi / self.lam_detection
        phase = torch.exp(
            -1j
            * wave_number
            / (2 * self.fml)
            * (local_y_grid.square() + local_x_grid.square())
        )
        if self.circle_aperture:
            radius = self.MLPitch / 2
            aperture = (
                local_y_grid.square() + local_x_grid.square()
            ) <= radius**2
            phase = phase * aperture
        return phase

    def _make_fresnel_transfer(self) -> torch.Tensor:
        fx = torch.fft.fftfreq(
            self.Nx,
            d=self.modulation_dx,
            device=self.device,
            dtype=self.real_dtype,
        )
        fy = torch.fft.fftfreq(
            self.Ny,
            d=self.modulation_dx,
            device=self.device,
            dtype=self.real_dtype,
        )
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        wave_number = 2 * torch.pi / self.lam_detection
        global_phase = torch.exp(
            torch.tensor(
                1j * wave_number * self.fml,
                device=self.device,
                dtype=self.complex_dtype,
            )
        )
        return global_phase * torch.exp(
            -1j
            * 2
            * torch.pi**2
            * (fx_grid.square() + fy_grid.square())
            * self.fml
            / wave_number
        )

    def fresnel2d(self, field: torch.Tensor) -> torch.Tensor:
        """Propagate centered spatial fields from the MLA to the sensor."""

        field_frequency = torch.fft.fft2(field, dim=(-2, -1))
        return torch.fft.ifft2(
            field_frequency * self.fresnel_transfer, dim=(-2, -1)
        )

    # ------------------------------------------------------------------
    # MATLAB-compatible shifts, detector binning, and rearrangement
    # ------------------------------------------------------------------
    @staticmethod
    def _zero_shift_2d(
        image: torch.Tensor, shift_y: int, shift_x: int
    ) -> torch.Tensor:
        """Integer translation with zero fill, matching MATLAB im_shift2/3."""

        height, width = image.shape[-2:]
        if abs(shift_y) >= height or abs(shift_x) >= width:
            return torch.zeros_like(image)

        output = torch.zeros_like(image)
        src_y0 = max(-shift_y, 0)
        src_y1 = min(height - shift_y, height)
        dst_y0 = max(shift_y, 0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        src_x0 = max(-shift_x, 0)
        src_x1 = min(width - shift_x, width)
        dst_x0 = max(shift_x, 0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = image[
            ..., src_y0:src_y1, src_x0:src_x1
        ]
        return output

    def _shift(
        self, image: torch.Tensor, shift_y: int, shift_x: int
    ) -> torch.Tensor:
        if self.shift_mode == "circular":
            return torch.roll(
                image, shifts=(shift_y, shift_x), dims=(-2, -1)
            )
        return self._zero_shift_2d(image, shift_y, shift_x)

    def _sample_coherent_field(self, field: torch.Tensor) -> torch.Tensor:
        offset = self.OSR // 2
        sampled = field[
            ...,
            offset : offset + self.out_Ny * self.OSR : self.OSR,
            offset : offset + self.out_Nx * self.OSR : self.OSR,
        ]
        return sampled * self.OSR

    def _bin_intensity(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.detector_binning == "sample":
            offset = self.OSR // 2
            sampled = intensity[
                ...,
                offset : offset + self.out_Ny * self.OSR : self.OSR,
                offset : offset + self.out_Nx * self.OSR : self.OSR,
            ]
            return sampled * self.OSR**2

        leading_shape = intensity.shape[:-2]
        reshaped = intensity.reshape(
            *leading_shape,
            self.out_Ny,
            self.OSR,
            self.out_Nx,
            self.OSR,
        )
        return reshaped.sum(dim=(-3, -1))

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

    def _normalize_views(
        self, u: ViewLike, v: ViewLike
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_tensor = torch.as_tensor(u, device=self.device, dtype=torch.long)
        v_tensor = torch.as_tensor(v, device=self.device, dtype=torch.long)
        u_tensor = u_tensor.reshape(-1)
        v_tensor = v_tensor.reshape(-1)
        if u_tensor.numel() != v_tensor.numel():
            raise ValueError("u and v must contain the same number of views")
        if torch.any((u_tensor < 0) | (u_tensor >= self.Nnum)):
            raise IndexError(f"u must lie in [0, {self.Nnum})")
        if torch.any((v_tensor < 0) | (v_tensor >= self.Nnum)):
            raise IndexError(f"v must lie in [0, {self.Nnum})")
        return u_tensor, v_tensor

    @staticmethod
    def phase_space_rearrange(raw: torch.Tensor) -> torch.Tensor:
        """Vectorized version of lines 220--259 in main_computePSF.m.

        ``raw`` must have shape ``[D, Q, Q, H, W]`` and already include the
        low-resolution source-dependent ``im_shift3`` alignment.  The returned
        tensor has shape ``[Q, Q, D, H, W]``.
        """

        if raw.ndim != 5:
            raise ValueError("raw must have shape [D, Q, Q, H, W]")
        depth_count, q_y, q_x, height, width = raw.shape
        if q_y != q_x:
            raise ValueError("The two angular dimensions must have equal size")
        q = q_y
        if height % q or width % q:
            raise ValueError("H and W must be divisible by Q")

        blocks_y, blocks_x = height // q, width // q
        arranged = raw.reshape(
            depth_count, q, q, blocks_y, q, blocks_x, q
        )
        arranged = torch.flip(arranged, dims=(1, 2))
        arranged = arranged.permute(4, 6, 0, 3, 1, 5, 2)
        return arranged.reshape(q, q, depth_count, height, width)

    def _propagate_and_rearrange(
        self,
        u: ViewLike,
        v: ViewLike,
        d: IndexLike,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool,
        piston_tip_tilt: bool,
        return_complex: bool,
    ) -> torch.Tensor:
        view_u, view_v = self._normalize_views(u, v)
        depth_indices = self._normalize_depth_indices(d)
        depth_count = depth_indices.numel()
        view_count = view_u.numel()

        if return_complex and self.detector_binning != "sample":
            raise ValueError(
                "coherent_psf requires detector_binning='sample'; intensity "
                "summation has no unique coherent-field equivalent"
            )

        objective = self._objective_field(
            depth_indices, phi, normalized, piston_tip_tilt
        )
        output_dtype = self.complex_dtype if return_complex else self.real_dtype
        output = torch.zeros(
            (
                view_count,
                depth_count,
                self.out_Ny,
                self.out_Nx,
            ),
            dtype=output_dtype,
            device=self.device,
        )

        center = self.Nnum // 2
        for source_y in range(self.Nnum):
            high_shift_y = self.OSR * (source_y - center)
            output_residue_y = self.Nnum - 1 - source_y

            for source_x in range(self.Nnum):
                high_shift_x = self.OSR * (source_x - center)
                output_residue_x = self.Nnum - 1 - source_x

                shifted_objective = self._shift(
                    objective, high_shift_y, high_shift_x
                )
                modulated = shifted_objective * self.MLARRAY
                detector_field = self.fresnel2d(modulated)

                # MATLAB line 143: undo the oversampled input displacement.
                detector_field = self._shift(
                    detector_field, -high_shift_y, -high_shift_x
                )

                if return_complex:
                    raw_response = self._sample_coherent_field(detector_field)
                    if self.normalize_raw:
                        energy = raw_response.abs().square().sum(
                            dim=(-2, -1), keepdim=True
                        )
                        raw_response = raw_response / torch.sqrt(
                            energy.clamp_min(torch.finfo(self.real_dtype).eps)
                        )
                else:
                    raw_response = self._bin_intensity(
                        detector_field.abs().square()
                    )
                    if self.normalize_raw:
                        energy = raw_response.sum(
                            dim=(-2, -1), keepdim=True
                        )
                        raw_response = raw_response / energy.clamp_min(
                            torch.finfo(self.real_dtype).eps
                        )

                # MATLAB line 231: scanning-light-field alignment.
                aligned = self._shift(
                    raw_response,
                    source_y - center,
                    source_x - center,
                )

                # Stream the phase-space permutation without retaining the
                # enormous [D,Q,Q,H,W] intermediate tensor.
                blocks_y = self.out_Ny // self.Nnum
                blocks_x = self.out_Nx // self.Nnum
                angular_samples = aligned.reshape(
                    depth_count,
                    blocks_y,
                    self.Nnum,
                    blocks_x,
                    self.Nnum,
                ).permute(2, 4, 0, 1, 3)
                selected_samples = angular_samples[view_u, view_v]
                output[
                    :,
                    :,
                    output_residue_y :: self.Nnum,
                    output_residue_x :: self.Nnum,
                ] = selected_samples

        return output

    # ------------------------------------------------------------------
    # Public API compatible with lfpsf_torch_batch.py
    # ------------------------------------------------------------------
    def coherent_psf(
        self,
        u: ViewLike,
        v: ViewLike,
        d: IndexLike,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        """Return selected coherent angular PSFs as [views, depths, Ny, Nx].

        Even for one requested view, all ``Nnum**2`` raw source-displacement
        propagations are evaluated before phase-space selection.
        """

        return self._propagate_and_rearrange(
            u,
            v,
            d,
            phi,
            normalized,
            piston_tip_tilt,
            return_complex=True,
        )

    def incoherent_psf(
        self,
        u: ViewLike,
        v: ViewLike,
        d: IndexLike,
        phi: Optional[Union[Sequence[float], torch.Tensor]],
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        """Return selected intensity PSFs as [views, depths, Ny, Nx]."""

        return self._propagate_and_rearrange(
            u,
            v,
            d,
            phi,
            normalized,
            piston_tip_tilt,
            return_complex=False,
        )

    def all_views_incoherent_psf(
        self,
        d: IndexLike = None,
        phi: Optional[Union[Sequence[float], torch.Tensor]] = None,
        normalized: bool = False,
        piston_tip_tilt: bool = False,
    ) -> torch.Tensor:
        """Return all angular PSFs with shape [Nnum,Nnum,D,Ny,Nx]."""

        u, v = np.unravel_index(
            np.arange(self.Nnum**2), (self.Nnum, self.Nnum)
        )
        psf = self.incoherent_psf(
            u,
            v,
            d,
            phi,
            normalized=normalized,
            piston_tip_tilt=piston_tip_tilt,
        )
        return psf.reshape(
            self.Nnum,
            self.Nnum,
            psf.shape[1],
            self.out_Ny,
            self.out_Nx,
        )


Cell2021PsfGenerator = PsfGenerator5D


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
        shift_mode="zero",
    )
    initialized = time.perf_counter()

    center = Nnum // 2
    uu,vv = torch.meshgrid(torch.arange(Nnum), torch.arange(Nnum),)
    blksize = 3
    allzz = torch.arange(lfpsf_shape[-3])
    single_point = generator.incoherent_psf(uu, vv, allzz[0:blksize], None, normalized=True, piston_tip_tilt=False)# .cpu()
    for blk in range(1, (len(allzz)-1) // blksize + 1):
        single_point_tmp = generator.incoherent_psf(uu, vv, allzz[blk*blksize:(blk+1)*blksize], None, normalized=True, piston_tip_tilt=False)
        # single_point = torch.cat((single_point, single_point_tmp.to(single_point.device) ), dim=-3)
    # result = generator.incoherent_psf(
    #     uu, vv, slice(None), None
    # )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    finished = time.perf_counter()
    # single_point = single_point.reshape( (-1,) + lfpsf_shape[-3:])
    # import tifffile
    # tifffile.imwrite('psf_ori.tif', single_point.cpu().numpy().astype(np.float32), metadata={'axes': 'TZYX'}, compression=None)
    import tifffile


    print("device:", device)
    print("output shape:", tuple(single_point.shape))
    print("initialization time [s]:", initialized - start)
    print("PSF time [s]:", finished - initialized)
    print("total time [s]:", finished - start)
