"""Differentiable light-field PSFs with batched angle-kernel initialization."""

import numpy as np
import torch

from typing import Union
import math

def osa_to_nm(j):
    """Convert an OSA/ANSI index to radial order n and azimuthal order m."""
    n = int((math.sqrt(8 * j + 1) - 1) / 2)
    m = 2 * j - n * (n + 2)
    return n, m

def generate_analytical_derivative_matrices(C=21, device='cpu'):
    """Build analytical Zernike derivative matrices Dx and Dy."""
    Dx = torch.zeros((C, C), dtype=torch.float32, device=device)
    Dy = torch.zeros((C, C), dtype=torch.float32, device=device)

    for j in range(C):
        n, m = osa_to_nm(j)

        for k in range(C):
            np, mp = osa_to_nm(k)

            if np >= n or (n - np) % 2 == 0:
                continue

            if abs(mp) != abs(m) + 1 and abs(mp) != abs(m) - 1:
                continue

            c = math.sqrt((n + 1) * (np + 1))
            if m == 0 or mp == 0:
                c *= math.sqrt(2)

            if (m >= 0 and mp >= 0) or (m < 0 and mp < 0):
                Dx[k, j] = c

            if m >= 0 and mp < 0:
                if abs(mp) == abs(m) + 1:
                    Dy[k, j] = c
                elif abs(mp) == abs(m) - 1:
                    Dy[k, j] = -c

            elif m < 0 and mp >= 0:
                if abs(mp) == abs(m) + 1:
                    Dy[k, j] = -c
                elif abs(mp) == abs(m) - 1:
                    Dy[k, j] = c

    return Dx, Dy

class PsfGenerator5D:

    def __init__(self, lfpsf_shape=(13,13,1,351,351), MLPitch=100e-6, dz=0.2e-6, xy_downsample = 1, M = 63, lam_detection=525e-9, n=1.515, na_detection=1.4, fml=2100e-6, Nnum = 13, OSR=3, n_threads=4, device='cuda:0', zernike_coef_in_lambda=True, input_views=None, circle_aperture=True, precompute_view_batch_size=3):

        lfpsf_shape = tuple(lfpsf_shape)
        psf_shape = (lfpsf_shape[-3], lfpsf_shape[-2]*OSR, lfpsf_shape[-1]*OSR)
        self.real_dtype = torch.float32

        self.Nz, self.Ny, self.Nx = psf_shape
        self.dz, self.dy, self.dx = dz, MLPitch / Nnum / M /OSR, MLPitch / Nnum / M /OSR
        self.fml = fml
        self.circle_aperture = circle_aperture
        self.complex_dtype = torch.complex64

        self.na_detection = na_detection
        self.lam_detection = lam_detection
        self.device = torch.device(device)
        self.xy_downsample = xy_downsample

        self.n = n
        self.Nnum = Nnum
        self.OSR = OSR

        self.pixelPitch = MLPitch / Nnum
        dtype = torch.float32

        self.k = torch.tensor(
                    2 * torch.pi / self.lam_detection,
                    device=self.device,
                    dtype=self.real_dtype,
                )
        self.hw_index = torch.fft.ifftshift(torch.arange(self.Nx//self.xy_downsample, device=self.device) - self.Nx//self.xy_downsample // 2)

        kx = torch.fft.fftfreq(
            self.Nx, d=self.dx, device=self.device, dtype=self.real_dtype
        )[self.hw_index]
        ky = torch.fft.fftfreq(
            self.Ny, d=self.dy, device=self.device, dtype=self.real_dtype
        )[self.hw_index]
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
        self.zernike_coef_in_lambda = zernike_coef_in_lambda

        self.phase_angular_spectrum = -self._H / self.rms_H
        self.phase_angular_spectrum[self.kmask2==False] = 0

        self.dz_rms = dz * self.rms_H / lam_detection

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

        if input_views is not None:
            u,v = np.unravel_index(input_views, shape=(Nnum,Nnum))
        else:
            u,v = np.unravel_index(np.arange(0,Nnum*Nnum), shape=(Nnum,Nnum))

        self.aperture_shift =  MLPitch / Nnum / lam_detection / fml / (1/self.Nx*xy_downsample/ MLPitch * Nnum)

        # Precompute view kernels in small batches to limit peak memory.
        if precompute_view_batch_size < 1:
            raise ValueError("precompute_view_batch_size must be positive")
        self.precompute_view_batch_size = precompute_view_batch_size
        self.precomputed_view_count = len(u)
        self.uv_mla_fresnel = torch.empty(
            (len(u), self.hw_index.numel(), self.hw_index.numel()),
            dtype=self.complex_dtype, device=self.device,
        )
        spatial_y = torch.arange(self.Ny, device=self.device)
        spatial_x = torch.arange(self.Nx, device=self.device)
        shifts_y = -torch.as_tensor((u - Nnum // 2) * self.OSR, device=self.device)
        shifts_x = -torch.as_tensor((v - Nnum // 2) * self.OSR, device=self.device)
        for start in range(0, len(u), precompute_view_batch_size):
            stop = min(start + precompute_view_batch_size, len(u))
            rows = (spatial_y[None, :] - shifts_y[start:stop, None]) % self.Ny
            cols = (spatial_x[None, :] - shifts_x[start:stop, None]) % self.Nx
            uv_fresnel = self.fresnel_2dkernel[rows[:, :, None], cols[:, None, :]]
            uv_mla_fresnel = uv_fresnel * self.MLARRAY.unsqueeze(0)
            kernel_frequency = torch.fft.fft2(
                torch.fft.ifftshift(uv_mla_fresnel, dim=(-2, -1)), dim=(-2, -1)
            )
            self.uv_mla_fresnel[start:stop] = torch.fft.fftshift(
                kernel_frequency[:, self.hw_index, :][:, :, self.hw_index]
                * self.kmask2, dim=(-2, -1)
            )

        self.myzifftn = lambda x: torch.fft.ifftn(x, dim=(-2,-1))

    def zernike_polynomial(self, idx, normalized = True):

        R = self.krho
        THETA = self.kphi

        if idx == 0:

            F = self.phase_angular_spectrum  # Mode 0 represents axial propagation.

        elif idx == 1:

            norm_factor = 2. if normalized else 1.
            F = norm_factor * torch.mul(R, torch.sin(THETA))
        elif idx == 2:

            norm_factor = 2. if normalized else 1.
            F = norm_factor * torch.mul(R, torch.cos(THETA))

        elif idx == 3:

            norm_factor = 6**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**2, torch.sin(2.*THETA))
        elif idx == 4:

            norm_factor = 3**0.5 if normalized else 1.
            F = norm_factor * (2.*R**2 - 1)
        elif idx == 5:

            norm_factor = 6**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**2, torch.cos(2.*THETA))

        elif idx == 6:

            norm_factor = 8**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**3, torch.sin(3.*THETA))
        elif idx == 7:

            norm_factor = 8**0.5 if normalized else 1.
            F = norm_factor * torch.mul(3.*R**3 - 2.*R, torch.sin(THETA))
        elif idx == 8:

            norm_factor = 8**0.5 if normalized else 1.
            F = norm_factor * torch.mul(3.*R**3 - 2.*R, torch.cos(THETA))
        elif idx == 9:

            norm_factor = 8**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**3, torch.cos(3.*THETA))

        elif idx == 10:

            norm_factor = 10**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**4, torch.sin(4.*THETA))
        elif idx == 11:

            norm_factor = 10**0.5 if normalized else 1.
            F = norm_factor * torch.mul(4.*R**4-3.*R**2, torch.sin(2.*THETA))
        elif idx == 12:

            norm_factor = 5**0.5 if normalized else 1.
            F = norm_factor * (6.*R**4-6.*R**2 + torch.ones_like(R))
        elif idx == 13:

            norm_factor = 10**0.5 if normalized else 1.
            F = norm_factor * torch.mul(4.*R**4-3.*R**2, torch.cos(2.*THETA))
        elif idx == 14:

            norm_factor = 10**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**4, torch.cos(4.*THETA))

        elif idx == 15:

            norm_factor = 12**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**5, torch.sin(5.*THETA))
        elif idx == 16:

            norm_factor = 12**0.5 if normalized else 1.
            F = norm_factor * torch.mul(5.*R**5 - 4.*R**3, torch.sin(3.*THETA))
        elif idx == 17:

            norm_factor = 12**0.5 if normalized else 1.
            F = norm_factor * torch.mul(10.*R**5 - 12.*R**3 + 3.*R, torch.sin(THETA))
        elif idx == 18:

            norm_factor = 12**0.5 if normalized else 1.
            F = norm_factor * torch.mul(10.*R**5 - 12.*R**3 + 3.*R, torch.cos(THETA))
        elif idx == 19:

            norm_factor = 12**0.5 if normalized else 1.
            F = norm_factor * torch.mul(5.*R**5 - 4.*R**3, torch.cos(3.*THETA))
        elif idx == 20:

            norm_factor = 12**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**5, torch.cos(5.*THETA))

        elif idx == 21:
            norm_factor = 14**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**6, torch.sin(6.*THETA))
        elif idx == 22:
            norm_factor = 14**0.5 if normalized else 1.
            F = norm_factor * torch.mul(6.*R**6 - 5.*R**4, torch.sin(4.*THETA))
        elif idx == 23:
            norm_factor = 14**0.5 if normalized else 1.
            F = norm_factor * torch.mul(15.*R**6 - 20.*R**4 + 6.*R**2, torch.sin(2.*THETA))
        elif idx == 24:
            norm_factor = 7**0.5 if normalized else 1.
            F = norm_factor * (20.*R**6 - 30.*R**4 + 12.*R**2 - torch.ones_like(R))
        elif idx == 25:
            norm_factor = 14**0.5 if normalized else 1.
            F = norm_factor * torch.mul(15.*R**6 - 20.*R**4 + 6.*R**2, torch.cos(2.*THETA))
        elif idx == 26:
            norm_factor = 14**0.5 if normalized else 1.
            F = norm_factor * torch.mul(6.*R**6 - 5.*R**4, torch.cos(4.*THETA))
        elif idx == 27:
            norm_factor = 14**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**6, torch.cos(6.*THETA))

        elif idx == 28:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**7, torch.sin(7.*THETA))
        elif idx == 29:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(7.*R**7 - 6.*R**5, torch.sin(5.*THETA))
        elif idx == 30:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(21.*R**7 - 30.*R**5 + 10.*R**3, torch.sin(3.*THETA))
        elif idx == 31:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(35.*R**7 - 60.*R**5 + 30.*R**3 - 4.*R, torch.sin(THETA))
        elif idx == 32:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(35.*R**7 - 60.*R**5 + 30.*R**3 - 4.*R, torch.cos(THETA))
        elif idx == 33:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(21.*R**7 - 30.*R**5 + 10.*R**3, torch.cos(3.*THETA))
        elif idx == 34:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(7.*R**7 - 6.*R**5, torch.cos(5.*THETA))
        elif idx == 35:
            norm_factor = 16**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**7, torch.cos(7.*THETA))

        elif idx == 36:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**8, torch.sin(8.*THETA))
        elif idx == 37:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(8.*R**8 - 7.*R**6, torch.sin(6.*THETA))
        elif idx == 38:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(28.*R**8 - 42.*R**6 + 15.*R**4, torch.sin(4.*THETA))
        elif idx == 39:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(56.*R**8 - 105.*R**6 + 60.*R**4 - 10.*R**2, torch.sin(2.*THETA))
        elif idx == 40:
            norm_factor = 9**0.5 if normalized else 1.
            F = norm_factor * (70.*R**8 - 140.*R**6 + 90.*R**4 - 20.*R**2 + torch.ones_like(R))
        elif idx == 41:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(56.*R**8 - 105.*R**6 + 60.*R**4 - 10.*R**2, torch.cos(2.*THETA))
        elif idx == 42:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(28.*R**8 - 42.*R**6 + 15.*R**4, torch.cos(4.*THETA))
        elif idx == 43:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(8.*R**8 - 7.*R**6, torch.cos(6.*THETA))
        elif idx == 44:
            norm_factor = 18**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**8, torch.cos(8.*THETA))

        elif idx == 45:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**9, torch.sin(9.*THETA))
        elif idx == 46:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(9.*R**9 - 8.*R**7, torch.sin(7.*THETA))
        elif idx == 47:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(36.*R**9 - 56.*R**7 + 21.*R**5, torch.sin(5.*THETA))
        elif idx == 48:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(84.*R**9 - 168.*R**7 + 105.*R**5 - 20.*R**3, torch.sin(3.*THETA))
        elif idx == 49:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(126.*R**9 - 280.*R**7 + 210.*R**5 - 60.*R**3 + 5.*R, torch.sin(THETA))
        elif idx == 50:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(126.*R**9 - 280.*R**7 + 210.*R**5 - 60.*R**3 + 5.*R, torch.cos(THETA))
        elif idx == 51:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(84.*R**9 - 168.*R**7 + 105.*R**5 - 20.*R**3, torch.cos(3.*THETA))
        elif idx == 52:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(36.*R**9 - 56.*R**7 + 21.*R**5, torch.cos(5.*THETA))
        elif idx == 53:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(9.*R**9 - 8.*R**7, torch.cos(7.*THETA))
        elif idx == 54:
            norm_factor = 20**0.5 if normalized else 1.
            F = norm_factor * torch.mul(R**9, torch.cos(9.*THETA))

        else:
            raise

        return F

    def masked_phase_array(self, phi, normalized=False, piston_tip_tilt=False):

        _phase = torch.zeros_like(self.krho)
        for j in range(len(phi)):
            if not piston_tip_tilt:
                _phase += phi[j] * self.zernike_polynomial(j+3, normalized)
            else:
                _phase += phi[j] * self.zernike_polynomial(j, normalized)

        return self.kmask2 * _phase

    def coherent_psf_v3(self, u, v, d, phi, normalized=False, piston_tip_tilt=False):
        Nnum = self.Nnum

        f = lambda x:[x] if type(x) ==int else x
        u, v = f(u), f(v)
        uv_fresnel = torch.roll(self.fresnel_2dkernel, shifts=(-int((u[0]-Nnum//2)*self.OSR), -int((v[0]-Nnum//2)*self.OSR)), dims=(-2,-1)).unsqueeze(0)
        for ii in range(1, len(u)):
            uv_fresnel = torch.cat((uv_fresnel, torch.roll(self.fresnel_2dkernel, shifts=(-int((u[ii]-Nnum//2)*self.OSR), -int((v[ii]-Nnum//2)*self.OSR)), dims=(-2,-1)).unsqueeze(0)))
        uv_mla_fresnel = uv_fresnel * self.MLARRAY.unsqueeze(0)

        uv_mla_fresnel = torch.fft.fft2( torch.fft.ifftshift( uv_mla_fresnel , dim=(-2,-1) ), dim=(-2,-1) )

        h_slice = slice((self.Ny - self.Ny//self.xy_downsample) //2, (self.Ny//self.xy_downsample + self.Ny)//2, None)
        w_slice = slice((self.Nx - self.Nx//self.xy_downsample) //2, (self.Nx//self.xy_downsample + self.Nx)//2, None)
        hw_index = torch.fft.ifftshift(torch.arange(self.Nx//self.xy_downsample, device=self.device) - self.Nx//self.xy_downsample // 2)
        uv_mla_fresnel = torch.fft.ifftshift(torch.fft.fftshift( uv_mla_fresnel , dim=(-2,-1) )[:, h_slice, w_slice], dim=(-2,-1) )

        if phi is None:
            ku = self.kbase[d, :, :].unsqueeze(0) * uv_mla_fresnel.unsqueeze(1)
        else:
            phi = self.masked_phase_array(phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt)[hw_index, :][:, hw_index]
            if self.zernike_coef_in_lambda:
                ku = self.kbase[d, :, :].unsqueeze(0) * torch.exp( -2.j * torch.pi * phi ).unsqueeze(0).unsqueeze(0) * uv_mla_fresnel.unsqueeze(1)
            else:
                ku = self.kbase[d, :, :].unsqueeze(0) * torch.exp(-2.j * torch.pi * phi / self.lam_detection).unsqueeze(0).unsqueeze(0) * uv_mla_fresnel.unsqueeze(1)

        res = self.myzifftn(ku) / self.energy_ratio_complex
        res = torch.fft.fftshift(res, dim=(-2,-1))
        return res[:, :, self.OSR//2::self.OSR, self.OSR//2::self.OSR] * self.OSR

    def coherent_psf(self, a, d, phi, normalized=False, piston_tip_tilt=False, psf_binning=3, out_size=(225, 225)):

        uv_mla_fresnel = self.uv_mla_fresnel[a,:,:]

        if phi is None:
            shift_int, uv_mla_fresnel_phi = self.decouple_shift_unwrapped_slicing(uv_mla_fresnel.unsqueeze(1)  , torch.fft.fftshift( -2*torch.pi * (self.phase_angular_spectrum.unsqueeze(0).unsqueeze(0) * d.reshape(1,-1,1,1) * self.dz_rms) ,dim=(-2,-1)) , psf_binning=psf_binning)

        else:
            phi = self.masked_phase_array(phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt)
            shift_int, uv_mla_fresnel_phi = self.decouple_shift_unwrapped_slicing(uv_mla_fresnel.unsqueeze(1)  , torch.fft.fftshift( -2*torch.pi * (self.phase_angular_spectrum.unsqueeze(0).unsqueeze(0) * d.reshape(1,-1,1,1) * self.dz_rms + phi.unsqueeze(0).unsqueeze(0) ) ,dim=(-2,-1)) , psf_binning=psf_binning)

        ku = torch.fft.fftshift( self.kbase.unsqueeze(0), dim=(-2,-1)) * uv_mla_fresnel_phi

        res = compute_centered_psf_mft_downsampled(ku, out_size=out_size, down_factor=psf_binning) * (psf_binning / self.energy_ratio_complex)

        return shift_int // psf_binning, res

    def coherent_psf_shifted(self, a, d, phi, normalized=False, piston_tip_tilt=False, psf_binning=3, shift_fixed=None, out_size=(225, 225)):

        uv_mla_fresnel = self.uv_mla_fresnel[a,:,:]

        if phi is None:
            uv_mla_fresnel_phi = self.apply_known_int_shift(uv_mla_fresnel.unsqueeze(1)  , torch.fft.fftshift( -2*torch.pi * (self.phase_angular_spectrum.unsqueeze(0).unsqueeze(0) * d.reshape(1,-1,1,1) * self.dz_rms) ,dim=(-2,-1)) , psf_binning=psf_binning, shift_int = shift_fixed)
        else:
            phi = self.masked_phase_array(phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt)
            uv_mla_fresnel_phi = self.apply_known_int_shift(uv_mla_fresnel.unsqueeze(1)  , torch.fft.fftshift( -2*torch.pi * (self.phase_angular_spectrum.unsqueeze(0).unsqueeze(0) * d.reshape(1,-1,1,1) * self.dz_rms + phi.unsqueeze(0).unsqueeze(0) ) ,dim=(-2,-1)) , psf_binning=psf_binning, shift_int = shift_fixed)

        ku = torch.fft.fftshift( self.kbase.unsqueeze(0), dim=(-2,-1)) * uv_mla_fresnel_phi
        res = compute_centered_psf_mft_downsampled(ku, out_size=out_size, down_factor=psf_binning) * (psf_binning / self.energy_ratio_complex)

        return res

    def calcRMS(self, phi, normalized=False, piston_tip_tilt=False):
        phase = self.masked_phase_array(phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt)
        phase = phase[self.kmask2]

        rms = ( ((phase - phase.mean())**2 ).sum() / phase.numel() )** 0.5
        return rms

    def incoherent_psf(self, a, d, phi, normalized=False, piston_tip_tilt=False, psf_binning=3, shift_fixed=None, out_size=(225, 225)):
        if shift_fixed is not None:
            _psf = self.coherent_psf_shifted(a, d, phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt, psf_binning=psf_binning, shift_fixed=shift_fixed, out_size=out_size)
            _psf = torch.abs(_psf) ** 2
            return shift_fixed, _psf
        else:
            shift_int, _psf = self.coherent_psf(a, d, phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt, psf_binning=psf_binning, out_size=out_size)
            _psf = torch.abs(_psf) ** 2

            return shift_int, _psf

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
    def fresnel2d(x, dx0, z, k, device='cpu'):

        u = torch.fft.fftfreq(x.shape[-1], d=dx0).to(device)
        v = torch.fft.fftfreq(x.shape[-2], d=dx0).to(device)

        V, U = torch.meshgrid(v, u, indexing='ij')

        H = x * torch.exp(1j * k * z) * torch.exp(-1j * 2 * torch.pi**2 * (U**2 + V**2).unsqueeze(0)  * z / k)

        return H

    def get_depth_shift(self, u, v, d, phi=None, normalized=True, piston_tip_tilt=False):
        psf_d = self.incoherent_psf(u,v, slice(d,d+1,None), phi=phi, normalized=normalized, piston_tip_tilt=piston_tip_tilt)
        shift = self.get_centerofmass(psf_d)
        return shift

    @staticmethod
    def get_centerofmass(psf):
        """ calculate psf center of mass.(psf>0 == True)
        input: [C,D,H,W]
        return:[C,D,2]
        """
        C, D, H, W = psf.shape
        device = psf.device

        psf = psf.reshape(C, D, H, W, 1)
        hwgrid = torch.stack(torch.meshgrid(torch.arange(H, device=device) - (H-1)/2,torch.arange(W, device=device) - (W-1)/2,indexing='ij'), dim=-1).unsqueeze(0)
        psf_centerofmass = torch.zeros(C, D, 2, device=device)
        for c in range(C):
            psf_centerofmass[c, :,:] = (  psf[c, :,:,:,:] * hwgrid / (psf[c, :,:,:,:].sum((-3,-2),keepdim =True) + 1e-9)  ).sum((-3,-2))

        return psf_centerofmass

    @staticmethod
    def decouple_shift_unwrapped_slicing(U, Phi, is_ifft=True, psf_binning=3):
        """
        Separate integer shifts from the local blur using adjacent phase differences.
        Amplitude-product weights suppress contributions across the pupil boundary.
        """
        A, _, H_dim, W_dim = U.shape
        _, C, _, _ = Phi.shape
        device = U.device

        U_x_fwd = U[..., :, 1:]
        U_x_bwd = U[..., :, :-1]
        Phi_x_fwd = Phi[..., :, 1:]
        Phi_x_bwd = Phi[..., :, :-1]

        dphi_U_x = torch.angle(U_x_fwd * U_x_bwd.conj())
        dphi_Phi_x = Phi_x_fwd - Phi_x_bwd

        I_x = U_x_fwd.abs() * U_x_bwd.abs()

        sum_X = (I_x * (dphi_U_x + dphi_Phi_x)).sum(dim=(-2, -1), keepdim=True)
        denom_X = I_x.sum(dim=(-2, -1), keepdim=True) + 1e-12

        U_y_fwd = U[..., 1:, :]
        U_y_bwd = U[..., :-1, :]
        Phi_y_fwd = Phi[..., 1:, :]
        Phi_y_bwd = Phi[..., :-1, :]

        dphi_U_y = torch.angle(U_y_fwd * U_y_bwd.conj())
        dphi_Phi_y = Phi_y_fwd - Phi_y_bwd

        I_y = U_y_fwd.abs() * U_y_bwd.abs()

        sum_Y = (I_y * (dphi_U_y + dphi_Phi_y)).sum(dim=(-2, -1), keepdim=True)
        denom_Y = I_y.sum(dim=(-2, -1), keepdim=True) + 1e-12

        fft_sign = -1.0 if is_ifft else 1.0

        shift_x_sub = fft_sign * (sum_X / denom_X) * W_dim / (2 * torch.pi)
        shift_y_sub = fft_sign * (sum_Y / denom_Y) * H_dim / (2 * torch.pi)

        shift_x_int = torch.round(shift_x_sub) // psf_binning * psf_binning
        shift_y_int = torch.round(shift_y_sub) // psf_binning * psf_binning

        int_shift = torch.cat(
            [shift_y_int.view(A, C, 1), shift_x_int.view(A, C, 1)],
            dim=-1
        ).to(torch.long)

        center_y = H_dim // 2
        center_x = W_dim // 2
        freq_y = (torch.arange(H_dim, dtype=torch.float32, device=device) - center_y) / H_dim
        freq_x = (torch.arange(W_dim, dtype=torch.float32, device=device) - center_x) / W_dim

        Y, X = torch.meshgrid(freq_y, freq_x, indexing='ij')
        Y = Y.view(1, 1, H_dim, W_dim)
        X = X.view(1, 1, H_dim, W_dim)

        phase_ramp = torch.exp(
            -1j * fft_sign * 2 * torch.pi * (X * shift_x_int + Y * shift_y_int)
        )

        blur_freq = U * torch.exp(1j * Phi) * phase_ramp

        return int_shift, blur_freq

    @staticmethod
    def apply_known_int_shift(
        U,
        Phi,
        shift_int,
        is_ifft=True,
        psf_binning=3,
    ):
        """
        Remove a known integer shift from the frequency-domain field.

        Args:
            U: Complex frequency-domain amplitude, [A, 1, H, W].
            Phi: Phase in radians, [1, C, H, W].
            shift_int: [A, C, 2] in (y, x) order, measured in binned PSF pixels.
            is_ifft: Use the inverse Fourier transform sign convention.
            psf_binning: PSF downsampling factor.

        Returns:
            blur_freq: [A, C, H, W]
        """
        _, _, H, W = U.shape
        device = U.device
        real_dtype = U.real.dtype

        # Convert binned PSF shifts to the full-resolution grid.
        shift = shift_int.detach().to(device=device, dtype=real_dtype)
        shift_y = shift[..., 0, None, None] * psf_binning
        shift_x = shift[..., 1, None, None] * psf_binning

        freq_y = (torch.arange(H, device=device, dtype=real_dtype) - H // 2) / H
        freq_x = (torch.arange(W, device=device, dtype=real_dtype) - W // 2) / W

        Y, X = torch.meshgrid(freq_y, freq_x, indexing="ij")
        Y = Y[None, None]
        X = X[None, None]

        fft_sign = -1.0 if is_ifft else 1.0
        phase_ramp = torch.exp(-1j * fft_sign * 2 * torch.pi* (X * shift_x + Y * shift_y))

        return U * torch.exp(1j * Phi) * phase_ramp

def compute_centered_psf_mft_downsampled(blur_freq, out_size=(3, 3), down_factor=5, is_ifft=True):
    """
    Evaluate a local matrix Fourier transform directly on the sampled output grid.

    Args:
        blur_freq: Frequency-domain field with its shift removed, [A, C, H, W].
        out_size: Output dimensions (out_H, out_W) after downsampling.
        down_factor: Spatial sampling stride.

    Returns:
        Complex local field with shape [A, C, out_H, out_W].
    """
    A, C, H, W = blur_freq.shape
    out_H, out_W = out_size
    device = blur_freq.device

    sign = 1.0 if is_ifft else -1.0

    # The input field is centered in frequency; output coordinates use the stride.
    u = torch.arange(H, dtype=torch.float32, device=device) - H // 2
    v = torch.arange(W, dtype=torch.float32, device=device) - W // 2

    y = (torch.arange(out_H, dtype=torch.float32, device=device) - out_H // 2) * down_factor
    x = (torch.arange(out_W, dtype=torch.float32, device=device) - out_W // 2) * down_factor

    phase_y = 2 * torch.pi * torch.outer(y, u) / H
    phase_x = 2 * torch.pi * torch.outer(x, v) / W

    K_y = torch.exp(1j * sign * phase_y)
    K_x = torch.exp(1j * sign * phase_x)

    temp = torch.matmul(K_y, blur_freq)

    psf_roi_complex = torch.matmul(temp, K_x.T)

    if is_ifft:
        psf_roi_complex = psf_roi_complex / (H * W)

    return psf_roi_complex


if __name__ == "__main__":
    import tifffile,os,time


    # oil parameters
    Nnum = 15
    lfpsf_shape = (Nnum, Nnum, 1, 375, 375)
    OSR=3
    M=20
    n=1.406
    na_detection=1.05
    MLPitch=56.4e-6
    fml=536.4e-6
    lam_detection=525*1e-9
    dz=0.6e-6
    device = 'cuda:0'
    xy_downsample = 3
    psf_binning = 1


    # water parameters
    Nnum = 13
    lfpsf_shape = (Nnum, Nnum, 1, 777, 777)
    OSR=3
    M=15.75
    n=1.333
    na_detection=1.0
    MLPitch=48.88e-6
    fml=383.7e-6
    lam_detection=525*1e-9
    dz=1e-6
    device = 'cuda:0'
    xy_downsample = 3
    psf_binning = 1


    # air parameters (active when running this file directly)
    Nnum = 15
    lfpsf_shape = (Nnum, Nnum, 1, 525, 525)
    OSR= 3
    M=7.85
    n=1
    na_detection=0.5
    MLPitch=56.4e-6
    fml=444.15e-6
    lam_detection=525*1e-9
    dz=5e-6
    device = 'cuda:0'
    xy_downsample = 3
    psf_binning = 1



    # single_point = torch.empty(lfpsf_shape, dtype=torch.float32, device='cpu')
    wf= [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -5.5864e-01,  0.0000e+00,
        -5.6725e-01,  1.2072e-01, -8.0501e-01, -1.9836e+00,  1.0349e-02,
         3.2685e-03, -4.8801e-04,  7.7510e-01,  1.0151e-03,  1.1323e-01,
        -4.6229e-02,  1.0003e-02, -1.3905e-01,  8.0669e-02, -3.1488e-02,
        -6.3695e-02]
    # input_views = [112, 113, 128, 127, 126, 111, 96, 97, 98, 99, 114, 129, 144, 143, 142, 141, 140, 125, 110, 95, 80, 81, 82, 83, 84, 85, 100, 115, 130, 145, 160, 159, 158, 157, 156, 155, 154, 139, 124, 109, 94, 79, 64, 65, 66, 67, 68, 69, 70, 71, 86, 101, 116, 131, 146, 161, 175, 174, 173, 172, 171, 170, 169, 153, 138, 123, 108, 93, 78, 63, 49, 50, 51, 52, 53, 54, 55, 117, 187, 107, 37]
    # wf = None
    input_views = list(range(Nnum*Nnum))
    # input_views = input_views[0:1]
    # input_views = [32]
    t0 = time.time()
    # lfpsf = PsfGenerator5D(lfpsf_shape=[13,13, 101,351, 351], n=1.515, na_detection=1.4)
    lfpsf = PsfGenerator5D(lfpsf_shape=lfpsf_shape, M=M, n=n, na_detection=na_detection, MLPitch=MLPitch, Nnum=Nnum, OSR=OSR, fml=fml, lam_detection=lam_detection, dz=dz, xy_downsample = xy_downsample, device=device, input_views=input_views)
    # single_point = lfpsf.incoherent_psf([0., 0., 0.], normalized=True, piston_tip_tilt=False)

    t1 = time.time()
    print("init time cost", t1 - t0)
    

    # kk = dz * lfpsf.rms_H / lam_detection # to make one pixel shift in depth 

    # u,v = np.unravel_index(np.arange(0,Nnum*Nnum), shape=(Nnum,Nnum))
    # blksize = 3
    # z_slice = slice(50,51,None)




    # input('debug')
    z_range = torch.arange(101,dtype=torch.float32,device=device) - 101//2
    # z_range = torch.arange(1,dtype=torch.float32,device=device) - 1//2
    # z_range = torch.arange(31,dtype=torch.float32,device=device) - 31//2
    aa = list(range(len(input_views)))
    # aa = input_views
    # aa = [0]
    blksize = 31
    wf = torch.zeros(21)
    kk = dz * lfpsf.rms_H / lam_detection # to make one pixel shift in depth 

    # wf[0] =  -24*kk
    # shift_int, tmp = lfpsf.incoherent_psf(aa, z_range[0:blksize], wf, normalized=True, piston_tip_tilt=True, psf_binning=psf_binning)
    # single_point = tmp.cpu()
    # for blk in range(1, (len(z_range)-1) // blksize + 1):
    #     shift_int_tmp, tmp = lfpsf.incoherent_psf(aa, z_range[blksize*blk:blksize*(blk+1)], wf, normalized=True, piston_tip_tilt=True, psf_binning=psf_binning)
    #     single_point = torch.cat((single_point, tmp.cpu()), dim=-3)
    #     shift_int = torch.cat((shift_int, shift_int_tmp), dim=-2)

    blksize = 5  # Keep all 225 views within 24 GB of GPU memory.
    shift_int, single_point = lfpsf.incoherent_psf(aa, z_range[0:blksize], wf, normalized=True, piston_tip_tilt=True, psf_binning=psf_binning)
    # single_point = single_point.cpu()
    for blk in range(1, (len(z_range)-1) // blksize + 1):
        shift_int_tmp, tmp = lfpsf.incoherent_psf(aa, z_range[blksize*blk:blksize*(blk+1)], wf, normalized=True, piston_tip_tilt=True, psf_binning=psf_binning)
        # single_point = torch.cat((single_point, tmp ), dim=-3)
        # shift_int = torch.cat((shift_int, shift_int_tmp), dim=-2)

    # single_point = single_point.reshape(lfpsf_shape[0:2]+ (single_point.shape[-3], lfpsf_shape[3]//xy_downsample, lfpsf_shape[4]//xy_downsample,))

    # wf = None
    # z_slice = slice(50,55,None)
    # u,v = np.unravel_index(np.arange(0,Nnum*Nnum), shape=(Nnum,Nnum))
    # blksize = 3
    # single_point = lfpsf.incoherent_psf(u[0:blksize],v[0:blksize], z_slice, wf, normalized=True, piston_tip_tilt=True).cpu()
    # for blk in range(1, (len(u)-1) // blksize + 1):
    #     single_point = torch.cat((single_point, lfpsf.incoherent_psf(u[blk*blksize:(blk+1)*blksize],v[blk*blksize:(blk+1)*blksize], z_slice, wf, normalized=True, piston_tip_tilt=True).cpu()), dim=0)

    # single_point = single_point.reshape(lfpsf_shape[0:2]+ (single_point.shape[-3], lfpsf_shape[3]//xy_downsample, lfpsf_shape[4]//xy_downsample,))


    # for u in range(Nnum):
    #     for v in range(Nnum):
    #         single_point[u, v] = lfpsf.incoherent_psf(u,v, None, normalized=True, piston_tip_tilt=False)
    #         # os.makedirs("./rms0_Cr0", exist_ok=True)
    #         # tifffile.imwrite("./rms0_Cr0/u%d_v%d.tif"%(u,v), single_point[u, v].cpu().numpy(), imagej=True, metadata={'spacing': (0.2, 0.11, 0.11), 'unit': 'micrometer', 'axes': 'ZYX'})
    #         pass
    torch.cuda.synchronize()
    t2 = time.time()
    print("psfcalc time cost", t2 - t1)
    print("total time cost", t2 - t0)

    # import lf_forward
    # # volume = torch.zeros(1, 101, 675, 675)
    # psf = single_point[:,:, :,:]
    # volume = torch.from_numpy( volume.astype(np.float32) ).unsqueeze(0)
    # res = lf_forward.apply_decoupled_psf(volume.cuda(), psf.cuda(), shift_int[:,:, :], method='fft')

    # t3 = time.time()
    # print("imaging time cost", t3 - t2)
    # print("total time cost", t3 - t0)
    # tifffile.imwrite('lf.tiff', res.cpu().numpy())

    if False:
        # os.makedirs("./lfpsf", exist_ok=True)
        # tifffile.imwrite("./lfpsf/lfpsf_z%dn%fNA%f_M%d_OSR%d_fml%.1fum_pitch%.1fum_dz%.1fum.tif"%(lfpsf_shape[2], int(n*1), int(na_detection*1), M, OSR, fml*1e6, MLPitch*1e6, dz*1e6), single_point.reshape(-1, *lfpsf_shape[2:]).cpu().numpy(), imagej=True, metadata={'spacing': (dz*1e6, MLPitch/Nnum/M*1e6, MLPitch/Nnum/M*1e6), 'unit': 'micrometer', 'axes': 'TZYX'}, bigtiff=True, compression=None)
        aa = 0 
        os.makedirs("../results/lfpsfv4_new/lfpsf_xy%d_z%dn%.3fNA%.2f_M%d_OSR%d_fml%.1fum_pitch%.1fum_dz%.1fum_xydownsample%d_psfbinning%d/"%(lfpsf_shape[-1], lfpsf_shape[-3], n, na_detection, M, OSR, fml*1e6, MLPitch*1e6, dz*1e6, xy_downsample, psf_binning), exist_ok=True)
        # for uu in range(single_point.shape[0]):
        #     for vv in range(single_point.shape[1]):
        # for uu in [2,7]:
        #     for vv in [2,7]:
        for aa in range(len(input_views)):
            tifffile.imwrite("../results/lfpsfv4_new/lfpsf_xy%d_z%dn%.3fNA%.2f_M%d_OSR%d_fml%.1fum_pitch%.1fum_dz%.1fum_xydownsample%d_psfbinning%d/aa%d.tif"%(lfpsf_shape[-1], lfpsf_shape[-3], n, na_detection, M, OSR, fml*1e6, MLPitch*1e6, dz*1e6, xy_downsample, psf_binning, input_views[aa]), single_point[aa,:,:,:].cpu().numpy(), imagej=True, metadata={'spacing': (MLPitch/Nnum/M*1e6, MLPitch/Nnum/M*1e6), 'unit': 'micrometer', 'axes': 'ZYX'}, compression=None)
            aa += 1
