"""Shared sampled wave-optics operator for the staged wDAO comparison.

The same finite 225-pixel kernels and fixed ideal shifts as Figure 5 are used.
This exposes the linear operator and its adjoint without differentiating the
wavefront. Caching the current OTF changes cost, not the forward calculation.
"""
from __future__ import annotations

import torch


class CachedOptics:
    def __init__(self, generator, projection):
        self.g = generator
        self.project = projection
        self.views = len(projection.aperture)
        self.height, self.width = projection.shape
        self.depths = len(projection.propagation)
        self.stride = projection.stride
        self.otfs = torch.empty((self.views, self.depths, self.height, self.width//2+1),
                               dtype=torch.complex64, device=generator.device)
        self.current_wavefront = None

    @torch.no_grad()
    def set_wavefront(self, wavefront):
        p = self.project
        phase = torch.einsum('m,mhw->hw', wavefront, p.basis[:len(wavefront)])
        pupil = p.propagation * torch.exp(-2j*torch.pi*phase)
        for v in range(self.views):
            field = p.aperture[v, None] * pupil
            field = field * p.ramp_y[v, :, :, None] * p.ramp_x[v, :, None, :]
            amplitude = torch.fft.ifft2(field)
            psf = (amplitude[..., p.sample_y[:, None], p.sample_x]/p.energy).abs().square()
            otf = torch.fft.rfft2(psf, s=(self.height, self.width))
            otf *= p.otf_y[v, :, :, None] * p.otf_x[v, :, None, :]
            self.otfs[v].copy_(otf)
        self.current_wavefront = wavefront.detach().clone()

    def forward_spectrum(self, spectrum, view):
        out = torch.fft.irfft2((spectrum*self.otfs[view]).sum(0),
                              s=(self.height, self.width))
        return out[::self.stride, ::self.stride]

    def forward(self, volume, view):
        return self.forward_spectrum(torch.fft.rfft2(volume), view)

    def forward_all(self, volume):
        spectrum = torch.fft.rfft2(volume)
        return torch.stack([self.forward_spectrum(spectrum, v) for v in range(self.views)])

    def adjoint(self, image, view):
        full = image.new_zeros((self.height, self.width))
        full[::self.stride, ::self.stride] = image
        return torch.fft.irfft2(torch.fft.rfft2(full)[None]*self.otfs[view].conj(),
                               s=(self.height, self.width))


def keys_kernel(distance):
    """Keys cubic convolution with MATLAB's a=-1/2."""
    x = distance.abs()
    return torch.where(x <= 1, 1.5*x**3-2.5*x**2+1,
                       torch.where(x < 2, -.5*x**3+2.5*x**2-4*x+2, torch.zeros_like(x)))


def pull_shift(image, shift_yx):
    """Evaluate image(y+dy,x+dx), Keys cubic interpolation, zero exterior.

    Positive observed-minus-predicted shifts move the measurement back into the
    prediction's coordinates, matching RUSH3D's interp2 pull convention.
    """
    out = image
    for axis, shift in ((-2, shift_yx[0]), (-1, shift_yx[1])):
        size = image.shape[axis]
        coordinates = torch.arange(size, device=image.device, dtype=image.dtype) + shift
        start = coordinates.floor().long()
        result = torch.zeros_like(out)
        for offset in (-1, 0, 1, 2):
            indices = start + offset
            weight = keys_kernel(coordinates-indices) * ((indices >= 0) & (indices < size))
            values = out.index_select(axis, indices.clamp(0, size-1))
            if axis == -2:
                weight = weight[:, None]
            result += values * weight
        out = result
    return out


@torch.no_grad()
def isra_epoch(operator, volume, data, shifts, *, weight):
    """One ordered angular pass of RUSH3D's relaxed A^T y/A^T Ax update.

    RUSH3D's public code names its reconstruction as deconvolution; its actual
    update is an ISRA-type least-squares ratio, not Poisson Richardson–Lucy.
    The observation is sampled on the archived 135-pixel LF grid here. Its
    adjoint therefore inserts zeros before backprojection, rather than treating
    interpolated detector samples as independent measured data.
    """
    for view in range(operator.views):
        prediction = operator.forward(volume, view)
        aligned = pull_shift(data[view], shifts[view]).clamp_min(0)
        numerator = operator.adjoint(aligned, view).clamp_min(0)
        denominator = operator.adjoint(prediction, view).clamp_min(0)
        floor = torch.finfo(volume.dtype).eps * denominator.amax().clamp_min(1e-20)
        ratio = numerator / denominator.clamp_min(floor)
        relaxation = .4 * weight[view] / weight.max()
        volume.mul_((1-relaxation) + relaxation*ratio)
    return volume
