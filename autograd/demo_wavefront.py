"""Joint bead reconstruction and wavefront estimation for paper Fig. 3(a-e).

Run: python autograd/demo_wavefront.py
Edit the settings below to use your own data. Raw light-field observations must
have shape [view, y, x], with views ordered as in PHYSICAL_VIEWS.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
from scipy.ndimage import gaussian_filter1d
import tifffile
import torch
import torch.nn.functional as F
from lfpsf_torch_shift_blur_batched import PsfGenerator5D


# 1. Settings. Optical lengths are in meters; wavefront coefficients are in waves.
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'data/demo'
DATA = EXAMPLE / 'lf.tiff'                  # Keep the sensor bias; do not clip.
TRUE_WF = EXAMPLE / 'wf_true.txt'           # Optional evaluation reference.
CLEAN_LF = EXAMPLE / 'lf_clean.tiff'        # Optional evaluation reference.
OUTPUT = ROOT / 'outputs/demo_wavefront_sample02'
DEVICE = 'cuda:0'                          # Use 'cpu' for small debugging cases.
VOLUME_SHAPE = (101, 675, 675)              # [z, y, x], with z centered at zero.
PSF_SIZE, SAMPLE_STRIDE, MODES = 225, 5, 21
BIAS, READ_VARIANCE = 105., 9.             # Known sensor bias and read variance.
INTENSITY_SCALE = 0.5829866912756507         # Output unit conversion only.
# Set INTENSITY_SCALE=1 for photon units when the exposure scale is unknown.
STEPS, VIEWS_PER_STEP, DEPTH_CHUNK = 2400, 3, 8
RELAXATION, LR_WAVEFRONT, REGULARIZATION = .2, .003, .1
BLUR_SIGMAS, BLUR_STAGE_STEPS = (4, 2, 1, 0), 400
INITIAL_UNIFORM_FRACTION, WAVEFRONT_LIMIT, SEED = .5, 4., 0
FIXED_MODES = [0, 1, 2, 4]                 # Axial propagation, tip, tilt, defocus.
OPTICS = dict(lfpsf_shape=(15, 15, 1, 525, 525), M=7.85, n=1,
    na_detection=.5, MLPitch=56.4e-6, Nnum=15, OSR=3, fml=444.15e-6,
    lam_detection=525e-9, dz=5e-6, xy_downsample=3)
PHYSICAL_VIEWS = [
    112, 113, 128, 127, 126, 111, 96, 97, 98, 99, 114, 129, 144,
    143, 142, 141, 140, 125, 110, 95, 80, 81, 82, 83, 84, 85, 100,
    115, 130, 145, 160, 159, 158, 157, 156, 155, 154, 139, 124, 109,
    94, 79, 64, 65, 66, 67, 68, 69, 70, 71, 86, 101, 116, 131, 146,
    161, 175, 174, 173, 172, 171, 170, 169, 153, 138, 123, 108, 93,
    78, 63, 49, 50, 51, 52, 53, 54, 55, 117, 187, 107, 37,
]


def relative_path(path):
    """Store paths relative to the repository root, including custom inputs."""
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class Projection(torch.nn.Module):
    """Project a volume using fixed zero-aberration shifts and updated local PSFs."""
    def __init__(self, optics):
        super().__init__()
        g, device = optics, optics.device
        nz, height, width = VOLUME_SHAPE
        if not 0 < PSF_SIZE <= min(height, width, *g.krho.shape):
            raise ValueError('PSF_SIZE must fit both the volume and optical grids.')
        self.shape, self.stride = (height, width), SAMPLE_STRIDE
        depths = torch.arange(nz, dtype=torch.float32, device=device) - nz//2
        # Precompute fixed optical terms without using ground truth.
        with torch.no_grad():
            shifts = []
            for view in range(len(PHYSICAL_VIEWS)):
                blocks = [g.incoherent_psf([view], z, torch.zeros(MODES, device=device),
                    normalized=True, piston_tip_tilt=True, psf_binning=1)[0]
                    for z in depths.split(21)]
                shifts.append(torch.cat(blocks, dim=1))
            shifts = torch.cat(shifts)
            self.shifts = shifts
            self.basis = torch.stack([torch.fft.fftshift(
                g.zernike_polynomial(j, normalized=True)*g.kmask2) for j in range(MODES)])
            self.propagation = torch.exp(-2j*torch.pi*
                torch.fft.fftshift(g.phase_angular_spectrum)*depths[:, None, None]*g.dz_rms)
            self.aperture = torch.fft.fftshift(g.kbase)*g.uv_mla_fresnel
            h, w = g.krho.shape
            fy = (torch.arange(h, device=device)-h//2)/h
            fx = (torch.arange(w, device=device)-w//2)/w
            self.ramp_y = torch.exp(2j*torch.pi*shifts[..., 0, None]*fy)
            self.ramp_x = torch.exp(2j*torch.pi*shifts[..., 1, None]*fx)
            self.sample_y = (torch.arange(PSF_SIZE, device=device)-PSF_SIZE//2)%h
            self.sample_x = (torch.arange(PSF_SIZE, device=device)-PSF_SIZE//2)%w
            sy = shifts[..., 0]+(height-PSF_SIZE)//2-height//2
            sx = shifts[..., 1]+(width-PSF_SIZE)//2-width//2
            iy = torch.arange(height, device=device)
            ix = torch.arange(width//2+1, device=device)
            self.otf_y = torch.exp((-2j*torch.pi/height)*
                (sy[..., None]*iy).remainder(height).double()).to(g.kbase.dtype)
            self.otf_x = torch.exp((-2j*torch.pi/width)*
                (sx[..., None]*ix).remainder(width).double()).to(g.kbase.dtype)
            self.energy = g.energy_ratio_complex

    def pupil_field(self, wf):
        return torch.exp(-2j*torch.pi*torch.einsum('m,mhw->hw', wf, self.basis))

    def psf(self, pupil, view, first, last):
        field = self.aperture[view, None]*self.propagation[first:last]*pupil[None]
        field = field*self.ramp_y[view, first:last, :, None]*self.ramp_x[view, first:last, None, :]
        amplitude = torch.fft.ifft2(field)
        return (amplitude[..., self.sample_y[:, None], self.sample_x]/self.energy).abs().square()

    def otf(self, pupil, view, first, last):
        psf = self.psf(pupil, view, first, last)
        return torch.fft.rfft2(psf, s=self.shape)*self.otf_y[view, first:last, :, None]*self.otf_x[view, first:last, None, :]

    def finish(self, spectrum):
        # Autograd uses the exact transpose of this sparse camera sampling.
        return torch.fft.irfft2(spectrum, s=self.shape)[::self.stride, ::self.stride]

    @torch.no_grad()
    def forward(self, volume, wf):
        """Evaluate all views; use ProjectionCache for optimization gradients."""
        cache = ProjectionCache(self, volume, wf)
        try:
            return torch.stack([cache(view) for view in range(len(PHYSICAL_VIEWS))])
        finally:
            cache.close()


class ProjectionCache:
    """Reuse volume FFTs across views and reconnect their accumulated gradients."""
    def __init__(self, project, volume, wf):
        self.project = project
        self.field = project.pupil_field(wf)
        self.field_leaf = self.field.detach().requires_grad_(self.field.requires_grad)
        self.records = []
        first = 0
        for part in volume.split(DEPTH_CHUNK):
            spectrum = torch.fft.rfft2(part, s=project.shape)
            leaf = spectrum.detach().requires_grad_(spectrum.requires_grad)
            last = first+len(part)
            self.records.append((first, last, spectrum, leaf))
            first = last

    def __call__(self, view):
        total = None
        for first, last, _, leaf in self.records:
            part = (leaf*self.project.otf(self.field_leaf, view, first, last)).sum(0)
            total = part if total is None else total+part
        return self.project.finish(total)

    def backward(self):
        outputs, gradients = [], []
        for _, _, spectrum, leaf in self.records:
            if spectrum.requires_grad and leaf.grad is not None:
                outputs.append(spectrum)
                gradients.append(leaf.grad)
        if self.field.requires_grad and self.field_leaf.grad is not None:
            outputs.append(self.field)
            gradients.append(self.field_leaf.grad)
        torch.autograd.backward(outputs, gradients)

    def close(self):
        self.records.clear()
        self.field = self.field_leaf = None


# 2. Paper initialization, noise objective, and spatial prior.
def blur(data, sigma):
    """Apply the same periodic Gaussian filter to observations and predictions."""
    if not sigma:
        return data
    radius = 3*sigma
    axis = torch.arange(-radius, radius+1, device=data.device, dtype=data.dtype)
    kernel = torch.exp(-axis.square()/(2*sigma*sigma))
    kernel /= kernel.sum()
    value = F.conv2d(F.pad(data[:, None], (radius, radius, 0, 0), mode='circular'),
        kernel[None, None, None, :])
    return F.conv2d(F.pad(value, (0, 0, radius, radius), mode='circular'),
        kernel[None, None, :, None])[:, 0]


@torch.no_grad()
def initialize_volume(project, data, wf):
    """Refocus inner views at zero aberration, then mix in a uniform component."""
    nz, height, width = VOLUME_SHAPE
    n = data.shape[-1]
    scan = OPTICS['Nnum']
    chosen = [i for i, v in enumerate(PHYSICAL_VIEWS)
        if (v//scan-scan//2)**2+(v%scan-scan//2)**2 <= 9]
    if len(chosen) < 2:
        chosen = list(range(len(PHYSICAL_VIEWS)))
    descriptor = blur(data, 1)-blur(data, 6)
    descriptor /= descriptor.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1)
    descriptor.clamp_(-4, 8)
    yy, xx = torch.meshgrid(torch.arange(n, device=data.device),
        torch.arange(n, device=data.device), indexing='ij')
    sums = data.new_zeros((nz, n, n))
    squares = torch.zeros_like(sums)
    pupil = project.pupil_field(wf)
    axis = torch.arange(PSF_SIZE, device=data.device)-PSF_SIZE//2
    for view in chosen:
        centers = []
        for first in range(0, nz, DEPTH_CHUNK):
            last = min(nz, first+DEPTH_CHUNK)
            psf = project.psf(pupil, view, first, last)
            mass = psf.sum((-2, -1)).clamp_min(1e-20)
            cy = (psf.sum(-1)*axis).sum(-1)/mass+project.shifts[view, first:last, 0]
            cx = (psf.sum(-2)*axis).sum(-1)/mass+project.shifts[view, first:last, 1]
            centers.append(torch.stack([cy, cx], -1)/SAMPLE_STRIDE)
        centers = torch.cat(centers)
        qy = (yy[None]+centers[:, 0, None, None]).remainder(n)+1
        qx = (xx[None]+centers[:, 1, None, None]).remainder(n)+1
        source = F.pad(descriptor[view][None, None].expand(nz, -1, -1, -1),
            (1, 1, 1, 1), mode='circular')
        grid = torch.stack([(2*qx+1)/(n+2)-1, (2*qy+1)/(n+2)-1], -1)
        aligned = F.grid_sample(source, grid, mode='bilinear',
            padding_mode='border', align_corners=False)[:, 0]
        sums += aligned
        squares += aligned.square()
    coherence = ((sums.square()-squares)/((len(chosen)-1)*squares).clamp_min(1e-8)).clamp(0, 1) if len(chosen) > 1 else torch.ones_like(sums)
    score = (sums/len(chosen)).clamp_min(0).square()*coherence
    score = F.avg_pool2d(score[:, None], 5, stride=1, padding=2)[:, 0]
    score = torch.as_tensor(gaussian_filter1d(score.cpu().numpy(), 3, axis=0,
        mode='nearest'), device=data.device)
    score /= score.mean().clamp_min(1e-8)
    seed = .3+.7*score.clamp(max=20)
    y = torch.arange(height, device=data.device, dtype=data.dtype)*(1/SAMPLE_STRIDE)
    x = torch.arange(width, device=data.device, dtype=data.dtype)*(1/SAMPLE_STRIDE)
    gy, gx = torch.meshgrid((2*y+1)/n-1, (2*x+1)/n-1, indexing='ij')
    grid = torch.stack([gx, gy], -1)[None]
    parts = []
    for part in seed.split(DEPTH_CHUNK):
        parts.append(F.grid_sample(part[:, None], grid.expand(len(part), -1, -1, -1),
            mode='bilinear', padding_mode='border', align_corners=False)[:, 0])
    mean = sum(part.double().sum() for part in parts)/(nz*height*width)
    log_seed = [((1-INITIAL_UNIFORM_FRACTION)*part/mean.float()+INITIAL_UNIFORM_FRACTION)
        .clamp_min(1e-30).log() for part in parts]
    return torch.cat([part.exp() for part in log_seed])


def data_loss(rate, observed):
    """Mean Poisson-Gaussian quasi-deviance; observed is signed and bias-subtracted."""
    y, rate = observed.double(), rate.double().clamp_min(0)
    ref = y.clamp_min(0)
    return 2*((rate-ref)-(y+READ_VARIANCE)*
        torch.log1p((rate-ref)/(ref+READ_VARIANCE))).mean()


def spatial_penalty(volume):
    terms = []
    for axis in (-2, -1):
        curvature = torch.diff(volume, n=2, dim=axis)
        terms.append(F.smooth_l1_loss(curvature, torch.zeros_like(curvature)))
    return sum(terms)/2


def view_schedule(nviews):
    """Reproduce the paper's seeded CPU permutations without a schedule file."""
    random = torch.Generator().manual_seed(SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(part.tolist() for part in
            torch.randperm(nviews, generator=random).split(VIEWS_PER_STEP))
    return schedule[:STEPS]


# 3. Nonnegative sensitivity-scaled object updates and Adam wavefront updates.
def reconstruct(project, observed):
    counts = observed.double().sum()
    if counts <= 0:
        raise ValueError('Total counts after bias subtraction must be positive; check BIAS and input units.')
    wf = torch.nn.Parameter(observed.new_zeros(MODES))
    mask = torch.ones_like(wf)
    mask[FIXED_MODES] = 0
    volume = torch.nn.Parameter(initialize_volume(project, observed, wf))
    with torch.no_grad():
        scale = (counts/project(volume, wf*mask).double().sum()).float()
        volume.mul_(scale)
    object_optimizer = torch.optim.SGD([volume], lr=RELAXATION, momentum=0)
    optics_optimizer = torch.optim.Adam([wf], lr=LR_WAVEFRONT,
        betas=(.9, .99), eps=1e-8, fused=observed.device.type == 'cuda')
    targets = {sigma: blur(observed, sigma) for sigma in BLUR_SIGMAS}
    history = []
    for step, selected in enumerate(view_schedule(len(observed))):
        sigma = BLUR_SIGMAS[min(len(BLUR_SIGMAS)-1, step//BLUR_STAGE_STEPS)]
        volume.grad = wf.grad = None
        cache = ProjectionCache(project, volume, wf*mask)
        positive, fit = None, 0.
        try:
            leaves = [record[3] for record in cache.records]
            for view in selected:
                rate = blur(cache(view)[None], sigma)[0]
                weight = rate.detach().clamp_min(0)/(rate.detach().clamp_min(0)+READ_VARIANCE)
                linear = 2*(rate*weight).mean()/len(selected)
                adjoints = torch.autograd.grad(linear, leaves, retain_graph=True)
                if positive is None:
                    positive = [g.detach().clone() for g in adjoints]
                else:
                    for total, gradient in zip(positive, adjoints):
                        total.add_(gradient)
                loss = data_loss(rate, targets[sigma][view])/len(selected)
                if not torch.isfinite(loss):
                    raise FloatingPointError('The data objective contains a nonfinite value.')
                loss.backward()
                fit += float(loss.detach())
                del rate, loss, linear, adjoints
            sensitivity, = torch.autograd.grad([record[2] for record in cache.records],
                volume, grad_outputs=positive, retain_graph=True)
            cache.backward()
        finally:
            cache.close()
        prior = REGULARIZATION*spatial_penalty(volume)
        prior_gradient, = torch.autograd.grad(prior, volume)
        volume.grad.add_(prior_gradient)
        del prior_gradient
        with torch.no_grad():
            if float(sensitivity.min()) < -max(1e-12, float(sensitivity.max())*1e-5):
                raise FloatingPointError('The object sensitivity must be nonnegative.')
            volume.grad = volume.detach().clamp_min(1e-8)*volume.grad/sensitivity.clamp_min(1e-12)
            if not torch.isfinite(volume.grad).all():
                raise FloatingPointError('The object update contains nonfinite values.')
        object_optimizer.step()
        with torch.no_grad():
            volume.clamp_(min=0)
            wf.mul_(mask).clamp_(-WAVEFRONT_LIMIT, WAVEFRONT_LIMIT)
        optics_optimizer.step()
        with torch.no_grad():
            wf.mul_(mask).clamp_(-WAVEFRONT_LIMIT, WAVEFRONT_LIMIT)
        volume.grad = wf.grad = None
        del sensitivity, positive
        if (step+1) % 100 == 0 or step+1 == STEPS:
            history.append(dict(step=step+1, sigma=sigma, data_loss=fit, prior=float(prior.detach())))
            print(f'{step+1:4d}/{STEPS}  sigma={sigma}  data={fit:.6g}', flush=True)
        del prior
    return volume.detach(), (wf*mask).detach(), history


# 4. Save estimates. Optional references are read only after optimization.
@torch.no_grad()
def save_results(g, project, estimated_photons, wf):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    volume = estimated_photons/INTENSITY_SCALE
    predicted_photons = project(estimated_photons, wf)
    reprojected = predicted_photons/INTENSITY_SCALE
    for name, value in [('volume_estimated', volume), ('reprojected_lf', reprojected),
            ('reprojected_lf_sensor', predicted_photons+BIAS)]:
        tifffile.imwrite(OUTPUT/f'{name}.tiff', value.cpu().numpy(), compression='zlib')
    np.savetxt(OUTPUT/'wf_estimated.txt', wf.cpu().numpy())
    pupil = torch.fft.fftshift(g.kmask2).cpu().numpy()

    def phase(coefficients):
        return torch.fft.fftshift(g.masked_phase_array(coefficients,
            normalized=True, piston_tip_tilt=True)).cpu().numpy()

    recovered = phase(wf)
    maps, summary = {'estimated': recovered}, {}
    if TRUE_WF is not None:
        truth = torch.as_tensor(np.loadtxt(TRUE_WF), dtype=wf.dtype, device=wf.device)
        if truth.shape != wf.shape:
            raise ValueError('TRUE_WF must contain MODES coefficients, or be set to None.')
        maps = {'true': phase(truth), 'estimated': recovered, 'error': phase(wf-truth)}
        summary['phase_rms_error'] = g.calcRMS(wf-truth, normalized=True, piston_tip_tilt=True).item()
    if CLEAN_LF is not None:
        clean = torch.as_tensor(tifffile.imread(CLEAN_LF), dtype=torch.float64, device=wf.device)
        if clean.shape != reprojected.shape:
            raise ValueError('CLEAN_LF must match the reprojected shape, or be set to None.')
        summary['lf_relative_l2'] = ((reprojected.double()-clean).norm()/clean.norm()).item()
    for name, waves in maps.items():
        tifffile.imwrite(OUTPUT/f'phase_{name}_waves.tiff', waves)
    # The model uses the phase convention exp(-2*pi*i*wavefront).
    tifffile.imwrite(OUTPUT/'phase_estimated_rad.tiff', -2*np.pi*recovered)
    tifffile.imwrite(OUTPUT/'pupil_mask.tiff', pupil.astype(np.uint8))

    return summary


def main():
    device = torch.device(DEVICE)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use a CUDA-enabled installation, or set DEVICE to cpu for small debugging cases.')
        torch.cuda.set_device(device)
    if min(STEPS, VIEWS_PER_STEP, DEPTH_CHUNK, BLUR_STAGE_STEPS) < 1 or min(INTENSITY_SCALE, READ_VARIANCE, RELAXATION, LR_WAVEFRONT) <= 0:
        raise ValueError('Iteration counts, batch sizes, intensity scale, read variance, and step sizes must be positive.')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    raw = torch.as_tensor(tifffile.imread(DATA), dtype=torch.float32, device=device)
    expected = (len(PHYSICAL_VIEWS), *(len(range(0, n, SAMPLE_STRIDE)) for n in VOLUME_SHAPE[-2:]))
    if tuple(raw.shape) != expected or not torch.isfinite(raw).all():
        raise ValueError(f'LF must have shape {expected} and finite values; got {tuple(raw.shape)}. Check the grid and views.')
    synchronize(device)
    started = time.perf_counter()
    g = PsfGenerator5D(**OPTICS, device=device, input_views=PHYSICAL_VIEWS)
    project = Projection(g)
    synchronize(device)
    setup_s = time.perf_counter()-started
    started = time.perf_counter()
    volume, wf, history = reconstruct(project, raw-BIAS)
    synchronize(device)
    solve_s = time.perf_counter()-started
    if not torch.isfinite(volume).all() or not torch.isfinite(wf).all():
        raise FloatingPointError('Reconstruction contains nonfinite values; check the data, learning rates, and model.')
    summary = save_results(g, project, volume, wf)
    summary.update(steps=STEPS, setup_s=setup_s, solve_s=solve_s,
        input=relative_path(DATA), intensity_scale=INTENSITY_SCALE, device=device.type,
        volume_shape=VOLUME_SHAPE, psf_size=PSF_SIZE, sample_stride=SAMPLE_STRIDE,
        physical_views=PHYSICAL_VIEWS, optics=OPTICS, modes=MODES, fixed_modes=FIXED_MODES,
        bias=BIAS, read_variance=READ_VARIANCE, seed=SEED, views_per_step=VIEWS_PER_STEP,
        depth_chunk=DEPTH_CHUNK, object_update='nonnegative_sensitivity_scaled',
        relaxation=RELAXATION, wavefront_learning_rate=LR_WAVEFRONT,
        wavefront_limit=WAVEFRONT_LIMIT, regularization=REGULARIZATION,
        blur_sigmas=BLUR_SIGMAS, blur_stage_steps=BLUR_STAGE_STEPS,
        initial_uniform_fraction=INITIAL_UNIFORM_FRACTION, history=history,
        paper_case='points_p0100_s01', paper_revision='figure4_ten_pupils_20261106')
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for key in ['phase_rms_error', 'lf_relative_l2']:
        if key in summary:
            print(f'{key}: {summary[key]:.8g}')
    print(f'Results: {relative_path(OUTPUT)}')


if __name__ == '__main__':
    main()
