"""主图 3(b) 的单例 demo：python autograd/demo_wavefront.py。

读取最新论文 Sample 2（零基 index 1）的 100 光子观测，同时恢复三维物体和波前。
所有实验步骤都在本文件，仅导入基础光学类；从顶部配置开始适配自己的数据。
raw LF 需为 [view, y, x]，视角顺序与 PHYSICAL_VIEWS 一致。
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
# from lfpsf_torch_batch5 import PsfGenerator5D
from lfpsf_torch_shift_blur_batched import PsfGenerator5D


# 1. 修改任务时先看这里。光学长度单位为 m，波前系数单位为 wave。
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'data/reference/points_photons100_sample02'
DATA = EXAMPLE / 'lf.tiff'                  # 保留传感器偏置，不提前截断负信号
TRUE_WF = EXAMPLE / 'wf_true.txt'           # 自己的数据没有真值时设为 None
CLEAN_LF = EXAMPLE / 'lf_clean.tiff'        # 仅用于评估；可设为 None
OUTPUT = ROOT / 'outputs/demo_wavefront_sample02'
DEVICE = 'cuda:0'                          # 可改为 'cpu' 做小规模调试
VOLUME_SHAPE = (101, 675, 675)              # [z, y, x]；z 坐标默认以中间层为零
PSF_SIZE, SAMPLE_STRIDE, MODES = 225, 5, 21
BIAS, READ_VARIANCE = 105., 9.             # 已知传感器偏置和读出噪声方差
INTENSITY_SCALE = 0.5829866912756507         # 本例曝光系数；只用于输出单位换算
# 自己的数据可设 INTENSITY_SCALE=1，保留光子域输出；不要用未知干净图估计它。
STEPS, WARMUP, VIEWS_PER_STEP = 2300, 100, 3
LR_VOLUME, LR_WAVEFRONT, LR_BACKGROUND = .3, .1, .01
REGULARIZATION, PRIOR_DEPTHS = 3e-5, 16
FIXED_MODES = [0, 1, 2, 4]                 # 固定为零的 OSA/ANSI 模式
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
    """物体和波前 → LF；固定零像差位移，随波前更新局域 PSF。"""
    def __init__(self, optics):
        super().__init__()
        g, device = optics, optics.device
        nz, height, width = VOLUME_SHAPE
        if not 0 < PSF_SIZE <= min(height, width, *g.krho.shape):
            raise ValueError('PSF_SIZE 必须同时适合横向体网格和光学计算网格。')
        self.shape, self.stride = (height, width), SAMPLE_STRIDE
        depths = torch.arange(nz, dtype=torch.float32, device=device) - nz//2
        # 固定光学量只计算一次。没有读取真实波前或真实物体。
        with torch.no_grad():
            shifts = []
            for view in range(len(PHYSICAL_VIEWS)):
                blocks = [g.incoherent_psf([view], z, torch.zeros(MODES, device=device),
                    normalized=True, piston_tip_tilt=True, psf_binning=1)[0]
                    for z in depths.split(21)]
                shifts.append(torch.cat(blocks, dim=1))
            shifts = torch.cat(shifts)
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

    def forward(self, volume, wf, views):
        pupil = self.propagation*torch.exp(-2j*torch.pi*
            torch.einsum('m,mhw->hw', wf, self.basis))
        field = self.aperture[views, None]*pupil[None]
        field = field*self.ramp_y[views, :, :, None]*self.ramp_x[views, :, None, :]
        amplitude = torch.fft.ifft2(field)
        psf = (amplitude[..., self.sample_y[:, None], self.sample_x]/self.energy).abs().square()
        otf = torch.fft.rfft2(psf, s=self.shape)
        otf = otf*self.otf_y[views, :, :, None]*self.otf_x[views, :, None, :]
        spectrum = (torch.fft.rfft2(volume)[:, None]*otf[None]).sum(2)
        return torch.fft.irfft2(spectrum, s=self.shape)[..., ::self.stride, ::self.stride]


# 2. 更换噪声模型或先验时修改这两个函数。
def data_loss(rate, raw):
    """Poisson–Gaussian 方差匹配准似然；保留 raw-BIAS 的负观测。"""
    y, rate = raw.double()-BIAS, rate.double().clamp_min(0)
    ref = y.clamp_min(0)  # 只定义与待估参数无关的参考项
    return 2*((rate-ref)-(y+READ_VARIANCE)*
        torch.log1p((rate-ref)/(ref+READ_VARIANCE))).sum()


def spatial_penalty(relative_volume):
    terms = []
    for axis in (-2, -1):
        curvature = torch.diff(relative_volume, n=2, dim=axis)
        terms.append(torch.nn.functional.smooth_l1_loss(
            curvature, torch.zeros_like(curvature), beta=.03))
    return sum(terms)/2


# 3. 求解函数只接收模型和观测，可直接复用于自己的任务。
def reconstruct(project, data):
    device, nviews = data.device, data.shape[1]
    log_signal = torch.nn.Parameter(data.new_zeros((1, *VOLUME_SHAPE)))
    log_background = torch.nn.Parameter(data.new_zeros(()))
    wf = torch.nn.Parameter(data.new_zeros(MODES))
    mask = torch.ones_like(wf)
    mask[FIXED_MODES] = 0
    views = torch.arange(nviews, device=device)
    counts = (data.double()-BIAS).sum()
    if counts <= 0:
        raise ValueError('扣偏置后的总计数必须为正，请检查 BIAS 和输入单位。')
    with torch.no_grad():
        flux = sum(project(log_signal.exp(), wf, v).double().sum()
            for v in views.split(VIEWS_PER_STEP))
        scale = (counts/flux).float()

    def volume():
        return scale*(log_signal.exp()+log_background.exp())*.5

    optimizer = torch.optim.Adam([
        {'params': [log_signal], 'lr': LR_VOLUME},
        {'params': [wf], 'lr': LR_WAVEFRONT},
        {'params': [log_background], 'lr': LR_BACKGROUND}], betas=(.9, .99), eps=1e-12)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.01, total_iters=WARMUP),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS-WARMUP)], milestones=[WARMUP])
    random = torch.Generator(device=device).manual_seed(0)
    prior_random = torch.Generator(device=device).manual_seed(1)
    batches_per_epoch = (nviews+VIEWS_PER_STEP-1)//VIEWS_PER_STEP
    for step in range(STEPS):
        if step % batches_per_epoch == 0:
            batches = iter(views[torch.randperm(nviews, generator=random,
                device=device)].split(VIEWS_PER_STEP))
        selected = next(batches)
        optimizer.zero_grad(set_to_none=True)
        current = volume()
        rate = project(current, wf*mask, selected)
        fit = data_loss(rate, data[:, selected])/counts*nviews/len(selected)
        z = torch.randint(VOLUME_SHAPE[0], (PRIOR_DEPTHS,), generator=prior_random, device=device)
        prior = spatial_penalty(current[:, z]/scale)
        loss = fit + REGULARIZATION*prior
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 0 or (step+1) % 400 == 0 or step+1 == STEPS:
            print(f'{step+1:4d}/{STEPS}  data={fit.item():.6g}  prior={prior.item():.6g}', flush=True)
    with torch.no_grad():
        return volume(), wf*mask


# 4. 保存自己的估计；真值可选，且只在优化结束后读取。
@torch.no_grad()
def save_results(g, project, estimated_photons, wf):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    volume = estimated_photons/INTENSITY_SCALE
    views = torch.arange(len(PHYSICAL_VIEWS), device=wf.device)
    reprojected = torch.cat([project(volume, wf, v) for v in views.split(VIEWS_PER_STEP)], 1)
    for name, value in [('volume_estimated', volume), ('reprojected_lf', reprojected),
            ('reprojected_lf_sensor', reprojected*INTENSITY_SCALE+BIAS)]:
        # tifffile.imwrite(OUTPUT/f'{name}.tiff', value[0].cpu().clamp(0,65535).numpy().astype(np.uint16), compression='zlib')
        tifffile.imwrite(OUTPUT/f'{name}.tiff', value[0].cpu().numpy(), compression='zlib')
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
            raise ValueError('TRUE_WF 必须包含 MODES 个系数；自己的数据可将它设为 None。')
        maps = {'true': phase(truth), 'estimated': recovered, 'error': phase(wf-truth)}
        summary['phase_rms_error'] = g.calcRMS(wf-truth, normalized=True, piston_tip_tilt=True).item()
    if CLEAN_LF is not None:
        clean = torch.as_tensor(tifffile.imread(CLEAN_LF), dtype=wf.dtype, device=wf.device)[None]
        if clean.shape != reprojected.shape:
            raise ValueError('CLEAN_LF 的形状必须与重投影一致；自己的数据可将它设为 None。')
        summary['lf_relative_l2'] = ((reprojected-clean).norm()/clean.norm()).item()
    for name, waves in maps.items():
        tifffile.imwrite(OUTPUT/f'phase_{name}_waves.tiff', waves)
    # exp(-2*pi*i*wavefront) 是本模型的相位约定。
    tifffile.imwrite(OUTPUT/'phase_estimated_rad.tiff', -2*np.pi*recovered)
    tifffile.imwrite(OUTPUT/'pupil_mask.tiff', pupil.astype(np.uint8))

    return summary


def main():
    device = torch.device(DEVICE)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('未检测到 CUDA；请使用 CUDA 环境，或将 DEVICE 改为 cpu 做小规模调试。')
        torch.cuda.set_device(device)
    if not (0 < WARMUP < STEPS and INTENSITY_SCALE > 0 and READ_VARIANCE > 0):
        raise ValueError('需要 0 < WARMUP < STEPS、正的 INTENSITY_SCALE 和 READ_VARIANCE。')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    raw = torch.as_tensor(tifffile.imread(DATA), dtype=torch.float32, device=device)
    expected = (len(PHYSICAL_VIEWS), *(len(range(0, n, SAMPLE_STRIDE)) for n in VOLUME_SHAPE[-2:]))
    if tuple(raw.shape) != expected or not torch.isfinite(raw).all():
        raise ValueError(f'LF 应为 {expected} 且数值有限，实际为 {tuple(raw.shape)}；请检查网格与视角。')
    synchronize(device)
    started = time.perf_counter()
    g = PsfGenerator5D(**OPTICS, device=device, input_views=PHYSICAL_VIEWS)
    project = Projection(g)
    synchronize(device)
    setup_s = time.perf_counter()-started
    started = time.perf_counter()
    volume, wf = reconstruct(project, raw[None])
    synchronize(device)
    solve_s = time.perf_counter()-started
    if not torch.isfinite(volume).all() or not torch.isfinite(wf).all():
        raise FloatingPointError('重建包含非有限值，请检查数据、学习率及模型。')
    summary = save_results(g, project, volume, wf)
    summary.update(steps=STEPS, setup_s=setup_s, solve_s=solve_s,
        input=relative_path(DATA), intensity_scale=INTENSITY_SCALE, device=device.type,
        volume_shape=VOLUME_SHAPE, psf_size=PSF_SIZE, sample_stride=SAMPLE_STRIDE,
        physical_views=PHYSICAL_VIEWS, optics=OPTICS, modes=MODES, fixed_modes=FIXED_MODES,
        bias=BIAS, read_variance=READ_VARIANCE, warmup=WARMUP,
        learning_rates=[LR_VOLUME, LR_WAVEFRONT, LR_BACKGROUND],
        regularization=REGULARIZATION, prior_depths=PRIOR_DEPTHS)
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print(f'结果：{relative_path(OUTPUT)}')


if __name__ == '__main__':
    main()
