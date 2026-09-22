"""100 个散点高斯–泊松噪声实验：直接运行本文件，无需命令行参数。

原生 Adam + 100 步预热 + 二阶平滑正则；只导入基础光学类。
固定 seed=0，排除 0/1/2/4；10 组 RMS1/len21 × 5 个峰值光子数，PSF225。
自动使用最多 8 张可见 GPU，每例仅用单卡。已完成且配置相同的结果会跳过。
"""
import csv
import json
from pathlib import Path
from inspect import getfile
import shutil
import time
import traceback

import numpy as np
import tifffile
import torch
import torch.multiprocessing as mp
# from lfpsf_torch_batch5 import PsfGenerator5D
from lfpsf_torch_shift_blur_batched import PsfGenerator5D

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/batch_points_poisson_gaussian_output'
SHAPE = (1, 101, 675, 675)
PSF_SIZE = 225
PHOTONS = (10, 30, 100, 300, 1000)
GAUSSIAN_MEAN, GAUSSIAN_VARIANCE = 105., 9.
STEPS, BATCH_SIZE = 2300, 3
WARMUP, REGULARIZATION, PRIOR_DEPTHS = 100, 3e-5, 16
EXCLUDED = [0, 1, 2, 4]

PHYSICAL_VIEWS = [
    112, 113, 128, 127, 126, 111, 96, 97, 98, 99, 114, 129, 144,
    143, 142, 141, 140, 125, 110, 95, 80, 81, 82, 83, 84, 85, 100,
    115, 130, 145, 160, 159, 158, 157, 156, 155, 154, 139, 124, 109,
    94, 79, 64, 65, 66, 67, 68, 69, 70, 71, 86, 101, 116, 131, 146,
    161, 175, 174, 173, 172, 171, 170, 169, 153, 138, 123, 108, 93,
    78, 63, 49, 50, 51, 52, 53, 54, 55, 117, 187, 107, 37,
]


@torch.no_grad()
def make_optics(device):
    g = PsfGenerator5D(lfpsf_shape=(15, 15, 1, 525, 525), M=7.85, n=1,
        na_detection=.5, MLPitch=56.4e-6, Nnum=15, OSR=3, fml=444.15e-6,
        lam_detection=525e-9, dz=5e-6, xy_downsample=3, device=device,
        input_views=PHYSICAL_VIEWS)
    depths = torch.arange(101, dtype=torch.float32, device=device) - 50
    zero_wf = torch.zeros(21, device=device)
    shifts = []
    for view in range(81):
        blocks = []
        for depth in depths.split(21):
            shift, _ = g.incoherent_psf([view], depth, zero_wf,
                normalized=True, piston_tip_tilt=True, psf_binning=1)
            blocks.append(shift)
        shifts.append(torch.cat(blocks, dim=1))
    return g, depths, torch.cat(shifts)


class Projection(torch.nn.Module):
    """固定光学量只初始化一次，当前 wf 的 PSF 和 OTF 在每次投影中生成。"""
    def __init__(self, g, depths, shifts, psf_size):
        device = g.device
        super().__init__()
        self.basis = torch.stack([torch.fft.fftshift(
            g.zernike_polynomial(j, normalized=True)*g.kmask2) for j in range(21)])
        self.propagation = torch.exp(-2j*torch.pi*
            torch.fft.fftshift(g.phase_angular_spectrum)*depths[:, None, None]*g.dz_rms)
        self.aperture = torch.fft.fftshift(g.kbase)*g.uv_mla_fresnel
        h, w = g.krho.shape
        fy = (torch.arange(h, device=device)-h//2)/h
        fx = (torch.arange(w, device=device)-w//2)/w
        self.ramp_y = torch.exp(2j*torch.pi*shifts[..., 0, None]*fy)
        self.ramp_x = torch.exp(2j*torch.pi*shifts[..., 1, None]*fx)
        self.sample_y = (torch.arange(psf_size, device=device)-psf_size//2)%h
        self.sample_x = (torch.arange(psf_size, device=device)-psf_size//2)%w
        height, width = SHAPE[-2:]
        sy = shifts[..., 0]+(height-psf_size)//2-height//2
        sx = shifts[..., 1]+(width-psf_size)//2-width//2
        iy = torch.arange(height, device=device)
        ix = torch.arange(width//2+1, device=device)
        self.otf_y = torch.exp((-2j*torch.pi/height)*
            (sy[..., None]*iy).remainder(height).double()).to(g.kbase.dtype)
        self.otf_x = torch.exp((-2j*torch.pi/width)*
            (sx[..., None]*ix).remainder(width).double()).to(g.kbase.dtype)
        self.energy = g.energy_ratio_complex

    def forward(self, volume, wf, views):
        pupil = self.propagation*torch.exp(-2j*torch.pi*torch.einsum('m,mhw->hw', wf, self.basis[:len(wf)]))
        field = self.aperture[views, None]*pupil[None]
        field = field*self.ramp_y[views, :, :, None]*self.ramp_x[views, :, None, :]
        amplitude = torch.fft.ifft2(field)
        psf = (amplitude[..., self.sample_y[:, None], self.sample_x]/self.energy).abs().square()
        otf = torch.fft.rfft2(psf, s=SHAPE[-2:])
        otf = otf*self.otf_y[views, :, :, None]*self.otf_x[views, :, None, :]
        spectrum = (torch.fft.rfft2(volume)[:, None]*otf[None]).sum(2)
        return torch.fft.irfft2(spectrum, s=SHAPE[-2:])[..., ::5, ::5]


def poisson_gaussian_loss(rate, raw):
    """方差匹配准似然；已知偏置105、方差9，保留扣偏置后的负观测。"""
    y = raw.double()-GAUSSIAN_MEAN
    rate = rate.double().clamp_min(0)
    ref = y.clamp_min(0)  # 只减去与参数无关的参考值，不截断实际观测 y。
    return 2*((rate-ref)-(y+GAUSSIAN_VARIANCE)*
        torch.log1p((rate-ref)/(ref+GAUSSIAN_VARIANCE))).sum()


def add_noise(clean, photons, generator):
    coefficient = float(photons/clean.max().item())
    poisson = torch.poisson(clean*coefficient, generator=generator)
    gaussian = GAUSSIAN_MEAN + GAUSSIAN_VARIANCE**.5*torch.randn(
        clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    return poisson+gaussian, coefficient, poisson, gaussian


def spatial_penalty(relative_volume):
    """每层 XY 二阶差分的平滑 L1；抑制网纹，同时避免强惩罚真实大梯度。"""
    terms = []
    for axis in (-2, -1):
        curvature = torch.diff(relative_volume, n=2, dim=axis)
        terms.append(torch.nn.functional.smooth_l1_loss(curvature, torch.zeros_like(curvature), beta=.03))
    return sum(terms)/2


SOURCE_VOLUME = 'seed=0 CUDA randint; 100 points, amplitude=10000'


def true_volume(device):
    torch.manual_seed(0)
    volume = torch.zeros(SHAPE, device=device)
    z = torch.randint(10, SHAPE[1]-10, (100,), device=device)
    y = torch.randint(50, SHAPE[2]-50, (100,), device=device)
    x = torch.randint(50, SHAPE[3]-50, (100,), device=device)
    volume[0, z, y, x] = 10000
    np.savetxt(OUTPUT/'random_points.csv', torch.stack((z, y, x), 1).cpu().numpy(),
        delimiter=',', header='z,y,x', comments='', fmt='%d')
    return volume


def cases():
    specs = [dict(name='rms1_len21_shared', length=21, rms=1., groups=[1])]
    specs += [dict(name=f'rms1_len21_{i:02d}', length=21, rms=1., groups=[1]) for i in range(1, 10)]
    return [dict(index=i, **case) for i, case in enumerate(specs)]


@torch.no_grad()
def wavefronts(g):
    random = torch.Generator(device='cpu').manual_seed(0)
    result = []
    for case in cases():
        wf = torch.randn(case['length'], generator=random)
        wf[EXCLUDED] = 0
        wf = wf.to(g.device)
        if case['rms'] == 0:
            wf.zero_()
        else:
            wf.mul_(case['rms']/g.calcRMS(wf, normalized=True, piston_tip_tilt=True))
        result.append(wf.cpu())
    return result


@torch.no_grad()
def simulate(g, volume, wf, depths, shifts, psf_size):
    """独立 MFT PSF + 显式空间移位造数据，不调用待测 Projection。"""
    height, width = SHAPE[-2:]
    vf = torch.fft.rfft2(volume)
    iy = torch.arange(height, device=g.device)[None, :, None]
    ix = torch.arange(width, device=g.device)[None, None, :]
    iz = torch.arange(SHAPE[1], device=g.device)[:, None, None]
    first = (height-psf_size)//2
    measured = []
    for view in range(81):
        blocks = []
        for start in range(0, len(depths), 21):
            part = slice(start, start+21)
            _, psf = g.incoherent_psf([view], depths[part], wf, normalized=True,
                piston_tip_tilt=True, psf_binning=1, shift_fixed=shifts[view:view+1, part],
                out_size=(psf_size, psf_size))
            blocks.append(psf[0])
        padded = volume.new_zeros((SHAPE[1], height, width))
        padded[:, first:first+psf_size, first:first+psf_size] = torch.cat(blocks)
        shift = shifts[view]
        rolled = padded[iz, (iy-shift[:, 0, None, None])%height, (ix-shift[:, 1, None, None])%width]
        otf = torch.fft.rfft2(torch.fft.ifftshift(rolled, dim=(-2, -1)))
        measured.append(torch.fft.irfft2((vf*otf).sum(1), s=(height, width))[..., ::5, ::5])
    return torch.stack(measured, 1).clamp_min(0)


def reconstruct(project, data):
    """真实像差和真实 volume 不传入求解函数；只用观测初始化。"""
    device = data.device
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    log_signal = torch.nn.Parameter(data.new_zeros(SHAPE))
    log_background = torch.nn.Parameter(data.new_zeros(()))
    wf = torch.nn.Parameter(data.new_zeros(21))
    mask = torch.ones_like(wf)
    mask[EXCLUDED] = 0
    views = torch.arange(81, device=device)
    counts = (data.double()-GAUSSIAN_MEAN).sum()
    if counts <= 0:
        raise ValueError('扣除已知偏置后的总计数非正，无法从观测初始化。')
    with torch.no_grad():
        flux = sum(project(log_signal.exp(), wf, v).double().sum() for v in views.split(BATCH_SIZE))
        scale = (counts/flux).float()
    optimizer = torch.optim.Adam([
        {'params': [log_signal], 'lr': .3}, {'params': [wf], 'lr': .1},
        {'params': [log_background], 'lr': .01}], betas=(.9, .99), eps=1e-12)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.01, total_iters=WARMUP),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STEPS-WARMUP)], milestones=[WARMUP])
    random = torch.Generator(device=device).manual_seed(0)
    prior_random = torch.Generator(device=device).manual_seed(1)
    history = []
    for step in range(STEPS):
        if step % 27 == 0:
            batches = iter(views[torch.randperm(81, generator=random, device=device)].split(BATCH_SIZE))
        selected = next(batches)
        optimizer.zero_grad(set_to_none=True)
        volume = scale*(log_signal.exp()+log_background.exp())*.5
        rate = project(volume, wf*mask, selected)
        data_loss = poisson_gaussian_loss(rate, data[:, selected])/counts*81/len(selected)
        z = torch.randint(SHAPE[1], (PRIOR_DEPTHS,), generator=prior_random, device=device)
        prior = spatial_penalty(volume[:, z]/scale)
        loss = data_loss + REGULARIZATION*prior
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step == 0 or (step+1) % 400 == 0 or step+1 == STEPS:
            row = dict(step=step+1, data_loss=data_loss.item(), prior=prior.item())
            if not np.isfinite(list(row.values())).all():
                raise FloatingPointError(f'nonfinite loss: {row}')
            history.append(row)
            print(f'{device} {row}', flush=True)
    with torch.no_grad():
        volume = scale*(log_signal.exp()+log_background.exp())*.5
        estimated_wf = wf*mask
    torch.cuda.synchronize(device)
    return volume.detach(), estimated_wf.detach(), time.perf_counter()-started, history


@torch.no_grad()
def save_phase(out, g, estimate, truth):
    pupil = torch.fft.fftshift(g.kmask2).cpu().numpy()
    for label, wf in [('true', truth), ('estimated', estimate), ('error', estimate-truth)]:
        waves = torch.fft.fftshift(g.masked_phase_array(wf, normalized=True, piston_tip_tilt=True)).cpu().numpy()
        for unit, value in [('waves', waves), ('rad', -2*np.pi*waves)]:
            tifffile.imwrite(out/f'phase_{label}_centered_{unit}.tiff', value)
    tifffile.imwrite(out/'pupil_mask_centered.tiff', pupil.astype(np.uint8))


def save_tiff(path, tensor):
    tifffile.imwrite(path, tensor[0].detach().cpu().numpy(), compression='zlib',
        imagej=True, metadata={'axes': 'ZYX'})


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for current, saved in [(Path(__file__), OUTPUT/'script_source.py'),
            (Path(getfile(PsfGenerator5D)), OUTPUT/'optics_source.py')]:
        if saved.exists() and current.read_bytes() != saved.read_bytes():
            raise RuntimeError(f'源码已改变，请设置新的 OUTPUT 目录：{current.name}')
        if not saved.exists():
            shutil.copy2(current, saved)
    if (OUTPUT/'catalog.json').exists():
        return
    g, depths, shifts = make_optics('cuda:0')
    truth = true_volume('cuda:0')
    save_tiff(OUTPUT/'volume_true.tiff', truth)
    (OUTPUT/'wavefronts').mkdir(exist_ok=True)
    records, clean_images = [], []
    for case, wf in zip(cases(), wavefronts(g)):
        clean = simulate(g, truth, wf.to(g.device), depths, shifts, PSF_SIZE).cpu()
        clean_images.append(clean)
        path = f'wavefronts/{case["index"]:02d}_{case["name"]}.txt'
        np.savetxt(OUTPUT/path, wf.numpy())
        records.append(dict(**case, path=path, actual_rms=g.calcRMS(
            wf.to(g.device), normalized=True, piston_tip_tilt=True).item()))
    # 噪声在 CPU 上按固定次序一次生成，与 GPU 数量及任务调度无关。
    noise_random = torch.Generator(device='cpu').manual_seed(0)
    jobs = []
    for photons in PHOTONS:
        for case, clean in zip(records, clean_images):
            out = OUTPUT/f'photons{photons}'/f'{case["index"]:02d}_{case["name"]}'
            out.mkdir(parents=True, exist_ok=True)
            raw, coefficient, poisson, gaussian = add_noise(clean, photons, noise_random)
            for name, value in [('lf', raw), ('lf_clean', clean),
                    ('lf_expected_photons', clean*coefficient),
                    ('poisson_counts', poisson), ('gaussian_noise', gaussian)]:
                save_tiff(out/f'{name}.tiff', value)
            info = dict(index=case['index'], name=case['name'], photons=photons,
                coefficient=coefficient, clean_lf_max=clean.max().item(),
                expected_photon_max=(clean*coefficient).max().item(),
                expected_photon_mean=(clean*coefficient).mean().item(),
                gaussian_empirical_mean=gaussian.double().mean().item(),
                gaussian_empirical_variance=gaussian.double().var(unbiased=False).item(),
                corrected_negative_fraction=((raw-GAUSSIAN_MEAN)<0).float().mean().item(),
                poisson_residual_mean=(poisson.double()-clean.double()*coefficient).mean().item(),
                output=str(out.relative_to(OUTPUT)), wf_path=case['path'])
            (out/'noise.json').write_text(json.dumps(info, indent=2))
            jobs.append(info)
    catalog = dict(seed=0, noise_seed=0, excluded=EXCLUDED, cases=records, jobs=jobs,
        photon_levels=PHOTONS, psf_size=PSF_SIZE, expected_cases=len(jobs),
        volume_shape=SHAPE, source_volume=SOURCE_VOLUME, steps=STEPS,
        warmup=WARMUP, regularization=REGULARIZATION, prior_depths=PRIOR_DEPTHS,
        gaussian_mean=GAUSSIAN_MEAN, gaussian_variance=GAUSSIAN_VARIANCE,
        noise_formula='torch.poisson(lf*coefficient) + 105 + 3*torch.randn_like(lf)',
        coefficient_definition='peak photons / clean LF maximum over all 81 views',
        noise_rng='one CPU Generator(seed=0), photon-major then sample order; Poisson then Gaussian',
        adam_lr=[.3, .1, .01], adam_betas=[.9, .99], adam_eps=1e-12,
        loss='variance-matched quasi likelihood; signed raw-105, variance=rate+9; not exact PG likelihood',
        simulation='independent MFT PSF with fixed zero-wf shifts and spatial roll convolution',
        volume_estimated_units='original volume units, dividing the photon-domain estimate by known coefficient',
        reprojected_lf_units='original clean LF units; compare against lf_clean.tiff',
        reprojected_lf_sensor_units='expected sensor mean = coefficient*reprojected_lf+105',
        phase_convention='fftshift centered; rad=-2*pi*waves; zero outside pupil',
        torch_version=torch.__version__)
    (OUTPUT/'catalog.tmp').write_text(json.dumps(catalog, indent=2))
    (OUTPUT/'catalog.tmp').replace(OUTPUT/'catalog.json')


def worker(rank, worker_count):
    device = f'cuda:{rank}'
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    catalog = json.loads((OUTPUT/'catalog.json').read_text())
    truth = torch.as_tensor(tifffile.imread(OUTPUT/'volume_true.tiff'), device=device)[None]
    started = time.perf_counter()
    g, depths, shifts = make_optics(device)
    torch.cuda.synchronize()
    optics_s = time.perf_counter()-started
    started = time.perf_counter()
    project = Projection(g, depths, shifts, PSF_SIZE)
    torch.cuda.synchronize()
    projection_setup_s = time.perf_counter()-started
    for job in catalog['jobs'][rank::worker_count]:
        out = OUTPUT/job['output']
        if (out/'summary.json').exists():
            print(f'skip {out.relative_to(OUTPUT)}', flush=True)
            continue
        print(f'START {device} photons={job["photons"]} sample={job["index"]+1}', flush=True)
        try:
            run_case(device, out, job, truth, g, project, optics_s, projection_setup_s)
        except Exception as error:
            (out/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__), indent=2))
            traceback.print_exc()
        torch.cuda.empty_cache()


def run_case(device, out, job, truth, g, project, optics_s, projection_setup_s):
    started = time.perf_counter()
    coefficient = job['coefficient']
    data = torch.as_tensor(tifffile.imread(out/'lf.tiff'), device=device)[None]
    clean = torch.as_tensor(tifffile.imread(out/'lf_clean.tiff'), device=device)[None]
    wf_true = torch.tensor(np.loadtxt(OUTPUT/job['wf_path']), dtype=torch.float32, device=device)
    views = torch.arange(81, device=device)
    t = time.perf_counter()
    with torch.no_grad():
        predicted_true = torch.cat([project(truth, wf_true, v) for v in views.split(BATCH_SIZE)], 1)
        forward_error = ((predicted_true-clean).norm()/clean.norm()).item()
    if not np.isfinite(forward_error) or forward_error > 2e-5:
        raise ValueError(f'MFT / FFT projection disagreement: {forward_error}')
    forward_validation_s = time.perf_counter()-t
    del predicted_true
    torch.cuda.reset_peak_memory_stats()
    estimated_photons, wf, solve_s, history = reconstruct(project, data)
    t = time.perf_counter()
    with torch.no_grad():
        volume = estimated_photons/coefficient
        reprojected = torch.cat([project(volume, wf, v) for v in views.split(BATCH_SIZE)], 1)
        sensor_rate = reprojected*coefficient
        sensor_mean = sensor_rate+GAUSSIAN_MEAN
        net = data-GAUSSIAN_MEAN
        row = dict(index=job['index'], name=job['name'], photons=job['photons'],
            coefficient=coefficient, psf_size=PSF_SIZE, wf_length=21, requested_rms=1.,
            phase_rms_error=g.calcRMS(wf-wf_true, normalized=True, piston_tip_tilt=True).item(),
            wf_l2_error=(wf-wf_true).norm().item(),
            volume_relative_l2=((volume-truth).norm()/truth.norm()).item(),
            lf_relative_l2=((reprojected-clean).norm()/clean.norm()).item(),
            noisy_lf_relative_l2=((net/coefficient-clean).norm()/clean.norm()).item(),
            sensor_fit_relative_l2=((sensor_mean-data).norm()/data.norm()).item(),
            standardized_residual_rms=((data.double()-sensor_mean.double()).square()/
                (sensor_rate.double().clamp_min(0)+GAUSSIAN_VARIANCE)).mean().sqrt().item(),
            relative_quasi_deviance=(poisson_gaussian_loss(sensor_rate, data)/net.double().sum()).item(),
            forward_relative_l2=forward_error, forward_validation_s=forward_validation_s,
            solve_s=solve_s, optics_setup_s=optics_s, projection_setup_s=projection_setup_s,
            standalone_setup_and_solve_s=optics_s+projection_setup_s+solve_s,
            steps=STEPS, device=torch.device(device).type,
            gpu_name=torch.cuda.get_device_name(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if not torch.isfinite(volume).all() or not torch.isfinite(reprojected).all() or not torch.isfinite(wf).all():
            raise FloatingPointError('nonfinite reconstruction')
        assert volume.min() >= 0 and (wf[EXCLUDED] == 0).all()
    torch.cuda.synchronize()
    row['evaluation_s'] = time.perf_counter()-t
    t = time.perf_counter()
    for name, value in [('volume_estimated', volume), ('reprojected_lf', reprojected),
            ('reprojected_lf_sensor', sensor_mean)]:
        save_tiff(out/f'{name}.tiff', value)
    np.savetxt(out/'wf_true.txt', wf_true.cpu().numpy())
    np.savetxt(out/'wf_estimated.txt', wf.cpu().numpy())
    save_phase(out, g, wf, wf_true)
    (out/'history.json').write_text(json.dumps(history, indent=2))
    row['save_s'] = time.perf_counter()-t
    row['case_wall_s'] = time.perf_counter()-started
    (out/'summary.tmp').write_text(json.dumps(row, indent=2, allow_nan=False))
    (out/'failure.json').unlink(missing_ok=True)
    (out/'summary.tmp').replace(out/'summary.json')
    print(f'DONE {json.dumps(row)}', flush=True)


def aggregate():
    catalog = json.loads((OUTPUT/'catalog.json').read_text())
    rows, missing, failures = [], [], []
    for job in catalog['jobs']:
        out = OUTPUT/job['output']
        if (out/'summary.json').exists():
            rows.append(json.loads((out/'summary.json').read_text()))
        else:
            missing.append(job['output'])
        if (out/'failure.json').exists():
            failures.append(job['output'])
    if rows:
        with (OUTPUT/'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    statistics = {}
    for photons in PHOTONS:
        subset = [r for r in rows if r['photons'] == photons]
        statistics[str(photons)] = {key: dict(mean=float(np.mean(values)), median=float(np.median(values)),
            maximum=float(np.max(values))) for key in ['phase_rms_error', 'volume_relative_l2',
            'lf_relative_l2', 'noisy_lf_relative_l2', 'standardized_residual_rms', 'solve_s',
            'standalone_setup_and_solve_s'] if (values := [r[key] for r in subset])}
    result = dict(completed=len(rows), expected=len(catalog['jobs']), missing=missing,
        failures=failures, statistics=statistics)
    (OUTPUT/'aggregate.json').write_text(json.dumps(result, indent=2))
    lines = ['# 散点高斯–泊松噪声实验', '', f'完成 {len(rows)}/50；PSF225，10组RMS1/len21。',
        '峰值光子数=P，系数=P/lf.max()；噪声为Poisson(lf*系数)+N(105,9)。',
        '原生Adam、2300步、100步预热、弱二阶正则；数据项为方差匹配准似然。',
        'LF误差相对干净LF；volume与reprojected_lf均保存为原始强度单位。',
        'lf.tiff是原始带偏置观测，reprojected_lf_sensor.tiff是对应传感器期望值。', '',
        '| 峰值光子数 | 完成 | WF RMS均值 / wave | LF误差均值 | volume误差均值 | 求解均值 / s |',
        '|---:|---:|---:|---:|---:|---:|']
    for photons in PHOTONS:
        subset = [r for r in rows if r['photons'] == photons]
        if subset:
            stats = statistics[str(photons)]
            values = [stats[k]['mean'] for k in ['phase_rms_error','lf_relative_l2','volume_relative_l2','solve_s']]
            lines.append(f'| {photons} | {len(subset)}/10 | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    lines += ['', '原始带噪声LF的偏置会使传感器域relative L2显得很小；主要比较干净LF和WF误差。',
        '散点volume逐体素误差对点位偏移敏感；不以该项独立判断质量。',
        '详见results.csv及每例summary.json；完成计算不代表精度达标。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('当前无法访问 CUDA GPU。')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    prepare()
    preparation_s = time.perf_counter()-started
    torch.cuda.empty_cache()
    workers = min(8, torch.cuda.device_count())
    mp.spawn(worker, args=(workers,), nprocs=workers, join=True)
    result = aggregate()
    (OUTPUT/'batch_time.json').write_text(json.dumps(dict(preparation_s=preparation_s,
        batch_wall_s=time.perf_counter()-started, workers=workers,
        timing='includes all 50 noise realizations, independent MFT simulation, reconstruction/evaluation/saving'), indent=2))
    if result['completed'] != result['expected'] or result['failures']:
        raise RuntimeError('实验未全部完成，请检查 OUTPUT 目录中的 aggregate.json')


if __name__ == '__main__':
    main()
