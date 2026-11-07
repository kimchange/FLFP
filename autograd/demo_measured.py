"""Joint reconstruction of cropped measured light fields.

Run: python autograd/demo_measured.py
"""
import json
import math
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


# 1. Settings. Change SAMPLE to select a measured region.
SAMPLE = 'beads'
# beads, neun_cells_left, neun_cells_right, worm_slice_upper, worm_slice_lower
ROOT = Path(__file__).resolve().parents[1]
DEVICE = 'cuda:0'
STEPS, VIEWS_PER_STEP, DEPTH_CHUNK = 2400, 3, 8
RELAXATION, REGULARIZATION = .2, .1
LR_WAVEFRONT, LR_PUPIL, LR_ILLUMINATION = .003, .0002, .003
BLUR_SIGMAS, BLUR_STAGE_STEPS, SEED = (4, 2, 1, 0), 400, 0
FIXED_MODES = [0, 1, 2, 4]
AIR_VIEWS = [
    112, 113, 128, 127, 126, 111, 96, 97, 98, 99, 114, 129, 144,
    143, 142, 141, 140, 125, 110, 95, 80, 81, 82, 83, 84, 85, 100,
    115, 130, 145, 160, 159, 158, 157, 156, 155, 154, 139, 124, 109,
    94, 79, 64, 65, 66, 67, 68, 69, 70, 71, 86, 101, 116, 131, 146,
    161, 175, 174, 173, 172, 171, 170, 169, 153, 138, 123, 108, 93,
    78, 63, 49, 50, 51, 52, 53, 54, 55, 117, 187, 107, 37,
]


# 2. Optical model, fitted pupil intensity, and excitation-shutter overlap.
class Projection:
    def __init__(self, settings, device=DEVICE):
        self.cfg = settings
        o = settings['optics']
        self.n, self.side = settings['volume_size'], settings['psf_size']
        self.scan = o['Nnum']
        self.views = [(v//15-7+self.scan//2)*self.scan+v%15-7+self.scan//2 for v in AIR_VIEWS]
        self.depths = (torch.arange(settings['depth_count'], device=device,
            dtype=torch.float32)-settings['depth_count']//2)*o['dz_um']
        g = PsfGenerator5D(lfpsf_shape=(self.scan,self.scan,1,
            o['frequency_base_size'],o['frequency_base_size']), M=o['M'], n=o['n'],
            na_detection=o['NA'], MLPitch=o['MLPitch_um']*1e-6, Nnum=self.scan,
            OSR=3, fml=o['Fml_um']*1e-6, lam_detection=o['wavelength_nm']*1e-9,
            dz=o['dz_um']*1e-6, xy_downsample=3, device=device, input_views=self.views)
        with torch.no_grad():
            phase_z = -2*torch.pi*torch.fft.fftshift(g.phase_angular_spectrum)[None]*\
                (self.depths/o['dz_um'])[:,None,None]*g.dz_rms
            shifts = []
            for view in range(len(self.views)):
                blocks = [g.decouple_shift_unwrapped_slicing(g.uv_mla_fresnel[view:view+1,None],
                    phase_z[None,first:first+DEPTH_CHUNK], psf_binning=1)[0][0]
                    for first in range(0,len(self.depths),DEPTH_CHUNK)]
                shifts.append(torch.cat(blocks))
            self.shifts = torch.stack(shifts).long()
            self.basis = torch.stack([torch.fft.fftshift(
                g.zernike_polynomial(j,normalized=True)*g.kmask2) for j in range(36)])
            self.propagation = torch.exp(-2j*torch.pi*torch.fft.fftshift(g.phase_angular_spectrum)*
                (self.depths/o['dz_um'])[:,None,None]*g.dz_rms)
            self.aperture = torch.fft.fftshift(g.kbase)*g.uv_mla_fresnel
            self.energy = torch.as_tensor(g.energy_ratio_complex,device=device)
            h,w = self.basis.shape[-2:]
            fy,fx = (torch.arange(h,device=device)-h//2)/h,(torch.arange(w,device=device)-w//2)/w
            self.ramp_y = torch.exp(2j*torch.pi*self.shifts[...,0,None]*fy)
            self.ramp_x = torch.exp(2j*torch.pi*self.shifts[...,1,None]*fx)
            self.axis = torch.arange(self.side,device=device,dtype=torch.float32)-self.side//2
            self.sy,self.sx = self.axis.long()%h,self.axis.long()%w
            iy = torch.arange(self.n,device=device,dtype=torch.float64)
            ix = torch.arange(self.n//2+1,device=device,dtype=torch.float64)
            shift = self.shifts-self.side//2
            self.otf_y = torch.exp(-2j*torch.pi/self.n*
                (shift[...,0,None]*iy).remainder(self.n)).to(self.aperture.dtype)
            self.otf_x = torch.exp(-2j*torch.pi/self.n*
                (shift[...,1,None]*ix).remainder(self.n)).to(self.aperture.dtype)
            q = torch.arange(settings['context_size'],device=device,dtype=torch.float32)*self.scan/3
            yy,xx = torch.meshgrid((2*q+1)/self.n-1,(2*q+1)/self.n-1,indexing='ij')
            self.grid = torch.stack([xx,yy],-1)[None]
            self.pupil_mask = self.basis[0] != 0
            self.pupil_log_count = math.log(int(self.pupil_mask.sum()))
        self.tangent_limit = o['NA']/math.sqrt(o['n']**2-o['NA']**2)

    def pupil_field(self,wave,pupil):
        phase = torch.exp(-2j*torch.pi*torch.einsum('m,mhw->hw',wave,self.basis))
        log_i = torch.einsum('m,mhw->hw',pupil,self.basis[1:])
        norm = torch.logsumexp(log_i[self.pupil_mask],0)-self.pupil_log_count
        return phase*torch.exp(.5*(log_i-norm))

    def psf(self,field,slit,view,first,last):
        value = self.aperture[view,None]*self.propagation[first:last]*field[None]
        value = value*self.ramp_y[view,first:last,:,None]*self.ramp_x[view,first:last,None,:]
        local = torch.fft.ifft2(value).index_select(-2,self.sy).index_select(-1,self.sx)
        ordinary = (local/self.energy).abs().square()
        cy = self.axis[None,:,None]+self.shifts[view,first:last,0,None,None]
        coordinate = cy*self.cfg['fine_pitch_um']+self.depths[first:last,None,None]*(.15*slit[2])
        width,detector = slit[0].exp(),slit.new_tensor(self.cfg['shutter_width_um'])
        support,plateau = (width+detector)/2,(width-detector).abs()/2
        gate = (F.relu(support-coordinate.abs())-F.relu(plateau-coordinate.abs()))/width
        return ordinary*gate

    def otf(self,field,slit,view,first,last):
        return torch.fft.rfft2(self.psf(field,slit,view,first,last),s=(self.n,self.n))*\
            self.otf_y[view,first:last,:,None]*self.otf_x[view,first:last,None,:]

    def finish(self,spectrum):
        image = torch.fft.irfft2(spectrum,s=(self.n,self.n))
        return F.grid_sample(image[None,None],self.grid,mode='bilinear',
            padding_mode='border',align_corners=False)[0,0]

    @torch.no_grad()
    def predict(self,volume,wave,pupil,slit):
        cache = ProjectionCache(self,volume,wave,pupil,slit)
        try:
            return torch.stack([cache(view) for view in range(len(self.views))])
        finally:
            cache.close()


class ProjectionCache:
    """Reuse volume FFTs and accumulate their exact gradients across views."""
    def __init__(self,model,volume,wave,pupil,slit):
        self.model,self.slit = model,slit
        self.field = model.pupil_field(wave,pupil)
        self.field_leaf = self.field.detach().requires_grad_(self.field.requires_grad)
        self.records = []
        first = 0
        for part in volume.split(DEPTH_CHUNK):
            spectrum = torch.fft.rfft2(part,s=(model.n,model.n))
            leaf = spectrum.detach().requires_grad_(spectrum.requires_grad)
            last = first+len(part)
            self.records.append((first,last,spectrum,leaf))
            first = last

    def __call__(self,view):
        total = None
        for first,last,_,leaf in self.records:
            part = (leaf*self.model.otf(self.field_leaf,self.slit,view,first,last)).sum(0)
            total = part if total is None else total+part
        return self.model.finish(total)

    def backward(self):
        outputs,gradients = [],[]
        for _,_,spectrum,leaf in self.records:
            if spectrum.requires_grad and leaf.grad is not None:
                outputs.append(spectrum)
                gradients.append(leaf.grad)
        if self.field.requires_grad and self.field_leaf.grad is not None:
            outputs.append(self.field)
            gradients.append(self.field_leaf.grad)
        torch.autograd.backward(outputs,gradients)

    def close(self):
        self.records.clear()
        self.field = self.field_leaf = None


def blur(data,sigma):
    if not sigma:
        return data
    radius = 3*sigma
    axis = torch.arange(-radius,radius+1,device=data.device,dtype=data.dtype)
    kernel = torch.exp(-axis.square()/(2*sigma*sigma))
    kernel /= kernel.sum()
    value = F.conv2d(F.pad(data[:,None],(radius,radius,0,0),mode='replicate'),
        kernel[None,None,None,:])
    return F.conv2d(F.pad(value,(0,0,radius,radius),mode='replicate'),
        kernel[None,None,:,None])[:,0]


@torch.no_grad()
def initialize_volume(model,data,wave,pupil,slit):
    n,nz = data.shape[-1],len(model.depths)
    chosen = [i for i,v in enumerate(model.views)
        if (v//model.scan-model.scan//2)**2+(v%model.scan-model.scan//2)**2 <= 9]
    descriptor = blur(data,1)-blur(data,6)
    descriptor /= descriptor.square().mean((1,2),keepdim=True).sqrt().clamp_min(1)
    descriptor.clamp_(-4,8)
    yy,xx = torch.meshgrid(torch.arange(n,device=data.device),
        torch.arange(n,device=data.device),indexing='ij')
    sums = data.new_zeros((nz,n,n))
    squares = torch.zeros_like(sums)
    field = model.pupil_field(wave,pupil)
    for view in chosen:
        centers = []
        for first in range(0,nz,DEPTH_CHUNK):
            last = min(nz,first+DEPTH_CHUNK)
            psf = model.psf(field,slit,view,first,last)
            mass = psf.sum((-2,-1)).clamp_min(1e-20)
            cy = (psf.sum(-1)*model.axis).sum(-1)/mass+model.shifts[view,first:last,0]
            cx = (psf.sum(-2)*model.axis).sum(-1)/mass+model.shifts[view,first:last,1]
            centers.append(torch.stack([cy,cx],-1)/(model.scan/3))
        centers = torch.cat(centers)
        qy,qx = yy[None]+centers[:,0,None,None],xx[None]+centers[:,1,None,None]
        grid = torch.stack([(2*qx+1)/n-1,(2*qy+1)/n-1],-1)
        source = descriptor[view][None,None].expand(nz,-1,-1,-1)
        aligned = F.grid_sample(source,grid,mode='bilinear',padding_mode='border',align_corners=False)[:,0]
        sums += aligned
        squares += aligned.square()
    coherence = ((sums.square()-squares)/((len(chosen)-1)*squares).clamp_min(1e-8)).clamp(0,1)
    score = (sums/len(chosen)).clamp_min(0).square()*coherence
    score = F.avg_pool2d(score[:,None],5,stride=1,padding=2)[:,0]
    score = torch.as_tensor(gaussian_filter1d(score.cpu().numpy(),3,axis=0,mode='nearest'),device=data.device)
    score /= score.mean().clamp_min(1e-8)
    seed = .3+.7*score.clamp(max=20)
    coordinate = torch.arange(model.n,device=data.device,dtype=data.dtype)*(3/model.scan)
    norm = (2*coordinate+1)/n-1
    gy,gx = torch.meshgrid(norm,norm,indexing='ij')
    grid = torch.stack([gx,gy],-1)[None]
    parts = [F.grid_sample(part[:,None],grid.expand(len(part),-1,-1,-1),
        mode='bilinear',padding_mode='border',align_corners=False)[:,0]
        for part in seed.split(DEPTH_CHUNK)]
    mean = sum(part.double().sum() for part in parts)/(nz*model.n**2)
    return torch.cat([(.5*part/mean.float()+.5).clamp_min(1e-30).log().exp() for part in parts])


# 3. Nonnegative object updates and Adam optical updates.
def data_loss(prediction,observed):
    mu,y = prediction.double().clamp_min(1e-6),observed.double().clamp_min(0)
    sy = y.clamp_min(1e-6)
    r = (mu-y)/sy
    near = y*(r-torch.log1p(r.clamp_min(-.5)))
    far = mu-y+y*(sy.log()-mu.log())
    return (2*torch.where((y>0)&(r.abs()<.5),near,far)).mean()


def spatial_penalty(volume):
    terms = []
    for axis in (1,2):
        curvature = torch.diff(volume,n=2,dim=axis)
        terms.append(F.smooth_l1_loss(curvature,torch.zeros_like(curvature)))
    return sum(terms)/2


def view_schedule(nviews):
    random = torch.Generator().manual_seed(SEED)
    schedule = []
    while len(schedule) < STEPS:
        schedule.extend(part.tolist() for part in
            torch.randperm(nviews,generator=random).split(VIEWS_PER_STEP))
    return schedule[:STEPS]


def reconstruct(model,data):
    counts = data.double().sum()
    if counts <= 0 or not torch.isfinite(data).all():
        raise ValueError('The light field must be finite and have positive total intensity.')
    wave = torch.nn.Parameter(data.new_zeros(36))
    pupil = torch.nn.Parameter(data.new_zeros(35))
    fixed_shutter = math.log(model.cfg['shutter_width_um'])
    slit = torch.nn.Parameter(data.new_tensor([
        math.log(model.cfg['initial_illumination_width_um']),fixed_shutter,0.]))
    mask = torch.ones_like(wave)
    mask[FIXED_MODES] = 0
    volume = torch.nn.Parameter(initialize_volume(model,data,wave,pupil,slit))
    active = [volume,wave,pupil,slit]
    with torch.no_grad():
        scale = (counts/model.predict(volume,wave*mask,pupil,slit).double().sum()).float()
        volume.mul_(scale)
    optics = torch.optim.Adam([dict(params=[wave],lr=LR_WAVEFRONT),
        dict(params=[pupil],lr=LR_PUPIL),dict(params=[slit],lr=LR_ILLUMINATION)],
        betas=(.9,.99),eps=1e-8,fused=data.device.type=='cuda')
    optimizer = torch.optim.SGD([volume],lr=RELAXATION,momentum=0)
    targets = {sigma:blur(data.clamp_min(0),sigma) for sigma in BLUR_SIGMAS}
    history = []

    @torch.no_grad()
    def bounds():
        volume.clamp_(min=0)
        wave.mul_(mask).clamp_(-4,4)
        pupil.clamp_(-1,1)
        low = slit.new_tensor([math.log(1),math.log(2),-model.tangent_limit/.15])
        high = slit.new_tensor([math.log(350),math.log(150),model.tangent_limit/.15])
        slit.copy_(slit.maximum(low).minimum(high))
        slit[1] = fixed_shutter

    for step,selected in enumerate(view_schedule(len(data))):
        sigma = BLUR_SIGMAS[min(len(BLUR_SIGMAS)-1,step//BLUR_STAGE_STEPS)]
        for parameter in active:
            parameter.grad = None
        cache = ProjectionCache(model,volume,wave*mask,pupil,slit)
        positive,fit = None,0.
        try:
            leaves = [r[3] for r in cache.records]
            for view in selected:
                prediction = blur(cache(view)[None],sigma)[0]
                linear = 2*prediction.mean()/len(selected)
                adjoints = torch.autograd.grad(linear,leaves,retain_graph=True)
                if positive is None:
                    positive = [g.detach().clone() for g in adjoints]
                else:
                    for total,gradient in zip(positive,adjoints):
                        total.add_(gradient)
                loss = data_loss(prediction,targets[sigma][view])/len(selected)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite data objective.')
                loss.backward()
                fit += float(loss.detach())
                del prediction,linear,loss,adjoints
            sensitivity, = torch.autograd.grad([r[2] for r in cache.records],volume,
                grad_outputs=positive,retain_graph=True)
            cache.backward()
        finally:
            cache.close()
        penalty = REGULARIZATION*spatial_penalty(volume)
        prior_gradient, = torch.autograd.grad(penalty,volume)
        volume.grad.add_(prior_gradient)
        prior = float(penalty.detach())
        del prior_gradient,penalty
        with torch.no_grad():
            if float(sensitivity.min()) < -max(1e-12,float(sensitivity.max())*1e-5):
                raise FloatingPointError('Negative object sensitivity.')
            volume.grad = volume.detach().clamp_min(1e-8)*volume.grad/sensitivity.clamp_min(1e-12)
            if not torch.isfinite(volume.grad).all():
                raise FloatingPointError('Nonfinite object update.')
        optimizer.step()
        bounds()
        optics.step()
        bounds()
        for parameter in active:
            parameter.grad = None
        del sensitivity,positive
        if (step+1)%100 == 0 or step+1 == STEPS:
            history.append(dict(step=step+1,sigma=sigma,data_loss=fit,prior=prior))
            print(f'{step+1:4d}/{STEPS}  sigma={sigma}  data={fit:.6g}',flush=True)
    return volume.detach(),(wave*mask).detach(),pupil.detach(),slit.detach(),history


def estimate_region(model,data):
    volume,wave,pupil,slit,history = reconstruct(model,data)
    with torch.no_grad():
        prediction = model.predict(volume,wave,pupil,slit)
        start,size = model.cfg['core_start'],model.cfg['core_size']
        native_start = round(start*3/model.scan)
        native_size = round(size*3/model.scan)
        phase = torch.einsum('m,mhw->hw',wave,model.basis)
        intensity = model.pupil_field(wave,pupil).abs().square()*model.pupil_mask
        return dict(volume=volume[:,start:start+size,start:start+size].contiguous().cpu().numpy(),
            light_field=prediction[:,native_start:native_start+native_size,
                native_start:native_start+native_size].contiguous().cpu().numpy(),
            phase=phase.cpu().numpy(),pupil=intensity.cpu().numpy(),
            parameters=dict(wavefront_waves=wave.cpu().tolist(),pupil_log_intensity=pupil.cpu().tolist(),
                illumination_width_um=float(slit[0].exp()),shutter_width_um=model.cfg['shutter_width_um'],
                illumination_tangent=float(.15*slit[2]),history=history))


# 4. Save the central crop and its fitted optical parameters.
def save_results(output,settings,result,observation,seconds):
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('volume',result['volume']),('reprojected_lf',result['light_field']),
            ('pupil_phase_waves',result['phase']),('pupil_intensity',result['pupil'])]:
        tifffile.imwrite(output/f'{name}.tiff',value,photometric='minisblack',compression='zlib')
    first,last = settings['lf_error_rows']
    observed = observation[:,first:last].astype('float64')
    residual = 100*np.linalg.norm(result['light_field'][:,first:last].astype('float64')-observed)/np.linalg.norm(observed)
    summary = dict(sample=settings['name'],settings=settings,steps=STEPS,seed=SEED,
        views_per_step=VIEWS_PER_STEP,relaxation=RELAXATION,regularization=REGULARIZATION,
        wavefront_learning_rate=LR_WAVEFRONT,pupil_learning_rate=LR_PUPIL,
        illumination_learning_rate=LR_ILLUMINATION,blur_sigmas=BLUR_SIGMAS,
        blur_stage_steps=BLUR_STAGE_STEPS,volume_axes='ZYX',light_field_axes='VYX',
        voxel_spacing_um=[settings['optics']['dz_um'],settings['fine_pitch_um'],settings['fine_pitch_um']],
        lf_relative_error_percent=float(residual),elapsed_s=seconds,
        optical_parameters=result['parameters'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'LF relative error: {residual:.6f}%\nResults: {output.relative_to(ROOT)}',flush=True)


def main(sample=SAMPLE):
    device = torch.device(DEVICE)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA-enabled PyTorch installation is required.')
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = ROOT/'data'/sample
    cfg = json.loads((folder/'settings.json').read_text())
    data = tifffile.imread(folder/'lf.tiff').astype('float32')
    size,context = cfg['core_size'],cfg['context_size']
    native_size = round(size*3/cfg['optics']['Nnum'])
    if data.shape != (len(AIR_VIEWS),context,context) or not np.isfinite(data).all():
        raise ValueError('LF shape must match the configured views and reconstruction context.')
    started = time.perf_counter()
    model = Projection(cfg,device)
    print(cfg['name'],flush=True)
    result = estimate_region(model,torch.as_tensor(data,device=device))
    y,x = cfg['roi_yx']
    observation = data[:,y:y+native_size,x:x+native_size]
    save_results(ROOT/'outputs'/sample,cfg,result,observation,time.perf_counter()-started)


if __name__ == '__main__':
    main()
