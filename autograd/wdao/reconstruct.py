"""Fixed-wavefront object fit with the same initialization and Adam schedule.

Uses demo_wavefront's optics, noise objective, prior, and configuration. The
staged estimate is installed only after the common zero-wavefront flux scale.
"""
import torch


def reconstruct_fixed(project, data, fixed_wavefront, config, *, steps=None):
    """Fit the object to signed raw observations; never update fixed_wavefront.

    config is demo_wavefront or an object with the same configuration/functions.
    A shorter steps value executes a prefix of the full learning-rate schedule.
    No clean LF, true object, or true wavefront enters this function.
    """
    steps = config.STEPS if steps is None else steps
    if not 0 < steps <= config.STEPS:
        raise ValueError('steps must be a positive prefix of the configured full run')
    device, nviews = data.device, data.shape[1]
    log_signal = torch.nn.Parameter(data.new_zeros((1, *config.VOLUME_SHAPE)))
    log_background = torch.nn.Parameter(data.new_zeros(()))
    wf = torch.nn.Parameter(data.new_zeros(config.MODES), requires_grad=False)
    if fixed_wavefront.shape != wf.shape or not torch.isfinite(fixed_wavefront).all():
        raise ValueError('fixed_wavefront must contain MODES finite coefficients')
    if torch.any(fixed_wavefront[config.FIXED_MODES] != 0):
        raise ValueError('Fixed gauge coefficients must be zero')
    mask = torch.ones_like(wf)
    mask[config.FIXED_MODES] = 0
    views = torch.arange(nviews, device=device)
    batch_size = min(config.VIEWS_PER_STEP, nviews)
    counts = (data.double()-config.BIAS).sum()
    if counts <= 0:
        raise ValueError('The total bias-subtracted signal must be positive')
    with torch.no_grad():
        flux = sum(project(log_signal.exp(), wf, v).double().sum()
                   for v in views.split(batch_size))
        scale = (counts/flux).float()
        wf.copy_(fixed_wavefront)
    optimizer = torch.optim.Adam([
        {'params': [log_signal], 'lr': config.LR_VOLUME},
        {'params': [wf], 'lr': config.LR_WAVEFRONT},
        {'params': [log_background], 'lr': config.LR_BACKGROUND}],
        betas=(.9, .99), eps=1e-12)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.01,
                                         total_iters=config.WARMUP),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   config.STEPS-config.WARMUP)],
        milestones=[config.WARMUP])
    random = torch.Generator(device=device).manual_seed(0)
    prior_random = torch.Generator(device=device).manual_seed(1)
    batches_per_epoch = (nviews+batch_size-1)//batch_size
    history = []
    for step in range(steps):
        if step % batches_per_epoch == 0:
            batches = iter(views[torch.randperm(nviews, generator=random,
                               device=device)].split(batch_size))
        selected = next(batches)
        optimizer.zero_grad(set_to_none=True)
        volume = scale*(log_signal.exp()+log_background.exp())*.5
        rate = project(volume, wf*mask, selected)
        fit = config.data_loss(rate, data[:, selected])/counts*nviews/len(selected)
        z = torch.randint(config.VOLUME_SHAPE[0], (config.PRIOR_DEPTHS,),
                          generator=prior_random, device=device)
        prior = config.spatial_penalty(volume[:, z]/scale)
        loss = fit+config.REGULARIZATION*prior
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(torch.stack((fit.detach(), prior.detach(), loss.detach())))
        if step == 0 or (step+1) % 400 == 0 or step+1 == steps:
            print(f'Fixed WF {step+1:4d}/{steps}  data={fit.item():.6g}  '
                  f'prior={prior.item():.6g}', flush=True)
    with torch.no_grad():
        volume = scale*(log_signal.exp()+log_background.exp())*.5
    if not torch.equal(wf, fixed_wavefront.to(wf)):
        raise RuntimeError('The fixed staged wavefront changed')
    return volume.detach(), wf.detach(), dict(initial_scale=float(scale),
        training_signal_sum=float(counts), zero_wf_unit_training_flux=float(flux),
        history_columns=['data_loss', 'prior', 'loss'],
        history=torch.stack(history).cpu().tolist())
