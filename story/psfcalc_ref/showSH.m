addpath(genpath('./'))
t = tiledlayout(5,9,'TileSpacing','Compact');

aber_phase_size = 289;

data(1,:)=linspace(-1,1,aber_phase_size);
data(2,:)=linspace(-1,1,aber_phase_size);
% load(['../01calcPhase/mat//Zernindex_',num2str(aberration),'_group',num2str(group),'.mat'],'c1');
% load(['../01calcPhase/mat_rms',num2str(aberration),'/Zernindex_',num2str(aberration),'_group',num2str(group),'.mat'],'c1');
aberration = 1;
for ii = 1:45
    c1 = zeros(1,45);c1(ii) = 1;
    aber_phase=SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
    c1 = c1 ./ calcRMS(aber_phase) .* 2*pi * aberration;
    aber_phase2 = SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
    nexttile(t)
    imshow(aber_phase2,[-4*pi,4*pi]);colormap(jet);

end

