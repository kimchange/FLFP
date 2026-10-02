
% The Code is created based on the method described in the following paper 
%   [1]  JIAMIN WU, ZHI LU, DONG JIANG and YUDUO GUO.etc,
%        3D observation of large-scale subcellular dynamics in vivo at the millisecond scale
%        in BioRxiv, 2019. 
%   [2] Robert Prevede, Young-Gyu Yoon, Maximilian Hoffmann, Nikita Pak.etc. 
%       "Simultaneous whole-animal 3D imaging of neuronal activity using light-field microscopy "   
%       in Nature Methods VOL.11 NO.7|July 2014.
% The Code is modified and extended from Robert's code
% 
%    Contact: ZHI LU (luz18@mails.tsinghua.edu.cn)
%    Date  : 10/24/2020

clear;
tic
addpath(genpath('./'))
%%%%%%%%%%%%%%%%%%%%%%% SIM PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%
NA =        1.4; %% numerical aperture of objective
MLPitch =   100*1e-6; %% pitch of the microlens pitch
Nnum =      13; %% number of virtual pixels
OSR =       3; %% spatial oversampling ratio for computing PSF
n =         1.515; %% refractive index
fml =       2100e-6; %% focal length of the microlens pitch
lambda =    525*1e-9; %% wavelength

% M =         63; %% magnification of the objective lens
% zmax =      12*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -12*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  0.24*1e-6; %% spacing between adjacent z-planes

% zmax =      20*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -20*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  0.4*1e-6; %% spacing between adjacent z-planes
% M =         63;

zmax =      10*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
zmin =      -10*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
zspacing =  0.2*1e-6; %% spacing between adjacent z-planes
M =         72.5;

eqtol = 1e-10;
k = 2*pi*n/lambda; %% k
k0 = 2*pi*1/lambda; %% k
d = fml;   %% optical distance between the microlens and the sensor
ftl = 165e-3;        %% focal length of tube lens
fobj = ftl/M;  %% focal length of objective lens
fnum_obj = M/(2*NA); %% f-number of objective lens (imaging-side)
fnum_ml = fml/MLPitch; %% f-number of micrl lens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% DEFINE OBJECT SPACE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if mod(Nnum,2)==0,
   error(['Nnum should be an odd number']); 
end
pixelPitch = MLPitch/Nnum; %% pitch of virtual pixels

x1objspace = [0]; 
x2objspace = [0];
x3objspace = [zmin:zspacing:zmax]+1e-9; % offset
objspace = ones(length(x1objspace),length(x2objspace),length(x3objspace));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p3max = max(abs(x3objspace));
x1testspace = (pixelPitch/OSR)* [0:1: Nnum*OSR*60];
x2testspace = [0];   
[psfLine] = calcPSFFT(p3max, fobj, NA, x1testspace, pixelPitch/OSR, lambda, d, M, n);
outArea = find(psfLine<0.1);
if isempty(outArea),
   error('Estimated PSF size exceeds the limit');   
end
IMGSIZE_REF = ceil(outArea(1)/(OSR*Nnum));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% OTHER SIMULATION PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%
disp(['Size of PSF ~= ' num2str(IMGSIZE_REF) ' [microlens pitch]' ]);
IMG_HALFWIDTH = max( Nnum*(IMGSIZE_REF + 1), 2*Nnum);
disp(['Size of IMAGE = ' num2str(IMG_HALFWIDTH*2*OSR+1) 'X' num2str(IMG_HALFWIDTH*2*OSR+1) '' ]);
x1space = (pixelPitch/OSR)*[-IMG_HALFWIDTH*OSR:1:IMG_HALFWIDTH*OSR]; 
x2space = (pixelPitch/OSR)*[-IMG_HALFWIDTH*OSR:1:IMG_HALFWIDTH*OSR]; 
x1length = length(x1space);
x2length = length(x2space);

x1MLspace = (pixelPitch/OSR)* [-(Nnum*OSR-1)/2 : 1 : (Nnum*OSR-1)/2];
x2MLspace = (pixelPitch/OSR)* [-(Nnum*OSR-1)/2 : 1 : (Nnum*OSR-1)/2];
x1MLdist = length(x1MLspace);
x2MLdist = length(x2MLspace);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%% FIND NON-ZERO POINTS %%%%%%%%%%%%%%%%%%%%%%%%%%
validpts = find(objspace>eqtol);
numpts = length(validpts);
[p1indALL p2indALL p3indALL] = ind2sub( size(objspace), validpts);
p1ALL = x1objspace(p1indALL)';
p2ALL = x2objspace(p2indALL)';
p3ALL = x3objspace(p3indALL)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% DEFINE ML ARRAY %%%%%%%%%%%%%%%%%%%%%%%%% 
MLARRAY = calcML(fml, k0, x1MLspace, x2MLspace, x1space, x2space,0); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%% Alocate Memory for storing PSFs %%%%%%%%%%%   
LFpsfWAVE_STACK = zeros(x1length, x2length, numpts);
psfWAVE_STACK = zeros(x1length, x2length, numpts);
disp(['Start Calculating PSF...']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%% PROJECTION FROM SINGLE POINT %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
centerPT = ceil(length(x1space)/2);
halfWidth =  Nnum*(IMGSIZE_REF + 0 )*OSR;
centerArea = (  max((centerPT - halfWidth),1)   :   min((centerPT + halfWidth),length(x1space))     );

disp(['Computing PSFs (1/3)']);
for eachpt=1:numpts,        
    p1 = p1ALL(eachpt);
    p2 = p2ALL(eachpt);
    p3 = p3ALL(eachpt);
    
    IMGSIZE_REF_IL = ceil(IMGSIZE_REF*( abs(p3)/p3max));
    halfWidth_IL =  max(Nnum*(IMGSIZE_REF_IL + 0 )*OSR, 2*Nnum*OSR);
    centerArea_IL = (  max((centerPT - halfWidth_IL),1)   :   min((centerPT + halfWidth_IL),length(x1space))     );
    disp(['size of center area = ' num2str(length(centerArea_IL)) 'X' num2str(length(centerArea_IL)) ]);    
    
    %% excute PSF computing funcion
    % [psfWAVE LFpsfWAVE] = calcPSF(p1, p2, p3, fobj, NA, x1space, x2space, pixelPitch/OSR, lambda, MLARRAY, d, M, n,  centerArea_IL);
    psfWAVE = calcPSF(p1, p2, p3, fobj, NA, x1space, x2space, pixelPitch/OSR, lambda, MLARRAY, d, M, n,  centerArea_IL);
    psfWAVE_STACK(:,:,eachpt)  = psfWAVE;
    % LFpsfWAVE_STACK(:,:,eachpt)= LFpsfWAVE;    

end 
% load('psfWAVE_STACK_liuyv.mat');
wf_psf=gather(imresize(abs(psfWAVE_STACK).^2,[floor(size(psfWAVE_STACK,1)/OSR/2)*2+1,floor(size(psfWAVE_STACK,2)/OSR/2)*2+1]));
for zlayer=1:size(wf_psf,3)
    wf_psf(:,:,zlayer)=wf_psf(:,:,zlayer)./sum(sum(sum(wf_psf(:,:,zlayer))));
end
% confocal_psf=wf_psf.^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
mkdir('PSF')
disp(['Saving PSF matrix file...']);
prefix = ['./PSF/Ideal_wfPSF_M',num2str(M),'_NA',num2str(NA),'_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_zspacing',num2str(zspacing*1e+6),'u'];
save([prefix,'.mat'],'wf_psf','-v7.3');
disp(['PSF computation complete.']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
bitdepth = 16;
psftif = wf_psf / max(wf_psf(:));
psftif = psftif * (2^bitdepth - 1);
% imwrite3d(res,'H.tif',bitdepth);
% imwrite3d(res,'psf.tif',bitdepth);
% imwrite3d(res,'psf_sai.tif',bitdepth);
imwrite3d(psftif,[prefix,'.tif'],bitdepth);
toc