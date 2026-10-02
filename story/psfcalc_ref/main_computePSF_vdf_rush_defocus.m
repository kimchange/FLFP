
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

% clear;
tic
% parpool(13);
addpath(genpath('./'))
addpath('../01calcPhase/')
% addpath('/mnt/kimchange/psfAnalyze/utils/');
% addpath('/home/slfm/as13000/kimchange/psfcalc/utils/');


% NA =        1.4; %% numerical aperture of objective
% MLPitch =   100*1e-6; %% pitch of the microlens pitch
% Nnum =      13; %% number of virtual pixels
% OSR =       3; %% spatial oversampling ratio for computing PSF
% n =         1.515; %% refractive index
% fml =       2100e-6; %% focal length of the microlens pitch
% lambda =    525*1e-9; %% wavelength
% M=63;



%%%%%%%%%%%%%%%%%%%%%%% SIM PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%
% NA =        0.5; %% numerical aperture of objective
% MLPitch =   100*1e-6; %% pitch of the microlens pitch
% Nnum =      13; %% number of virtual pixels
% OSR =       3; %% spatial oversampling ratio for computing PSF
% % n =         1.515; %% refractive index
% n =         1; %% refractive index
% fml =       2100e-6; %% focal length of the microlens pitch
% lambda =    525*1e-9; %% wavelength


% for aberration = [0.4:0.2:3]  %% =1 means PSF with aberration; =0 means ideal PSF
% for group = [2:5]
% mkdir(['./PSF/group',num2str(group)])
% for aberration = [0.4:0.4:2]  %% =1 means PSF with aberration; =0 means ideal PSF
% disp(['now aberration ',num2str(aberration)])


% M =         63; %% magnification of the objective lens
% zmax =      12*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -12*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  0.24*1e-6; %% spacing between adjacent z-planes

% zmax =      75*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -75*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  1.5*1e-6; %% spacing between adjacent z-planes
% M =         22.56;

% zmax =      50*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -50*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  1*1e-6; %% spacing between adjacent z-planes
% M =         20;

% zmax =      14.8*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -14.8*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  0.2*1e-6; %% spacing between adjacent z-planes

M =         20;
NA =        1.05; %% numerical aperture of objective
MLPitch =   56.4e-6; %% pitch of the microlens pitch
Nnum =      15; %% number of virtual pixels
OSR =       3; %% spatial oversampling ratio for computing PSF
n =         1.406; %% refractive index
fml =       536.4e-6; %% focal length of the microlens pitch
lambda =    525*1e-9; %% wavelength

zmax =      30*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
zmin =      -30*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
zspacing =  0.6*1e-6; %% spacing between adjacent z-planes
zmax =      50*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
zmin =      -50*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
zspacing =  1*1e-6; %% spacing between adjacent z-planes
ftl = 180e-3; 
% M =         20;
% NA =        0.5; %% numerical aperture of objective
% MLPitch =   97.5*1e-6; %% pitch of the microlens pitch
% Nnum =      15; %% number of virtual pixels
% OSR =       3; %% spatial oversampling ratio for computing PSF
% n =         1; %% refractive index
% fml =       2100e-6; %% focal length of the microlens pitch
% lambda =    525*1e-9; %% wavelength


% % M = 58.9875;
% M = 72.5;
% zmax =      15.*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -15.*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  0.3*1e-6; %% spacing between adjacent z-planes
% zmax =      0*1e-6; %% the axial location of the highest z-plane with respect to the focal plane
% zmin =      -0*1e-6; %% the axial location of the lowest z-plane with respect to the focal plane
% zspacing =  0.2*1e-6; %% spacing between adjacent z-planes
% M =         63;

is_circle_aperture = 1;debye_integral = 0;
eqtol = 1e-10;
k = 2*pi*n/lambda; %% k
k0 = 2*pi*1/lambda; %% k
d = fml;   %% optical distance between the microlens and the sensor
% ftl = 165e-3;        %% focal length of tube lens
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
x3objspace = [zmin:zspacing:zmax] + eps; % offset
objspace = ones(length(x1objspace),length(x2objspace),length(x3objspace));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p3max = max(abs(x3objspace));
x1testspace = (pixelPitch/OSR)* [0:1: Nnum*OSR*60];
x2testspace = [0];   
[psfLine] = calcPSFFT(p3max, fobj, NA, x1testspace, pixelPitch/OSR, lambda, d, M, n);
outArea = find(psfLine<0.05);
% input('debugkimchange')
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
MLARRAY = calcML(fml, k0, x1MLspace, x2MLspace, x1space, x2space, is_circle_aperture); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%% Alocate Memory for storing PSFs %%%%%%%%%%%   
% LFpsfWAVE_STACK = zeros(x1length, x2length, numpts);
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


IMGSIZE_REF_IL = ceil(IMGSIZE_REF*( abs(0)/p3max));
halfWidth_IL =  max(Nnum*(IMGSIZE_REF_IL + 0 )*OSR, 2*Nnum*OSR);
centerArea_IL = (  max((centerPT - halfWidth_IL),1)   :   min((centerPT + halfWidth_IL),length(x1space))     );
disp(['size of center area = ' num2str(length(centerArea_IL)) 'X' num2str(length(centerArea_IL)) ]);  
psfWAVE_focus = calcPSF(0, 0, eps, fobj, NA, x1space, x2space, pixelPitch/OSR, lambda, MLARRAY, d, M, n,  centerArea_IL);

% imshow(fftshift(fft2(psfWAVE_focus)) ,[])



cutoff_freq = 2*NA/lambda;
fft_top_freq =  1/(pixelPitch/M/OSR);
aber_phase_size = cutoff_freq / fft_top_freq * size(psfWAVE_STACK,1);
aber_phase_size = floor(aber_phase_size/2)*2+1;
data=zeros(2,aber_phase_size);  %% 289 is the diameter of the aperture in the image of abs(fftshift(fft2(psfWAVE))
data(1,:)=linspace(-1,1,aber_phase_size);
data(2,:)=linspace(-1,1,aber_phase_size);

aperture_mask = zeros(size(psfWAVE_focus));
[hh,ww] = ndgrid( - (size(psfWAVE_focus,1)-1)/2  :  (size(psfWAVE_focus,1)-1)/2, - (size(psfWAVE_focus,2)-1)/2  :  (size(psfWAVE_focus,2)-1)/2 );
aperture_mask(hh.^2 + ww.^2 <= floor(aber_phase_size/2).^2 ) = 1;



dx0 = pixelPitch/OSR;
Nx = size(psfWAVE_focus,1);
Ny = size(psfWAVE_focus,2);

du = 1./(Nx*dx0);
% u = [0:ceil(Nx/2)-1 ceil(-Nx/2):-1]*du; 
dv = 1./(Ny*dx0);
% v = [0:ceil(Ny/2)-1 ceil(-Ny/2):-1]*dv; 
% (repmat(u',1,length(v)).^2+repmat(v,length(u),1).^2)

% load(['../01calcPhase/mat//Zernindex_',num2str(aberration),'_group',num2str(group),'.mat'],'c1');
% load(['../01calcPhase/mat_rms',num2str(aberration),'/Zernindex_',num2str(aberration),'_group',num2str(group),'.mat'],'c1');

% aber_phase=SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
% c1 = c1 ./ calcRMS(aber_phase) .* 2*pi * aberration;
% aber_phase2 = SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
% aber_phase2=padarray( aber_phase2,[(size(psfWAVE_STACK,1)-size(aber_phase,1))/2,(size(psfWAVE_STACK,2)-size(aber_phase,2))/2] );

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
    if debye_integral == 1
        psfWAVE = calcPSF(p1, p2, p3, fobj, NA, x1space, x2space, pixelPitch/OSR, lambda, MLARRAY, d, M, n,  centerArea_IL);
    else
        if abs(p3) <= 1e-9
            psfWAVE = psfWAVE_focus;
        else
            % pupilWAVE=ones(pixelSize_OSR,pixelSize_OSR).*aperture_mask;
            % pupilWAVE = gpuArray(single(pupilWAVE));
            % tempP = k0*n*p3*realsqrt((1-(fxcoor.*lambda./n.*M).^2-(fycoor.*lambda./n.*M).^2).*aperture_mask);
            tempP = k0*n*p3*realsqrt( (1 - (du .* hh .* lambda ./n.*M ).^2-(dv .* ww .* lambda ./n.*M ).^2) .* aperture_mask  );

            % tempP_ = tempP( (size(tempP,1)+1) / 2 - floor(aber_phase_size/2) : (size(tempP,1)+1) / 2 + floor(aber_phase_size/2)  ,  (size(tempP,2)+1) / 2 - floor(aber_phase_size/2) : (size(tempP,2)+1) / 2 + floor(aber_phase_size/2)  );
            % defocus_level = calcRMS(tempP_)
            % c1 = zeros(1,45);c1(4) = 1;
            % defocus_phase = SH(c1,data);
            % c1 = c1 ./ calcRMS(defocus_phase) .* defocus_level;
            % defocus_phase = SH(c1,data);
            % calcRMS(defocus_phase)
            % defocus_phase = padarray( defocus_phase, [(size(psfWAVE_focus,1)-size(defocus_phase,1))/2,(size(psfWAVE_focus,2)-size(defocus_phase,2))/2] );

            
            tempP = gpuArray(single(tempP));
            psfWAVE = ifft2(ifftshift(fftshift(fft2(psfWAVE_focus)).*exp(1j.*tempP)));



            % psfWAVE=fftshift(ifft2(ifftshift(squeeze(psfWAVE_fdomain))));
            % c1 = zeros(1,45);c1(4) = 1;
            % defocus_phase = SH(c1,data);
            % c1 = c1 ./ calcRMS(aber_phase) .* 2*pi * aberration;
            % defocus_phase = SH(c1,data);
            % psfWAVE = ifft2(ifftshift(fftshift(fft2(psfWAVE_focus)).*exp(1j.*defocus_phase)));
        end

    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % if aberration>0
    %     data=zeros(2,289);  %% 289 is the diameter of the aperture in the image of abs(fftshift(fft2(psfWAVE))
    %     data(1,:)=linspace(-1,1,289);
    %     data(2,:)=linspace(-1,1,289);
    %     load(['01calcPhase/group',num2str(group),'/Zernindex_',num2str(aberration),'.mat'],'c1');
    %     aber_phase=SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
    %     aber_phase2=padarray( aber_phase,[(size(psfWAVE,1)-size(aber_phase,1))/2,(size(psfWAVE,2)-size(aber_phase,2))/2] );
    % else
    %     aber_phase2=zeros(size(psfWAVE));
    % end
    % psfWAVE_STACK(:,:,eachpt) = ifft2(ifftshift(fftshift(fft2(psfWAVE)).*exp(1j.*aber_phase2)));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    psfWAVE_STACK(:,:,eachpt)  = psfWAVE;
    % LFpsfWAVE_STACK(:,:,eachpt)= LFpsfWAVE;    

end 

% disp(['debugkimchange'])
% pause


save(['debye_integral_',num2str(debye_integral),'psfWAVE_STACK_M',num2str(M),'_NA',num2str(NA),'_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_zspacing',num2str(zspacing*1e+6),'u.mat'],'psfWAVE_STACK','-V7.3');
% disp(['debugkimchange'])
% pause
% save(['psfWAVE_STACK_M',num2str(M),'_NA',num2str(NA),'_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_zspacing',num2str(zspacing*1e+6),'u.mat'],'psfWAVE_STACK','-V7.3');
% disp(['loading cache'])
% load(['psfWAVE_STACK_M',num2str(M),'_NA',num2str(NA),'_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_zspacing',num2str(zspacing*1e+6),'u.mat']);

% for group = [2:5]
% for group = [11:20]
gpuDevice(2)
for aberration = [0]%[1]
for group = [0]%[1:10]
% mkdir(['./PSF/group',num2str(group)])
% for aberration = [0.4:0.4:2]  


disp(['now aberration ',num2str(aberration)])


if aberration>0
    cutoff_freq = 2*NA/lambda;
    fft_top_freq =  1/(pixelPitch/M/OSR);
    aber_phase_size = cutoff_freq / fft_top_freq * size(psfWAVE_STACK,1);
    aber_phase_size = floor(aber_phase_size/2)*2+1;
    data=zeros(2,aber_phase_size);  %% 289 is the diameter of the aperture in the image of abs(fftshift(fft2(psfWAVE))
    data(1,:)=linspace(-1,1,aber_phase_size);
    data(2,:)=linspace(-1,1,aber_phase_size);
    % load(['../01calcPhase/mat//Zernindex_',num2str(aberration),'_group',num2str(group),'.mat'],'c1');
    % load(['../01calcPhase/mat_rms',num2str(aberration),'/Zernindex_',num2str(aberration),'_group',num2str(group),'.mat'],'c1');
    c1 = zeros(1,45);c1(11) = 1;
    aber_phase=SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
    c1 = c1 ./ calcRMS(aber_phase) .* 2*pi * aberration;
    aber_phase2 = SH(c1,data); % c0 is the experimentally measured Zernike cofficients 
    aber_phase2=padarray( aber_phase2,[(size(psfWAVE_STACK,1)-size(aber_phase,1))/2,(size(psfWAVE_STACK,2)-size(aber_phase,2))/2] );
else
    aber_phase2=zeros([size(psfWAVE_STACK,1),size(psfWAVE_STACK,2)]);
end
% psfWAVE_STACK_aber = ifft2(ifftshift(fftshift(fft2(psfWAVE_STACK)).*exp(1j.*aber_phase2)));


cutoff_freq = 2*NA/lambda;
fft_top_freq =  1/(pixelPitch/M/OSR);
aber_phase_size = cutoff_freq / fft_top_freq * size(psfWAVE_STACK,1);
aber_phase_size = floor(aber_phase_size/2)*2+1;
pupilMask = 1;
mask_size =  aber_phase_size * pupilMask;
mask_size = floor(mask_size/2)*2+1;
[ii, jj] = ndgrid(-floor(mask_size/2):floor(mask_size/2), -floor(mask_size/2):floor(mask_size/2) );

mask_phase = zeros(mask_size, mask_size, 'single');
mask_phase(ii.^2 + jj.^2 <= floor(mask_size/2).^2) = 1;
mask_phase=padarray( mask_phase,[(size(psfWAVE_STACK,1)-size(mask_phase,1))/2,(size(psfWAVE_STACK,2)-size(mask_phase,2))/2] );


% psfWAVE_STACK_aber = ifft2(ifftshift(fftshift(fft2(psfWAVE_STACK)).*exp(1j.*aber_phase2))) .* mask_phase;
psfWAVE_STACK_aber = ifft2(ifftshift(fftshift(fft2(psfWAVE_STACK)).*exp(1j.*aber_phase2  ) .* mask_phase  )) ;
% load('psfWAVE_STACK.mat')

% disp('debug 0525')
% pause
% psfWAVE_STACK = gpuArray(single(psfWAVE_STACK));

psfWAVE_STACK_aber = gpuArray(single(psfWAVE_STACK_aber));
MLARRAY = gpuArray(MLARRAY);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%% Compute Light Field PSFs (light field) %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
x1objspace = (pixelPitch/M)*[-floor(Nnum/2):1:floor(Nnum/2)];
x2objspace = (pixelPitch/M)*[-floor(Nnum/2):1:floor(Nnum/2)];
XREF = ceil(length(x1objspace)/2);
YREF = ceil(length(x1objspace)/2);
CP = ( (centerPT-1)/OSR+1 - halfWidth/OSR :1: (centerPT-1)/OSR+1 + halfWidth/OSR  );
% H = gpuArray.zeros( length(CP), length(CP), length(x1objspace), length(x2objspace), length(x3objspace) ,'single');
H = zeros( length(CP), length(CP), length(x1objspace), length(x2objspace), length(x3objspace) );
disp(['Computing LF PSFs (2/3)']);
% for i=1:length(x1objspace)*length(x2objspace)*length(x3objspace) ,
% delete(gcp('nocreate'));
for a = 1:length(x1objspace)
    for b = 1:length(x2objspace)
        for c = 1:length(x3objspace)
            % [a, b, c] = ind2sub([length(x1objspace) length(x2objspace) length(x3objspace)], i);  
            psfREF = psfWAVE_STACK_aber(:,:,c);  
            
            psfSHIFT= im_shiftn(psfREF, [OSR*(a-XREF), OSR*(b-YREF)] );
            [f1,dx1,x1]=fresnel2D(psfSHIFT.*MLARRAY, pixelPitch/OSR, d,lambda);
            % f1= im_shift2(f1, -OSR*(a-XREF), -OSR*(b-YREF) );
            
            xmin =  max( centerPT  - halfWidth, 1);
            xmax =  min( centerPT  + halfWidth, size(f1,1) );
            ymin =  max( centerPT  - halfWidth, 1);
            ymax =  min( centerPT  + halfWidth, size(f1,2) );

            % f1_AP = gpuArray.zeros(size(f1),'single');
            f1_AP = zeros(size(f1),'single');
            f1_AP( (xmin:xmax), (ymin:ymax) ) = f1( (xmin:xmax), (ymin:ymax) );
            f1_AP = f1_AP ./ max(f1_AP(:));
            [f1_AP_resize, x1shift, x2shift] = pixelBinning(abs(f1_AP.^2), OSR);           
            f1_CP = f1_AP_resize( CP - x1shift, CP-x2shift );

            H(:,:,a,b,c) = gather(f1_CP ./ sum(f1_CP(:)));
        end
    end
end
% toc


% for aa=1:size(H,3)
%     for multiWDF=1:size(H,4)
%         for kk=1:size(H,5)
%             temp=H(:,:,aa,multiWDF,kk);
%             H(:,:,aa,multiWDF,kk)= H(:,:,aa,multiWDF,kk)./sum(temp(:));
%         end
%     end
% end

% x1space = (pixelPitch/1)*[-IMG_HALFWIDTH*1:1:IMG_HALFWIDTH*1];
% x2space = (pixelPitch/1)*[-IMG_HALFWIDTH*1:1:IMG_HALFWIDTH*1]; 
% x1space = x1space(CP);
% x2space = x2space(CP);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%% Clear variables that are no longer necessary %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear psfWAVE_STACK;
clear psfWAVE_STACK_aber;
% clear LFpsfWAVE_STACK;
clear LFpsfWAVE_VIEW;
clear psfWAVE_VIEW;
clear LFpsfWAVE;
clear PSF_AP;
clear PSF_AP_resize;
clear PSF_CP;
clear f1;
clear f1_AP;
clear f1_AP_resize;
clear f1_CP;
clear psfREF;
clear psfSHIFT;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol = 0.005;
for i=1:size(H,5),
   H4Dslice = H(:,:,:,:,i);
   H4Dslice(find(H4Dslice< (tol*max(H4Dslice(:))) )) = 0;
   H(:,:,:,:,i) = H4Dslice;   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%% Estimate PSF size again  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
centerCP = ceil(length(CP)/2);
CAindex = zeros(length(x3objspace),2);
for i=1:length(x3objspace),
    IMGSIZE_REF_IL = ceil(IMGSIZE_REF*( abs(x3objspace(i))/p3max));
    halfWidth_IL =  max(Nnum*(IMGSIZE_REF_IL + 0 ), 2*Nnum);
    CAindex(i,1) = max( centerCP - halfWidth_IL , 1);
    CAindex(i,2) = min( centerCP + halfWidth_IL , size(H,1));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
disp(['Computing LF PSFs (3/3)']);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %%%%%%%%%%%%%%%% Convert to phase-space PSF  %%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% IMGsize=size(H,1)-mod((size(H,1)-Nnum),2*Nnum);
% psf=zeros(IMGsize,IMGsize,Nnum,Nnum,size(H,5)); %% phase-space PSF
% for z=1:size(H,5)
 
%     sLF=zeros(IMGsize,IMGsize,Nnum,Nnum); %% scanning light field images
%     index1=round(size(H,1)/2)-fix(size(sLF,1)/2);
%     index2=round(size(H,1)/2)+fix(size(sLF,1)/2);
%     for ii=1:size(H,3)
%         for jj=1:size(H,4)
%             sLF(:,:,ii,jj)=im_shift3(squeeze(H(index1:index2,index1:index2,ii,jj,z)),ii-((Nnum+1)/2), jj-(Nnum+1)/2);
%         end
%     end
    
%     multiWDF=zeros(Nnum,Nnum,size(sLF,1)/size(H,3),size(sLF,2)/size(H,4),Nnum,Nnum); %% multiplexed phase-space
%     for i=1:size(H,3)
%         for j=1:size(H,4)
%             for a=1:size(sLF,1)/size(H,3)
%                 for b=1:size(sLF,2)/size(H,4)
%                     multiWDF(i,j,a,b,:,:)=squeeze(  sLF(  (a-1)*Nnum+i,(b-1)*Nnum+j,:,:  )  );
%                 end
%             end
%         end
%     end
%     WDF=zeros(  size(sLF,1),size(sLF,2),Nnum,Nnum  ); %% multiplexed phase-space
%     for a=1:size(sLF,1)/size(H,3)
%         for c=1:Nnum
%             x=Nnum*a+1-c;
%             for b=1:size(sLF,2)/size(H,4)
%                 for d=1:Nnum
%                     y=Nnum*b+1-d;
%                     WDF(x,y,:,:)=squeeze(multiWDF(:,:,a,b,c,d));
%                 end
%             end
%         end
%     end
%     psf(:,:,:,:,z)=WDF;
    
% end
% psf=single(psf);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%% Convert to phase-space PSF  %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
IMGsize=size(H,1)-mod((size(H,1)-Nnum),2*Nnum);
psf=zeros(IMGsize,IMGsize,Nnum,Nnum,size(H,5),'single'); %% phase-space PSF

index1=round(size(H,1)/2)-fix(size(psf,1)/2);
index2=round(size(H,1)/2)+fix(size(psf,1)/2);

H = single(H);

for ii=1:size(H,3)
    for jj=1:size(H,4)
        % psf(:,:,ii,jj,:)=im_shift2(squeeze(H(index1:index2,index1:index2,ii,jj,:)),ii-((Nnum+1)/2), jj-(Nnum+1)/2);
        % psf(:,:,ii,jj,:)=im_shift2(squeeze(H(index1:index2,index1:index2,ii,jj,:)),0, 0);
        psf(:,:,ii,jj,:)=squeeze(H(index1:index2,index1:index2,ii,jj,:));
    end
end
% psf=single(psf);
disp(['reshape'])
psf = reshape(psf,[Nnum,size(psf,1)/size(H,3),Nnum,size(psf,2)/size(H,4),Nnum,Nnum, size(H,5)]);
clear H;
psf = permute(psf,[5,2,6,4,1,3,7]);
psf = flip(psf,1);
psf = flip(psf,3);
% psf = flip(psf,2);
% psf = flip(psf,4);
psf = reshape(psf,[IMGsize,IMGsize,Nnum,Nnum,length(x3objspace)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
mkdir('PSF')
disp(['Saving PSF matrix file...']);
% prefix = ['./PSF/debye_integral_',num2str(debye_integral),'_group',num2str(group),'_aberration_induced',num2str(aberration),'_circleAperture',num2str(is_circle_aperture),'_PhazeSpacePSF_M',num2str(M),'_NA',num2str(NA),'_n',num2str(n),'_lambda',num2str(lambda*1e9),'nm_fml',num2str(fml*1e6),'u_ftl',num2str(ftl*1e6),'u_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_zspacing',num2str(zspacing*1e+6),'u'];
prefix = ['./PSF/debye_integral_',num2str(debye_integral),'_group',num2str(group),'_aberration_induced',num2str(aberration),'_circleAperture',num2str(is_circle_aperture),'_pupilMask',num2str(pupilMask),'_PhazeSpacePSF_M',num2str(M),'_NA',num2str(NA),'_n',num2str(n),'_lambda',num2str(lambda*1e9),'nm_fml',num2str(fml*1e6),'u_ftl',num2str(ftl*1e6),'u_zmin',num2str(zmin*1e+6),'u_zmax',num2str(zmax*1e+6),'u_zspacing',num2str(zspacing*1e+6),'u'];
save([prefix,'.mat'],'psf','-v7.3');
disp(['PSF computation complete.']);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% bitdepth = 16;
% psftif = psf / max(psf(:));
% psftif = psftif * (2^bitdepth - 1);
% res = zeros(size(psftif,1)*size(psftif,3), size(psftif,2)*size(psftif,4), size(psftif,5));
% for i = 1:size(psftif,5)
%     res(:,:,i) = sai_stack2sai_img(psftif(:,:,:,:,i));
% %     res(:,:,i) = sai2mpi(psf(:,:,:,:,i));
% end
% imwrite3d(res,[prefix,'.tif'],bitdepth);
% toc

