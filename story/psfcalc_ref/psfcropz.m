addpath(genpath('./'))
% load('./PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-20u_zmax20u_zspacing0.4u.mat');
% name = ['./PSF/psf-20190818_63x.mat'];
% name = ['./PSF/psf-20190818_63x.mat'];
% name = ['./PSF/Experimental_psf_M63_NA1.4_zmin-12u_zmax12u.mat'];
% name = ['./PSF/experimental_zspacing0.4u_z75.mat'];
% name = ['./PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-20u_zmax20u_zspacing0.4u.mat'];
% name = ['./PSF/circle_aperture_1_aberration_induced0_PhazeSpacePSF_M63_NA1.4_zmin-20u_zmax20u_zspacing0.4u.mat']
% name = ['../PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u_freqReweighted.mat']
% name = ['../PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u.mat']
% name = ['../PSF/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u_freqReweighted0.2.mat']
% name = ['../../PSF/expsf-M63-20190818.mat']
name = ['/home/slfm/as13000/kimchange/psfcalc/PSF/Ideal_circle_aperture_1_aberration_induced0_PhazeSpacePSF_M43.067_NA1.05_zmin-50u_zmax50u_zspacing1u.mat'];
name = ['/home/slfm/as13000/kimchange/psfcalc/PSF/Ideal_circle_aperture_0_aberration999_PhazeSpacePSF_M63_NA1.4_zmin-15u_zmax15u_zspacing0.2u.mat'];
name = ['/home/slfm/as13000/kimchange/PSF/group0_aberration_induced0_PhazeSpacePSF_M63_NA1.4_n1.515_lambda525nm_fml2100u_ftl165000u_zmin-14.8u_zmax14.8u_zspacing0.2u.mat'];
% name = ['/home/slfm/as13000/kimchange/PSF/group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M63_NA1.4_n1.515_lambda525nm_fml2100u_ftl165000u_zmin-14.8u_zmax14.8u_zspacing0.2u.mat'];
name = ['./PSF/group0_aberration_induced0.3_circleAperture1_PhazeSpacePSF_M63_NA1.4_n1.515_lambda525nm_fml2100u_ftl165000u_zmin-14.8u_zmax14.8u_zspacing0.2u.mat'];
name = ['./PSF//debye_integral_0_group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M60_NA1.42_n1.518_lambda525nm_fml2100u_ftl165000u_zmin-10u_zmax10u_zspacing0.2u.mat'];
name = ['./PSF//debye_integral_0_group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M60_NA1.42_n1.518_lambda525nm_fml2100u_ftl165000u_zmin-10u_zmax10u_zspacing0.2u.mat'];
name = ['./PSF/debye_integral_0_group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M60_NA1.42_n1.518_lambda525nm_fml2100u_ftl165000u_zmin0u_zmax0u_zspacing0.2u.mat'];
name = ['./PSF/test_SHRMS_debye_integral_0_group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M60_NA1.42_n1.518_lambda525nm_fml2100u_ftl165000u_zmin-10u_zmax10u_zspacing0.2u.mat'];
name = ['./PSF/debye_integral_1_group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M60_NA1.42_n1.518_lambda525nm_fml2100u_ftl165000u_zmin-10.4u_zmax10.4u_zspacing0.2u.mat']

% name = ['./PSF/group0_aberration_induced0_circleAperture1_PhazeSpacePSF_M60_NA1.42_n1.518_lambda525nm_fml2100u_ftl165000u_zmin-10u_zmax10u_zspacing0.2u.mat'];
load(name);
psf = uint8(psf / max(psf(:)) * 255*4);
% psf = psf(:,:,:,:,2:end-1);
if size(psf,1) > 377
    marginsize = (size(psf,1)-377) / 2;
    psf = psf( 1 + marginsize : end - marginsize, 1 + marginsize : end - marginsize, :, :, : );
end

psf = padarray(psf, [(377-size(psf,1))/2, (377-size(psf,2))/2], 0, 'both');
psf([1,end],:,:,:,:) = 255; psf(:,[1,end],:,:,:) = 255;
res = uint8(zeros(size(psf,1)*size(psf,3), size(psf,2)*size(psf,4), size(psf,5)));
for i = 1:size(psf,5)
    res(:,:,i) = sai_stack2sai_img(psf(:,:,:,:,i));
%     res(:,:,i) = sai2mpi(psf(:,:,:,:,i));
end

% if size(res,1) > 2000 && size(res,3) > 100
%     imwriteTFSK(res(:,:,1:round(size(res,3)/2)), ...
%         [recon_name_perfix,'_vid', num2str(frame),'_iter_',num2str(i),'.0.tiff']);
%     imwriteTFSK(A(:,:,round(size(A,3)/2)+1:end), ...
%         [recon_name_perfix,'_vid', num2str(frame),'_iter_',num2str(i),'.1.tiff']);
% else
%     imwriteTFSK(res , [recon_name_perfix,'_vid', num2str(frame),'_iter_',num2str(i),'.tiff']);
% end
% res = res / max(res(:)) * 255;
imwriteTFSK(uint8(res), [name(1:end-4),'.tiff']);
% imwrite(res(:,:,1), [name(1:end-4),'.tiff']);
% for ii = 2:size(psf,5)
%     imwrite(res(:,:,ii), [name(1:end-4),'.tiff'], 'WriteMode', 'append');
% end