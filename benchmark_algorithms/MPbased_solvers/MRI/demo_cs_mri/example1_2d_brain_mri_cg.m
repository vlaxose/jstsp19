clear all;

% Compressed sensing reconstruction of undersampled muticoil 2D MRI data
% Undersampling along the phase-encoding dimension is simulated using a
% random sampling mask 
% Reconstruction is performed using the non-linear conjugate gradient algorithm
% The agorithm enforces joint sparsity on the multicoil image ensemble

% load fully-sampled data 
load data_2d_brain.mat;
[nx,ny,nc]=size(kdata);
% load undersampling mask
load maskR2.mat;
% simulate undersampling
for ch=1:nc, data_acc(:,:,ch)=kdata(:,:,ch).*mask;end

% parameters for reconstruction
param.E = Emat_xy(mask,b1); % multicoil model (b1: coil sensitivities)
param.W = Wavelet('Daubechies',4,4);param.L1Weight=0.005;
param.TV = TVOP();param.TVWeight=0.001;
param.y = data_acc;
param.nite = 8;
param.display=1;
ite=3;

% fully-sampled reconstruction
param.Efull = Emat_xy(ones(size(mask)),b1);
recon_full=param.Efull'*kdata;
% initial reconstruction
recon_dft=param.E'*data_acc;

tic
recon_cs=recon_dft;
for n=1:ite,
	recon_cs = CSL1NlCg(recon_cs,param);
end
toc

figure;
subplot(2,3,1),imshow(abs(recon_full),[0,1]);title('Fully-sampled')
subplot(2,3,2),imshow(abs(recon_dft),[0,1]);title('Zero-filled FFT')
subplot(2,3,3),imshow(abs(recon_cs),[0,1]);title('CS')
subplot(2,3,5),imshow(abs(abs(recon_dft)-abs(recon_full)),[0,0.2]);title(strcat('x5-Error (RMSE = ',num2str(rms(abs(abs(recon_full(:))-abs(recon_dft(:)))),3),')'))
subplot(2,3,6),imshow(abs(abs(recon_cs)-abs(recon_full)),[0,0.2]);title(strcat('x5-Error (RMSE = ',num2str(rms(abs(abs(recon_full(:))-abs(recon_cs(:)))),3),')'))

