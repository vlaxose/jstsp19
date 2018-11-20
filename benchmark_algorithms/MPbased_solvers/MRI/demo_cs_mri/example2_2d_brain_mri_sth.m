clear all;

% Compressed sensing reconstruction of undersampled muticoil 2D MRI data
% Undersampling along the phase-encoding dimension is simulated using a
% random sampling mask 
% Reconstruction is performed using the soft-thresholding algorithm
% The agorithm enforces joint sparsity on the multicoil image ensemble

% load fully-sampled data 
load data_2d_brain.mat;
[nx,ny,nc]=size(kdata);
% load undersampling mask
load maskR2.mat
% simulate undersampling
for ch=1:nc, data_acc(:,:,ch)=kdata(:,:,ch).*mask;end

% parameters for reconstruction
param.FT = Emat_xy(ones(nx,ny),b1); % multicoil model (b1: coil sensitivities)
param.W = Wavelet('Daubechies',4,4); % sparsifying transform
param.y=data_acc;
param.lambda=0.005;
param.nite=40;
param.tol=5e-4;
param.display=1;

% fully-sampled reconstruction
recon_full=param.FT'*kdata;
% initial reconstruction
recon_dft=param.FT'*data_acc;

% repetitions
tic
recon_cs=CSL1SoftThresh(recon_dft,param);
toc

figure;
subplot(2,3,1),imshow(abs(recon_full),[0,1]);title('Fully-sampled')
subplot(2,3,2),imshow(abs(recon_dft),[0,1]);title('Zero-filled FFT')
subplot(2,3,3),imshow(abs(recon_cs),[0,1]);title('CS')
subplot(2,3,5),imshow(5*abs(abs(recon_dft)-abs(recon_full)));title(strcat('x5-Error (RMSE = ',num2str(rms(abs(abs(recon_full(:))-abs(recon_dft(:)))),3),')'))
subplot(2,3,6),imshow(5*abs(abs(recon_cs)-abs(recon_full)));title(strcat('x5-Error (RMSE = ',num2str(rms(abs(abs(recon_full(:))-abs(recon_cs(:)))),3),')'))

