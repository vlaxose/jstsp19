clear all;

% Compressed sensing reconstruction of 2D cardiac cine MRI data (video of a
% beating heart)
% Undersampling is simulated using a different ky undersampling for each 
% time point (practical case)
% Reconstruction is performed using the non-linear conjugate gradient algorithm
% The agorithm enforces joint sparsity on the multicoil image ensemble

% More details about the algorithm can be found in the following paper:
% Combination of compressed sensing and parallel imaging for highly 
% accelerated first-pass cardiac perfusion MRI.
% Otazo R, Kim D, Axel L, Sodickson DK.
% Magn Reson Med. 2010 Sep;64(3):767-76.

% load fully-sampled data 
load data_2d_cardiac;
[nx,ny,nt,nc]=size(kdata);
% load undersampling mask
load maskR4_cine.mat
% simulate undersampling
for ch=1:nc, data_acc(:,:,:,ch)=kdata(:,:,:,ch).*mask;end

% parameters for reconstruction
param.E = Emat_xyt(mask,b1); % multicoil model (b1: coil sensitivities)
param.W = TempFFT(3);param.L1Weight=0.005;
param.TV = TVOP();param.TVWeight=0;
param.y = data_acc;
param.nite = 8;
param.display=1;
ite=3;

% fully-sampled reconstruction
param.Efull = Emat_xyt(ones(size(mask)),b1);
recon_full=param.Efull'*kdata;
% initial reconstruction
recon_dft=param.E'*data_acc;

tic
recon_cs=recon_dft;
for n=1:ite,
	recon_cs = CSL1NlCg(recon_cs,param);
end
toc

% display 4 frames
recon_full2=recon_full(49:end,49:end,1);recon_full2=cat(2,recon_full2,recon_full(49:end,49:end,7));recon_full2=cat(2,recon_full2,recon_full(49:end,49:end,13));recon_full2=cat(2,recon_full2,recon_full(49:end,49:end,19));
recon_dft2=recon_dft(49:end,49:end,1);recon_dft2=cat(2,recon_dft2,recon_dft(49:end,49:end,7));recon_dft2=cat(2,recon_dft2,recon_dft(49:end,49:end,13));recon_dft2=cat(2,recon_dft2,recon_dft(49:end,49:end,19));
recon_cs2=recon_cs(49:end,49:end,1);recon_cs2=cat(2,recon_cs2,recon_cs(49:end,49:end,7));recon_cs2=cat(2,recon_cs2,recon_cs(49:end,49:end,13));recon_cs2=cat(2,recon_cs2,recon_cs(49:end,49:end,19));
figure;
subplot(3,2,1),imshow(abs(recon_full2),[0,1]);title('Fully-sampled')
subplot(3,2,3),imshow(abs(recon_dft2),[0,1]);title('Zero-filled FFT')
subplot(3,2,4),imshow(3*abs(abs(recon_dft2)-abs(recon_full2)));title(strcat('x3-Error (RMSE = ',num2str(rms(abs(abs(recon_full(:))-abs(recon_dft(:)))),3),')'))
subplot(3,2,5),imshow(abs(recon_cs2),[0,1]);title('CS')
subplot(3,2,6),imshow(3*abs(abs(recon_cs2)-abs(recon_full2)));title(strcat('x3-Error (RMSE = ',num2str(rms(abs(abs(recon_full(:))-abs(recon_cs(:)))),3),')'))


