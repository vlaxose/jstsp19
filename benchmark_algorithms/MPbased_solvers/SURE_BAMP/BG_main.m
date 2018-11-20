% BAMP reconstruction of the Bernoulli-Gaussian data
% by Chunli 26/01/2014

% clc; clear;
% addpath 'Hybrid_MIMO/';
% [yv, Phi] = Adaptive_Channel_Estimation_Multi_Path2;
% load HybridSystem.mat
% generate the dense signal----------------------------------
N = 5000;     % signal dimension
r = 0.5;
m = round(r*N);      % measurement

lambda = 0.1;
n1 = round(lambda*N);  
x = zeros(1,N);
x(1:n1) = randn(1, n1);   % Gaussian component


perm = randperm(N);
xo = intrlv(x, perm);

% % sensing matrix---------------------------------------------
Phi = randn(m,N)/sqrt(m);
for ii=1:N
    Phi(:,ii)=Phi(:,ii)/norm(Phi(:,ii));
end

y = Phi*(xo)';
% % adding noise------------------------------------------------plot(rex)
SNR = 50;
varw = norm(y)^2/m*10^(-SNR/10);
stdw = sqrt(varw);
% %stdw = 0;

% % CS observation
input = y + stdw*randn(m, 1);

% BAMP denoising commence
rex = BAMP_GM_simple(input, Phi, 1, 0, 0, lambda, 1e-6, 200);
% print result
NMSE = 10*log10(sum((xo.^2))/sum((xo-rex').^2))