%% Synthetic Hyperspectral Unmixing using Pure Pixel Data
% Given a pure pixel data set Y of size M by T, this script unmixes Y ~ SA, 
% where the endemember matrix S of size M by N, and abundance maps
% A of size N by T.   
%
% This file uses HUTAMP joint endmember/abundance recovery.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release

%%
clc
clear all
clf

%load synthetic endmembers S and abundances A
load synth_HSI_USGS_stripes.mat
S = A;
A = X;

%Get problem sizes
[M, N] = size(S);
[~, T] = size(A);
T1 = sqrt(T);

%% Form measurements
SNRdB = 30;

Z = S*A;
Z_mean = sum(sum(Z))/M/T;

%Define noise variance
muw = sum(sum((Z).^2))/M/T*10^(-SNRdB/10);

%Form noisy measurements
Y = Z + sqrt(muw)*randn(M,T);
Y = reshape(Y, M, T1, T1);


%% Perform HUTAMP
tstart = tic;

%Track HUTAMP's progress
Z = (S - mean(S(:)))*A;
optBiGAMP.error_function = @(qval) 20*log10(norm(qval(1:end-1,:) - Z,'fro') / norm(Z,'fro'));
%Reduce EM tolerance for more precise estimates
optALG.EMtol = 1e-9;
%Run HUTAMP
[estFin, stateFin, estHist] = HUTAMP(Y, N, optALG);
tGAMP = toc(tstart);

% Compute error for HUTAMP
% Since matrices have permutation ambiguities, do a greedy search to ground
% truth.
P = find_perm(S,estFin.Shat);
estFin.Shat = estFin.Shat*P;
dic_error = 20*log10(norm(S - estFin.Shat,'fro')/norm(S,'fro'));
disp(['HUTAMP Dictionary error was ' num2str(dic_error) 'dB'])

estFin.Ahat = P'*estFin.Ahat;
sig_error = 20*log10(norm(A - estFin.Ahat,'fro')/norm(A,'fro'));
disp(['HUTAMP Signal error was ' num2str(sig_error) 'dB'])

%Plot abundance maps
figure(3)
for n = 1:N
    subplot(2, N, n)
    imshow(reshape(A(n,:),T1,T1))
    title(materials{n})
    subplot(2, N, n+N)
    imshow(reshape(estFin.Ahat(n,:),T1,T1))
end
