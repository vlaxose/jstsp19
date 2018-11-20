% PCAFSNMF
% This function returns estimated endmembers using FSNMF with PCA postprocessing.
% 
% Coded by Jeremy Vila
% 4-03-14

function [Shat, ind] = PCA_FSNMF(Y,N)

%Perform FSNMF on projected data
ind = FastSepNMF(Y,N,1);

T = size(Y,2);
%Dimensionality of subspace
p = N-1;

%Find mean across materials for each spectral value
mn0 = mean(Y,2);      
mn = repmat(mn0,[1 T]); % mean of each band
Y0 = Y - mn;           % data with zero-mean 

[Ud,~,~] = svds(Y0*Y0'/T,p);  % computes the p-projection matrix 

Yp =  Ud' * Y0;                 % project thezeros mean data onto p-subspace

%Project back to original space to get recovered endmembers
Shat = Ud*Yp(:,ind) + repmat(mn0,[1 N]);

return