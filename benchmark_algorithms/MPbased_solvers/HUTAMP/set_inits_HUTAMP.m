%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by HUTAMP
%
% This function sets initializations for all unspecified HUTAMP parameters
% to defaults
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

function stateFin = set_inits_HUTAMP(Y, Shat, Ahat, N, optALG, stateFin)

[M,T] = size(Y);

%% Initialize MRF parameters
if isfield(optALG,'MRF_betaH')
    stateFin.betaH = optALG.MRF_betaH;
else
    stateFin.betaH = 0.4*ones(N,1);
end
if isfield(optALG,'MRF_betaV')
    stateFin.betaV = optALG.MRF_betaV;
else
    stateFin.betaV = 0.4*ones(N,1);
end
if isfield(optALG,'MRF_alpha')
    stateFin.alpha = optALG.MRF_alpha;
else
    stateFin.alpha = 0.4*ones(N,1);
end

%% %Initialize Gauss Markov parameters
%Determine signal variance of matrix or of each column
sig_var = (norm(Y,'fro')^2-T*mean(stateFin.noise_var,2));
sig_var = resize(sig_var,M,1);

%Initialize the per-material correlation paremeter in Gauss Markov models
if isfield(optALG,'specCorr')
    stateFin.specCorr = resize(optALG.specCorr,M,N);
else
    inProdSum = 0;
    for m = 1:M-1
        inProdSum = inProdSum + Y(m,:)*Y(m+1,:)'./sig_var(m);
    end
    
    stateFin.specCorr = resize(max(min(...
        1 - inProdSum/(M-1), 0.5), 0.015),M,N,1);
end
%Initialize the mean of the per-material Gauss Markov chains
if isfield(optALG,'specMean')
    stateFin.specMean = resize(optALG.specMean,M,N);
else
    stateFin.specMean = repmat(mean(Shat),M,1);
end
%Initialize the variance the per-material Gauss Markov chains
if isfield(optALG,'specVar')
    stateFin.specVar = resize(optALG.specVar,M,N);
else
    stateFin.specVar = repmat(1*var(Shat),M,1);
end

return