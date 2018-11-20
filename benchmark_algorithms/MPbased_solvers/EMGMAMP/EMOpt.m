%***********NOTE************
%
% This function sets the EM options to defaults.
%
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 8/11/14
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Added maxBethe option
%
% Version 1.1

function optEM = EMOpt()

%Toggle 'heavy_tailed mode'
optEM.heavy_tailed = true;

%Set default SNR (in dB) used in the initialization of the noise variance
optEM.SNRdB = 10;

%maximum number of EM iterations.
optEM.maxEMiter = 200;

%maximum tolerance to exit EM loop.
optEM.maxTol = 1e-4;

%Toggle learning of mean, variances, weights, sparsity rate, and noise variance
optEM.learn_mean = true;
optEM.learn_var = true;
optEM.learn_weights = true;
optEM.learn_lambda = true;
optEM.learn_noisevar = true;

%Toggle whether noise variance is learned with Z variables or X variables
optEM.hiddenZ = true;

%Toggle whether noise is learned via maximization of Bethe free energy
optEM.maxBethe = true;

%Assume each column of X has distinct signal parameters
optEM.sig_dim = 'col';

%Assume columns of Y have common noise parameters
optEM.noise_dim = 'joint';

%Set default number of mixture components
optEM.L = 3;

%Set minium allowed variance of a GM component
optEM.minVar = 1e-5;

%Set maximum number of mixture components for model order selection
optEM.Lmax = 5;

%Set maximum number of iterations for learning model order
optEM.maxLiter = 3;

%Toggle on when the sensing matrix gives issues (e.g., correlated columns).
%This option automatically overrides GAMP options to handle these cases
optEM.robust_gamp = false;

return
