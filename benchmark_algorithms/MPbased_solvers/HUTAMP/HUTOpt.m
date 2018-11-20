%***********NOTE************
%
% This function sets the HUTAMP options to defaults.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/13
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

function optALG = HUTOpt()

%maximum number of SIMP iterations
optALG.maxEMiter = 20;

%stopping tolerance of EM loop
optALG.EMtol = 1e-7;

%Assumed SNRdB
optALG.SNRdB = 10;

%learn noise variance via Bethe Free enrgy maximization
optALG.maxBethe = true;

%Toggle learning of the mean, variances, weights, sparsity rates, and noise variance
optALG.learn_scales = true;
optALG.learn_loc = true;
optALG.learn_weights = true;
optALG.learn_lambda = false;
optALG.learn_noisevar = true;

%Toggle learning GaussMarkov parameters
optALG.learn_GausMark = true;

%Toggle learning of MRF parameters
optALG.learn_MRF = true;

%Set number of NNGM components
optALG.L = 3;

%Set spectral and spatial coherence to true
optALG.spectralCoherence = true;
optALG.spatialCoherence = true;

%Set default initialization to VCA
optALG.EEinit = 'VCA';

%Set minimum and maximum number of materials
optALG.Nmin = 2;
optALG.Nmax = 10;

return