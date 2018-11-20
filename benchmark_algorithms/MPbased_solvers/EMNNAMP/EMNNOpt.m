%***********NOTE************
%
% This function sets the EMNNAMP options to defaults.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 10/13/14
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Added maxBethe option
%   v 1.2 (JV)- Increased max iters. default amxBEthe = true. 
%
% Version 1.2

function [optALG, optEM] = EMNNOpt()

%Set 
optALG.alg_type = 'NNGMAMP';

%Set noise to be laplacian or AWGN
optALG.laplace_noise = false;

%Set tolerance for linear equality constraint
optALG.LEtol = 1e-9;

%maximum number of SIMP iterations
optEM.maxEMiter = 200;

%EM tolerance
optEM.EMtol = 1e-5;

optEM.SNRdB = 20;

%Toggle learning of the mean, variances, weights, sparsity rates, and noise variance
optEM.learn_loc = true;
optEM.learn_scales = true;
optEM.learn_weights = true;
optEM.learn_tau = true;
optEM.learn_noisevar = true;
optEM.learn_inExpRate = true;
optEM.learn_outLapRate = true;

%Toggle whether noise variance is learned with Z variables or X variables
optEM.hiddenZ = false;

%Toggle whether output channel parameters are learned via maximization of
%Bethe Free energy.
optEM.maxBethe = true;

%Assume all coefficients of X have the same marginals
optEM.inDim = 'col';

%Assume columns of Y have common noise parameters
optEM.outDim = 'joint'; % otherwise use 'col' for distinct parameters

%Set number of NNGM components
optEM.L = 3;

% Set minimum value for scale parameters
optEM.minScale = 1e-10;

return