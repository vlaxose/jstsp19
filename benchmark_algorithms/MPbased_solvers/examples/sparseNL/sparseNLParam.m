% sparseNLParam:  Parameter file for SE analysis of sparse nonlinear 
% estimation.  This file is a common file used by both the simulation and
% SE analysis programs.
%
% In this problem, x is a Bernoulli-Gaussian random vector, that we want to
% estimate from measurements of the form
%
%   y = f(A*x) + w,
%
% where A is a random matrix, w is Gaussian noise and f(z) is the sigmoidal
% output f(z) = 1/(1+exp(-ascale*z)), for a constant ascale > 0.

% Sets path
addpath('../../main/');
addpath('../../stateEvo/');

% Parameters
nx = 1000;        % Number of input components (dimension of x)
nz = 500;         % Number of output components (dimension of y)
sparseRat = 0.1;  % fraction of components of x that are non-zero
snr = 20;         % Peak SNR in dB.
psat = 0.3;       % Probability that ascale*z hits the saturation limit
satlev = 0.5;     % Saturation limit
xmean1 = 0;       % Input mean and variance (prior to sparsification)
xvar1 = 1;          
nit = 20;         % number of iterations of GAMP

% Compute scale factor based on saturation limit
% P(|ascale*z| >= 1) = psat.
xvar0 = sparseRat*xvar1;
zvar0 = xvar0;
ascale = 1/sqrt(2*zvar0)/satlev/erfinv(1-psat);

% Create output function
outFn = @(z) 2./(1+exp(-ascale*z))-1;
dz = 1e-3;
outDeriv = (outFn(dz)-outFn(0))/dz;
wvar = 10^(-0.1*snr);

% Generate input estimation class
inputEst0 = AwgnEstimIn(xmean1, xvar1);
inputEst = SparseScaEstim( inputEst0, sparseRat );
