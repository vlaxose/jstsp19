% ProbitSEParam
%
% This script contains parameters that are used in the probit channel state
% evolution test, in which we compare the empirical performance of GAMP
% against the state evolution predictions of GAMP ensemble-averaged
% performance.  The specific model that we assume pairs a
% Bernoulli-Gaussian prior on the entries of the "true" weight vector, x,
% that defines a separating hyperplane between two discrete classes, 0 & 1.
% The training feature matrix, A_train, has entries that are drawn from an
% i.i.d. Gaussian distribution.  The class label training vector, y_train,
% is generated as
%                           y_train = (z > 0),
% where z = A_train * x.
%
% The discriminative classification model used in the probit state
% evolution assumes the following relationship between the entries of
% y_train and z:
%                       y_train(m) = h(z(m), w(m)),
% where h: R x R -> {0,1} is defined as h(z, w) = I{z > w}, i.e., h(z, w) =
% 1 when z > w, and h(z, w) = 0 otherwise.  The entries of the "noise"
% vector w are generated i.i.d. Gaussian with zero mean and variance v^2.
% Under this discriminative model,
%                   Pr{y = 1 | z} = Pr{h(z, w) = 1 | z},
%                                 = p(w < z | z),
%                                 = Phi(z/v),
% where Phi(.) is the CDF of the standard normal distribution.
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/11/12
% Change summary: 
%       - Created from sparseNLParam file (12/11/12; JAZ)
% Version 0.2
%

% Model parameters
N = 1000;           % Number of features (dimension of x)
NoverM = 2;         % Ratio of # of features-to-# of training samples
MoverK = 20;      	% Ratio of # of training samples-to-# of "active" features
BGmean = 0;        	% Bernoulli-Gaussian active mean
BGvar = 1;         	% Bernoulli-Gaussian active variance
probit_var = 1e-2;  % Probit channel variance (v^2 above)
nit = 25;           % # of iterations of GAMP
maxSumVal = false;  % Run MMSE GAMP (false) or MAP GAMP (true)?

% Empirical script parameters
ntest = 250;                % Number of Monte Carlo tests
savedat = true;             % Save data to file (true/false)?
saveFile = 'ProbitData';    % Filename

% *************************************************************************

% Add to path
addpath('../../main/');
addpath('../../stateEvo/');

% Calculate certain quantities based on model parameters
Mtrain = round(N / NoverM);     % # of training samples
K = round(Mtrain / MoverK);     % # of "active" (discriminative) features
sparseRat = K/N;                % Bernoulli-Gaussian activity probability

% Generate input estimation class
inputEst0 = AwgnEstimIn(BGmean, BGvar);
inputEst = SparseScaEstim(inputEst0, sparseRat);