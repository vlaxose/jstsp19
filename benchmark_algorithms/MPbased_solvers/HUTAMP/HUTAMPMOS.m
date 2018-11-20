%% HUTAMP Model Order Selection
% Given the hyperspectral data Y of size M by T (where T = T_1 * T_2), we
% assume the (noisy) bilinear model
%
% Y = SA + W,
%
% where, with N materials, the columns S (of size M by N) are the 
% endmembers, the rows of A (of size N by T) are the abundances, and W 
% is assumed to be i.i.d. AWGN with variance psi.  For each material N, we
% assumed a separate Gaussian prior on the endmembers and a Bernoulli
% non-negative Gaussian mixture on the abundances.
%
% Furthermore, we leverage known endmember coherence and spatial coherence 
% using a "turbo" message passing approach through a Gauss-Markov process 
% across endmember spectra and Markov Random Field across material 
% abundances, respectively.
%
% To perform the bilinear inference task, we employ the "Bilinear
% Generalized Approximate Message Passing" (BiGAMP) algorithm described in
% "Bilinear Generalized Approximate Message Passing" by Jason Parker,
% Philip Schniter, and Volkan Cevher.
%
% The remaining parameters of the models are automatically tuned using the
% expectation maximization algorithm.
%
% Since, in general, the number of materials is apriori unknown, we propose
% using a penalized log-likelihood function to select the number of
% materials N.  For the penalty, we use the small sample corrected Aikaike
% Information Criterion (AICc).  The Algorithm is as follows:
%
% 1) Run HUTAMP with N = optALG.Nmin.
% 2) Run HUTAMP with N+1
% 3) Evaluate penalized log-likelihood cost.
%  a) if cost increased, go to 2)
%  b) if cost decreased, select N* = N - 1;
%
% Syntax 
%   [estFin] = HUTAMP(Y, N) or, for full inputs and outputs
%   [estFin, stateFin, estHist] = HUTAMP(Y, N, optALG, optBiGAMP)
%
% Inputs
%  Y            The HUT data cube of size M by T_1 by T_2.
%  optBiGAMP    numerous BiGAMP options.  All options are described in
%               BiGAMPOpt().  The main are highlighted below:
%   .nit            Number of BiGAMP iterations [default 30]
%   .verbose        Turns On/Off Verboisity [default false]
%   .varThresh      Places a threshold on variance terms [defaul 1e6]
%   .tol            Defines threshold tolerance for exiting BiGAMP 
%                   [default 1e-6]
%   .step           Defines initial step size [default 0.01]
%   .stepMin        For adaptive step sizes, sets min step size 
%                   [default 0.01]
%   .stepMax        For adaptive step sizes, sets max step size 
%                   [default 0.3]
%   .adaptStep      Turns on/off adaptive step size [default true]
%   .error_function   Function Handle which calculates the error of Z = AX
%
% Inputs
%  Y            The HUT data cube of size M by T_1 by T_2.
%  N            Number of materials present.  If unknown, assume to be
%               small [default = 5]
%  optBiGAMP    numerous BiGAMP options.  All options are described in
%               BiGAMPOpt().  The main are highlighted below:
%   .nit                number of iterations
%   .pvarMin            minimum variance of each element of p
%   .xvarMin            minimum variance of each element of A 
%                       (BiGAMP X <-> HU A)
%   .Avarmin            minimum variance of each element of S
%                       (BiGAMP A <-> HU S)
%   .step               initial stepsize
%   .stepMin            minimum stepsize
%   .stepMax            maximum stepsize
%   .stepIncr           Multiplicative stepsize increase, when successful
%   .stepDecr           Multiplicative stepsize decrease, when unsuccessful
%   .pvarStep           Logical flag to use stepsize in pvar calculation.
%   .varNorm            Option to "normalize" variances for computation.
%   .adaptStep          adaptive stepsize [important if A column-correlated]
%   .stepWindow         stepsize check window size
%   .bbStep             Barzilai Borwein stepsize [if A column-correlated]
%   .verbose            Print results in each iteration
%   .tol                Convergence tolerance
%   .stepTol            minimum allowed stepsize
%   .xhat0              the initialization of x for BiGAMP [default handled
%                       internally] (BiGAMP X <-> HU A)
%   .xvar0              the intialization of variance of each x_n for BiGAMP
%                       [default handled internally]
%   .Ahat0              the initialization of A for BiGAMP [default handled
%                       internally] (BiGAMP X <-> HU A)
%   .Avar0              the intialization of variance of elements of A for 
%                       BiGAMP [default handled internally] (BiGAMP A <-> HU S)
%   .shat0              the initialization of s for BiGAMP [default handled
%                       internally] (BiGAMP A <-> HU S)
%
%  optALG        numerous algorithm options
%   .maxEMiter          Define maximum EM iterations [default=20]
%   .EMtol              Define exit tolerance for EM [default=1e-6]
%   .SNRdB              Assumed SNR in dB of the measurements [default=0]
%   .spectral_coherence Turn off/on performing spectral coherence and
%                       learning the parameters via EM [default=true].
%   .spatial_coherence  Turn off/on performing spatial coherence and
%                       learning the parameters via EM [default=true].
%   .learn_lambda       Set to true to learn sparsity rate, set to false
%                       to never update sparsity rate (stays at 
%                       initialization) [default=false]
%   .learn_weights      Set to true to learn the active_weights omega of the
%                       Gaussian Mixture [default=true]
%   .learn_loc          Set to true to learn active locations (theta), set 
%                       to false to never update theta (stays at 
%                       initialization) [default=true]
%   .learn_scales       Set to true to learn active scales (phi), set to
%                       false to never update active variance (stays at
%                       initialization) [default=true]
%   .learn_noisevar     Set to true to learn noise variance (psi), set to
%                       false to never update noise variance (stays at
%                       initialization) [default=true]
%   .EEinit             Method to initialize Shat0.  Can either be {VCA, 
%                       FSNMF, data} or a numeric matrix.  [default FSNMF]
%   .maxBethe           Set to true to learn the output parameters via
%                       maximization of the Bethe Free Energy [default =
%                       true]
%   .Nmin               Minimum number of materials [default 2]
%   .Nmax               Maximum number of materials [default 10]
%
% You can also initialize the parameters of the prior distributions and for
% the turbo methods.  Only set these if you are confident that these are
% true and if experienced with this software.
%   .L              Number of NN-gaussian mixture components for the
%                   abundances [default = 3]
%   .lambda         Sparsity rate on abundances [default = 1/N]
%   .active_weights Weights on the abundance NNGM components 
%                   [default is a fit to (0,1) Uniform distribution]
%   .active_loc     Locations on the abundance NNGM components
%                   [default is a fit to uniform distribution]
%   .active_scales  Scales on the abundance NNGM components
%                   [default is a fit to uniform distribution]
%   .MRF_betaH      Horizontal beta parameter [default is 0.4]
%   .MRF_betaV      Vertical beta parameter [default is 0.4]
%   .MRF_alpha      MRF alpha parameter [default is 0.4]
%   .specMean       Mean of the 
%                   [default is final estimate of VCA]
%   .specVar        Variances on the abundance NNGM components
%                   [default is 0.1]
%   .specCorr       Gauss-Markov correlation parameter [default is based on
%                   correlation of the first materials spectra]
%   .noise_var      Initial noise variance of W [default based on 
%                   energy of Y]
%
% Outputs
% estFin        Structure containing many BiGAMP outputs
%   .Shat           MMSE estimate of endmembers S
%   .Ahat           MMSE estimate of abundances A
%   .Svar           MMSE variance estimates of S
%   .Avar           MMSE variance estiamtes of A
%   .rhat           BiGAMP quantity
%   .rvar           BiGAMP quantity
%   .qhat           BiGAMP quantity
%   .qvar           BiGAMP quantity
%   .zvar           BiGAMP quantity
%   .phat           BiGAMP quantity
%   .pvar           BiGAMP quantity
% stateFin       Structure containing various parameters
% estHist        Structure containing various BiGAMP histories.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Added new noise variance learning procedure

function [estFin, stateFin, estHist, cost_vec]...
    = HUTAMPMOS(Y, optALG, optBiGAMP)

%Find and set dimensions of problem
[M, T1, T2] = size(Y);

if T2 == 1
    T = T1;
else
    T = T1*T2;
end

%Preset all GAMP defaults
if nargin <= 1
    optALG = [];
end

%Preset all EM defaults
if nargin <= 2
    optBiGAMP = [];
end

%Set default BiGAMP options
[optALG, optBiGAMP] = set_opts_HUTAMP(optALG, optBiGAMP);

%Set N to minimum number of materials
N = optALG.Nmin;
[estFin, stateFin, estHist] = HUTAMP(Y, N, optALG, optBiGAMP);

%Based on noise variance learning technique the degrees of freedom change
if optALG.maxBethe
    noiseConst = M;
else 
    noiseConst = 1;
end

%evaluate cost
logpyz = M*T*log(norm(reshape(Y,M,T) - estFin.Shat*estFin.Ahat,'fro')^2/M/T);
%Define degrees of freedom from S,A, and parameters
L = size(stateFin.active_weights,3);
DoF = M*N + (N-1)*T + 5*N + 2*N*L + N*(L-1) + noiseConst;
AICc = (2*M*T*DoF) /  (M*T - DoF - 1); 

%Initialize cost vector
cost_vec = nan(optALG.Nmax - optALG.Nmin,1);
cost = logpyz + AICc;
cost_vec(1) = cost;

%loop over increasing number of materials
for N = optALG.Nmin+1:optALG.Nmax
    [estFin_new, stateFin_new, estHist_new] = HUTAMP2(Y, N, optALG, optBiGAMP);
    
    %evaluate cost
    logpyz = M*T*log(norm(reshape(Y,M,T) - estFin_new.Shat*estFin_new.Ahat,'fro')^2/M/T);
    DoF = M*N + (N-1)*T + 5*N + 2*N*L + N*(L-1) + noiseConst; %Compute new degrees of freedom
    AICc = (2*M*T*DoF) /  (M*T - DoF - 1); 
    cost_new = logpyz + AICc;
    cost_vec(N - optALG.Nmin + 1) = cost_new;  %save the cost
    
    if cost_new < cost
        estHist = estHist_new;
        estFin = estFin_new;
        stateFin = stateFin_new;
        cost = cost_new;
    else
        break
    end
    
end

return