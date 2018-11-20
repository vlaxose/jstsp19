%EMGMAMP  Expectation-Maximization Gaussian-Mixture AMP
% From the (possibly complex-valued) M-by-T matrix of noisy linear
% observations Y = AX + W, EMGMAMP returns approximately MMSE estimates
% of the N-by-T signal matrix X, where the focus is the sub-Nyquist case
% M < N, as in sparse reconstruction and compressive sensing.  In EMGMAMP,
% it is assumed that the signal coefficients X(n,t) are apriori independent
% and Bernoulli-Gaussian-mixture distributed according to the pdf:
%
%   p(X(n,t)) = (1-lambda(t))*delta(X(n,t))
%      + sum_k^L lambda(t)*omega(t,k) (C)Normal(X(n,t);theta(t,k),phi(t,k))
%
% where n = 1:N indexes the row of the signal matrix X,  t = 1:T
% indexes the column (timestep) of all matrices, and delta() denotes the
% Dirac delta pdf.  Above, lambda(t), omega(t,:), theta(t,:), and phi(t,:)
% are the sparsity rate, mixture weights, active-coefficient means, and
% active-coefficient variances, respectively, of the coefficients within
% the t-th column of X.  In EMGMAMP, the noise samples W(m,t) are assumed
% apriori independent and zero-mean Gaussian distributed:
%
%    p(W(m,t)) = (C)Normal(W(m,t);0,psi(t))
%
% where m = 1:M indexes the rows of the matrices W and Y, and psi(t)
% is the variance within the t-th column of W.
%
% The estimate of X is calculated using the Generalized Approximate Message
% Passing (GAMP) algorithm, as described in "Generalized Approximate Message
% Passing for Estimation with Random Linear Mixing" by Sundeep Rangan,
% with certain adaptive-stepsize enhancements.
%
% The parameters lambda, omega, theta, phi, and psi are automatically
% learned via the Expectation Maximization (EM) algorithm, using X and Z as
% the "hidden" data.
%
% The EMGMAMP algorithm is described in detail in "Expectation-Maximization
% Gaussian-Mixture Approximate Message Passing" by Jeremy Vila and
% Philip Schniter. However, the noise-variance learning in this implementation
% is an improvement over the method suggested in that paper.
%
%Syntax:
% [Xhat, EMfin] = EMGMAMP(Y, A) % suggested default
% [Xhat, EMfin, estHist, optEMfin, optGAMPfin] = EMGMAMP(Y, A, optEM, optGAMP) % full version
%
%Inputs:
% Y - matrix of observed measurements
%
% A - known mixing matrix (in either explicit or GAMP-object format)
%
% optEM - structure containing various EM option fields [optional]
%   .heavy_tailed       Set to true for heavy-tailed (compressible or
%                       non-compressible) signals.  [default=true]
%   .SNRdB              dB SNR used to initialize noise variance [default=20]
%   .learn_lambda       Set to true to learn lambda, set to false
%                       to never update lambda (stays at initialization)
%                       [default=true]
%   .learn_weights      Set to true to learn the weights omega of the
%                       Gaussian Mixture [default=true]
%   .learn_mean         Set to true to learn active mean (theta), set to false
%                       to never update theta (stays at initialization)
%                       [default=true]
%   .learn_var          Set to true to learn active variance (phi), set to
%                       false to never update active variance (stays at
%                       initialization) [default=true]
%   .learn_noisevar     Set to true to learn noise variance (psi), set to
%                       false to never update noise variance (stays at
%                       initialization) [default=true]
%   .maxBethe           Set to true to learn the output parameters via
%                       maximization of the Bethe Free Energy [default =
%                       true]
%   .sig_dim            Set to 'col' to allow different EM-learned signal
%                       params for each column in matrix X.  Set to 'row' 
%                       to allow different EM-learned signal params for 
%                       each row in matrix X. Set to 'joint' to force 
%                       common EM-learned parameters for all columns.
%                       [default='col']
%   .noise_dim          Set to 'col' to allow different EM-learned noise
%                       params for each column in matrix Y.  Set to 'row' 
%                       to allow different EM-learned signal params for 
%                       each row in matrix Y. Set to 'joint' to force 
%                       common EM-learned parameters for all columns.
%                       [default='joint']
%   .robust_gamp        Toggle on when the sensing matrix gives issues 
%                       (e.g., correlated columns). This option 
%                       automatically overrides GAMP options to handle 
%                       these cases [default = false]
%
%   Note: EMGMAMP(Y, A) or EMGMAMP(Y, A, []) sets optEM at defaults
%
%   WARNING! Set the following initializations only if confident they
%   accurately represent the model.  Otherwise, let the defaults preside.
%   Also, ensure that all GM-component vectors are of compatible lengths.
%
%   .L                  Number of mixture components [default=3, max=20]
%   .cmplx_in           Set to true if input is complex.  [default = 
%                       ~isreal(A^T*randn(M,1)) || ~isreal(Y)]
%   .cmplx_out          Set to true if output is complex.  [default = 
%                       ~isreal(Y)]
%   .lambda             Initialize overall sparsity rate lambda. Must be
%                       either scalar or 1-by-T vector.  [default is
%                       based on the LASSO phase transition curve]
%   .active_weights     Initialize GM component weights (omega).
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .active_mean        Initialize mean of active components (theta)
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .active_var         Initialize variance of active components (phi)
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .noise_var          Initialize the noise variance.  Must be either
%                       scalar or 1-by-T vector [default is based on .SNRdB]
%
%   Note: Setting optEM.heavy_tailed=true overrides the above parameter
%   initializations, and sets optEM.learn_mean=false.
%
% optGAMP - structure containing various GAMP option fields [optional] (see GampOpt.m)
%
%   Note: EMGMAMP(Y, A) or EMGMAMP(Y, A, [], []) sets GAMP parameters at defaults
%
%Outputs:
% Xhat - GAMP-estimated posterior means
%
% EMfin - Contains EM-estiated signal and noise parameters.
%   .Xvar               GAMP-estimated MMSE variance
%   .Rhat               GAMP quantity
%   .Rvar               GAMP quantity
%   .Zhat               GAMP-estimated MMSE mean of noiseless measurement
%   .Zvar               GAMP-estimated MMSE var of noiseless measurement
%   .Phat               GAMP quantity
%   .Pvar               GAMP quantity
%   .pi_n               Posterior activity probabilities.
%   .gamma_n            Posterior mean.
%   .nu_n               Posterior variance.
%   .beta_n             Per-Component posterior scaling quantities.
%   .lambda             Sparsity rate (lambda) [1-by-T for 'col' mode]
%   .active_weights     weights of GM (omega) [L-by-T for 'col' mode]
%   .active_mean        Means of active components (theta) [L-by-T for 'col' mode]
%   .active_var         Variances of active components (phi) [L-by-T for 'col' mode]
%   .noise_var          Variance of noise (psi) [1-by-T for 'col' mode]
%
% estHist - GAMP output structure containing per-iteration GAMP data (see gampEst.m)
%
% optEMfin - Contains the final values of optEM
%
% optGAMPfin - Contains the final values of optGAMP
%
%
% Coded by: Jeremy Vila & Philip Schniter, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu, schniter@ece.osu.edu
% Last change: 11/19/15
% Change summary:
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Fixed some minor bugs
%   v 2.0 (JV)- Accounts for MMV model.  Also made changes due to changes
%               in GAMP.
%   v 2.1 (JV)- Changed the way optGAMP is initialized.  Use must call
%               constructor first.
%   v 2.2 (JV)- Updated noise variance learning in accordance to new GAMP
%               outputs
%   v 3.0 (JV)- Added option to output GAMP Histories.
%   v 4.0 (JV)- Now outputs EM parameters Rhat, Rvar, Zhat, Zvar, pi,
%               gamma, beta, and nu.
%   v 4.1 (JV & PS) - Fixed some small bugs.  Increased speed of GAMP by
%               not requiring it to save GAMP history.  Changed some
%               default GAMP options.
%   v 4.2 (JV)- Adjusted EM and GAMP defaults for enhanced speed and
%               robustness.  Also adjusted input/ouput format.
%   v 4.3 (JV)- optEM.heavy_tailed = true is now the default.  Also added
%               optEM.robust_gamp option to handle cases when matrix is 
%               problematic
%   v 4.4 (JV)- added optEM.sig_dim = 'row' and optEM.noise_dim = 'row' to
%               learn EM parameters on each row.
%   v 5.0 (JV)- Updated noise variance learning procedure.  Also adaptively
%               set EM tolerance based on the learned SNR.
%   v 5.1 (JV)- Allowed noise variance learning with assuming either hidden
%               z or x variables.
%   v 5.2 (JV)- Allowed noise variance learning via maximization of bethe
%               free energy
%   v 5.3 (JV)- Improved warm starting and appending to history.  Changed
%               noise variance learning to maxBethe = true.
%   v 5.4 (JV)- Improved GAMP error checking as an additional stopping criterio
%   v 6.0 (PS)- changed noise-variance learning method, and ditched the 
%               external EM loop.  Now learning is done inside the EstimIn
%               and EstimOut functions
% Version 6.0
%
function [Xhat, EMfin, estHist, optEMfin, optGAMPfin] = EMGMAMP(Y, A, optEM, optGAMP)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
[M,N] = A.size();
T = size(Y, 2);
if size(Y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end

%Merge user-specified GAMP and EM options with defaults
if nargin <= 2
    optEM = [];
end
if nargin <= 3
    optGAMP = [];
end
[optGAMP, optEM] = check_opts(optGAMP, optEM);

%Initialize GM parameters
if optEM.L == 1
    %Initialize BG
    [lambda, theta, phi, optEM] = set_initsBG(optEM, Y, A, M, N, T);
    L = 1;
    if optEM.heavy_tailed
        theta = zeros(N,T);
        optEM.learn_mean = false;
    end
else
    %Initialize GM
    [lambda, omega, theta, phi, optEM] = set_initsGM(optEM, Y, A, M, N, T);
    L = size(theta,3);
    if (L ~= size(phi,3) || L ~= size(omega,3))
        error('There are an unequal amount of components for the active means, variances, and weights')
    end
end

%Specify input channel for real or complex signal distributions
if ~optEM.cmplx_in
    if L == 1
        inputEst = AwgnEstimIn(theta,phi,false, 'autoTune',true, 'tuneDim',optEM.sig_dim, 'mean0Tune',optEM.learn_mean, 'var0Tune',optEM.learn_var);
    else
        inputEst = GMEstimIn(omega,theta,phi, 'autoTune',true, 'tuneDim',optEM.sig_dim, 'omegaTune',optEM.learn_weights, 'thetaTune',optEM.learn_mean, 'phiTune',optEM.learn_var);
    end
else
    if L == 1
        inputEst = CAwgnEstimIn(theta,phi,false, 'autoTune',true, 'tuneDim',optEM.sig_dim, 'mean0Tune',optEM.learn_mean, 'var0Tune',optEM.learn_var);
    else
        inputEst = CGMEstimIn(omega,theta,phi, 'autoTune',true, 'tuneDim',optEM.sig_dim, 'omegaTune',optEM.learn_weights, 'thetaTune',optEM.learn_mean, 'phiTune',optEM.learn_var);
    end
end
inputEst = SparseScaEstim(inputEst,lambda,0, 'autoTune',optEM.learn_lambda, 'tuneDim',optEM.sig_dim);

%Specify output channel for real or complex noise distributions
if optEM.maxBethe, tuneMethod = 'Bethe'; else tuneMethod = 'ML'; end;
if ~optEM.cmplx_out
    outputEst = AwgnEstimOut(Y,optEM.noise_var,false, 'autoTune',optEM.learn_noisevar, 'tuneDim',optEM.noise_dim, 'tuneMethod',tuneMethod, 'tuneDamp',optEM.damp_wvar, 'counter',0);
else
    outputEst = CAwgnEstimOut(Y,optEM.noise_var,false, 'autoTune',optEM.learn_noisevar, 'tuneDim',optEM.noise_dim, 'tuneMethod',tuneMethod, 'tuneDamp',optEM.damp_wvar, 'counter',0);
end

%Perform GAMP
if nargout < 3
    [estFin,optFin] = gampEst(inputEst, outputEst, A, optGAMP);
    %[estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, optGAMP); figure(3); gampShowHist(estHist,optFin); 
else
    [estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, optGAMP);
end

%Output final parameter estimates
Xhat = estFin.xhat;
EMfin.Xvar = estFin.xvar;
EMfin.Zhat = estFin.zhat;
EMfin.Zvar = estFin.zvar;
EMfin.Rhat = estFin.rhat;
EMfin.Rvar = estFin.rvar;
EMfin.Phat = estFin.phat;
EMfin.Pvar = estFin.pvar;

%Output learned prior parameters
lambda = inputEst.p1;
if L==1 %BG parameters
    theta = inputEst.estim1.mean0;
    phi = inputEst.estim1.var0;

    if strcmp(optEM.sig_dim,'joint')
        EMfin.lambda = lambda(1,1);
        EMfin.active_mean = theta(1,1);
        EMfin.active_var = phi(1,1);
    elseif strcmp(optEM.sig_dim,'col')
        EMfin.lambda = lambda(1,:);
        EMfin.active_mean = theta(1,:);
        EMfin.active_var = phi(1,:);
    elseif strcmp(optEM.sig_dim,'row')
        EMfin.lambda = lambda(:,1);
        EMfin.active_mean = theta(:,1);
        EMfin.active_var = phi(:,1);
    end
    
else %GM parameters
    omega = inputEst.estim1.omega;
    theta = inputEst.estim1.theta;
    phi = inputEst.estim1.phi;

    if strcmp(optEM.sig_dim,'joint')
        EMfin.lambda = lambda(1,1);
        EMfin.active_weights = reshape(omega(1,1,:),1,L)';
        EMfin.active_mean = reshape(theta(1,1,:),1,L).';
        EMfin.active_var = reshape(phi(1,1,:),1,L)';
    elseif strcmp(optEM.sig_dim,'col')
        EMfin.lambda = lambda(1,:);
        EMfin.active_weights = reshape(omega(1,:,:),T,L)';
        EMfin.active_mean = reshape(theta(1,:,:),T,L).';
        EMfin.active_var = reshape(phi(1,:,:),T,L)';
    elseif strcmp(optEM.sig_dim,'row')
        EMfin.lambda = lambda(:,1);
        EMfin.active_weights = reshape(omega(:,1,:),N,L)';
        EMfin.active_mean = reshape(theta(:,1,:),N,L).';
        EMfin.active_var = reshape(phi(:,1,:),N,L)';
    end
end

%Output learned noise variance 
muw = outputEst.wvar;
muw = resize(muw,M,T);
if strcmp(optEM.noise_dim,'joint')
    EMfin.noise_var = muw(1,1);
elseif strcmp(optEM.noise_dim,'col')
    EMfin.noise_var = muw(1,:);
elseif strcmp(optEM.noise_dim,'row')
    EMfin.noise_var = muw(:,1);
end

%Output final options
optEM = rmfield(optEM,'noise_var');
optEMfin = optEM;
optGAMPfin = optFin;


return;
