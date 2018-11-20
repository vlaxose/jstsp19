%EMBGAMP  Expectation-Maximization Bernoulli-Gaussian AMP
% From the (possibly complex-valued) M-by-T matrix of noisy linear 
% observations Y = AX + W, EMBGAMP returns approximately MMSE estimates 
% of the N-by-T signal matrix X, where the focus is the sub-Nyquist case 
% M < N, as in sparse reconstruction and compressive sensing.  In EMBGAMP,
% it is assumed that the signal coefficients X(n,t) are apriori independent 
% and Bernoulli-Gaussian distributed according to the pdf:
%
% p_X(X(n,t)) = (1-lambda(t))*delta(X(n,t)) 
%               + lambda(t) (C)normal(X(n,t);theta(t),phi(t))
%
% where n = 1:N indexes the row of the signal matrix X,  t = 1:T 
% indexes the column (timestep) of all matrices, and delta() denotes the
% Dirac delta pdf.  Above, lambda(t), theta(t), and phi(t) are the 
% sparsity rate, active-coefficient mean, and active-coefficient variance,
% respectively, of the coefficients within the t-th column of X.  
% In EMBGAMP, the noise samples W(m,t) are assumed apriori independent and 
% zero-mean Gaussian distributed: 
%
%    p_W(W(m,t)) = (C)Normal(W(m,t);0,psi(t))
%
% where m = 1:M indexes the rows of the matrices W and Y, and psi(t) 
% is the variance within the t-th column of W.  
%
% The estimate of X is calculated using the Generalized Approximate Message 
% Passing (GAMP) algorithm, as described in "Generalized Approximate Message 
% Passing for Estimation with Random Linear Mixing" by Sundeep Rangan,
% with certain adaptive-stepsize enhancements.
%
% The parameters lambda, theta, phi, and psi are automatically learned
% via the Expectation Maximization (EM) algorithm, using X and W as 
% the "hidden" data. 
%
% The EMBGAMP algorithm is described in detail in "Expectation-Maximization
% Bernoulli-Gaussian Approximate Message Passing" by Jeremy Vila and 
% Philip Schniter.
%
%Syntax:
% [Xhat, EMfin] = EMBGAMP(Y, A) % suggested default
% [Xhat, EMfin, estHist, optEMfin] = EMBGAMP(Y, A, optEM, optGAMP) % full version
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
%   .maxEMiter          number of EM iterations [default=20]
%   .EMtol              EMBGAMP terminates when norm change between 
%                       sparsity rate is less than this tolerance. Set to
%                       -1 to turn off and run max iters [default=1e-5]
%   .learn_lambda       Set to true to learn lambda , set to false 
%                       to never update lambda (stays at initialization) 
%                       [default=true]
%   .learn_mean         Set to true to learn active mean (theta), set to false 
%                       to never update theta (stays at initialization) 
%                       [default=true]
%   .learn_var          Set to true to learn active variance (phi), set to 
%                       false to never update active variance (stays at  
%                       initialization) [default=true]
%   .learn_noisevar     Set to true to learn noise variance (psi), set to
%                       false to never update noise variance (stays at  
%                       initialization) [default=true]
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
%   .minVar             Set minimum GM-component variance [default 1e-5]
%   .robust_gamp        Toggle on when the sensing matrix gives issues 
%                       (e.g., correlated columns). This option 
%                       automatically overrides GAMP options to handle 
%                       these cases [default = false]
%
%   Note: EMBGAMP(Y, A) or EMBGAMP(Y, A, []) sets EM params at defaults.
%
%   WARNING! Set the following initializations only if confident they
%   accurately represent the model.  Otherwise, let the defaults preside.
%
%   .cmplx_in           Set to true if input is complex.  [default = 
%                       ~isreal(A^T*randn(M,1)) || ~isreal(Y)]
%   .cmplx_out          Set to true if output is complex.  [default = 
%                       ~isreal(Y)]
%   .lambda             Initialize overall sparsity rate lambda. Must be 
%                       either scalar or 1-by-T vector.  [default is 
%                       based on the LASSO phase transition curve]
%   .active_mean        Initialize mean of active components (theta)
%                       Must be scalar or 1-by-T vector.
%                       [Defaults set to the scalar 0]
%   .active_var         Initialize variance of active components (phi)
%                       Must be scalar or 1-by-T vector.
%                       [Defaults set to scalar calculated from 
%                       estimated signal variance]
%   .noise_var          Initialize the noise variance.  Must be either 
%                       scalar or 1-by-T vector [default is based on .SNRdB]
%
% optGAMP - structure containing various GAMP option fields [optional]
%   .nit                number of iterations
%   .pvarMin            minimum variance of each element of p
%   .XvarMin            minimum variance of each element of x
%   .step               step size 
%   .stepMin            minimum step size 
%   .stepMax            maximum step size
%   .stepIncr           Multiplicative step size increase, when successful
%   .stepDecr           Multiplicative step size decrease, when unsuccessful
%   .pvarStep           Logical flag to include a step size in the pvar/zvar calculation.
%   .varNorm            Option to "normalize" variances for computation.
%   .adaptStep          adaptive step size [important if A column-correlated]
%   .stepWindow         step size check window size
%   .bbStep             Barzilai Borwein step size [if A column-correlated]
%   .verbose            Print results in each iteration
%   .tol                Convergence tolerance
%   .stepTol            minimum allowed step size
%   .Avar               variance in A entries (may be scalar or matrix)
%   .xhat0              the initialization of x for GAMP [default handled
%                       internally]
%   .xvar0              the intialization of variance of each x_n for GAMP
%                       [default handled internally]
%   .shat0              the initialization of s for GAMP [default handled
%                       internally]
%
%   Note: EMBGAMP(Y, A) or EMBGAMP(Y, A, [], []) sets GAMP parameters at defaults
%
%Outputs:
% Xhat - GAMP-estimated posterior means
%
% EMfin               Contains various posterior quantities needed for EM
%                       updates of the signal and noise.
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
% estHist - GAMP output structure containing per-iteration GAMP data
%   .pass               Per iteration logical value indicating successful step
%   .step               Per-iteration stepsizes.
%   .stepMax            Per-iteration maximum allowed stepsizes.
%   .val                Per-iteration value of the GAMP cost function.
%
% optEMfin - Contains the final values of EMopt
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 6/14/13
% Change summary: 
%   v 1.0 (JV)- First release
%   v 1.1 (JV)- Now able to handle either explicit matrices A or GAMP
%               objects.
%   v 1.2 (JV)- Changed updates to noise variance.  Improves performance
%               in low SNRs.
%   v 1.3 (JV)- Allows learning/not learning of all parameters.  Default
%               initialization of active variance now accounts for
%               unnormalized A matrix.
%   v 2.0 (JV)- Accounts for MMV model.  Also made changes due to changes
%               in GAMP.
%   v 2.1 (JV)- Changed the way optGAMP is initialized.  Use must call
%               constructor first.
%   v 2.2 (JV)- Updated noise variance learning in accordance to new GAMP
%               outputs
%   v 3.0 (JV)- Added option to output GAMP Histories.
%   v 4.0 (JV)- Now outputs EM parameters Rhat, Rvar, Zhat, Zvar, pi, 
%               gamma, beta, and nu.
%   v 4.2 (JV)- Adjusted EM and GAMP defaults for enhanced speed and
%               robustness.  Also adjusted input/ouput format.
%   v 4.3 (JV)- optEM.heavy_tailed = true is now the default.  Also added
%               optEM.robust_gamp option to handle cases when matrix is 
%               problematic
%   v 4.4 (JV)- added optEM.sig_dim = 'row' and optEM.noise_dim = 'row' to
%               learn EM parameters on each row.
%
% Version 4.4
%
function [Xhat, EMfin, estHist, optEMfin] = EMBGAMP(Y, A, optEM, optGAMP)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
[M,~] = A.size();
if size(Y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end

%Set default GAMP options if unspecified
if nargin <= 2
    optEM = [];
end

if nargin <= 3
    optGAMP = [];
end

%Check inputs and initializations
[optGAMP, optEM] = check_opts(optGAMP, optEM);

optEM.L = 1;
%optEM.heavy_tailed = false;

histFlag = false;

if nargout >=3 ;
    histFlag = true;
end;

if histFlag
    [Xhat, EMfin, estHist, optEMfin] = EMGMAMP(Y, A, optEM, optGAMP);
else
    [Xhat, EMfin] = EMGMAMP(Y, A, optEM, optGAMP);
end

return;
