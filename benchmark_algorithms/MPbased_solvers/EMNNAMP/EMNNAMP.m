%EMNNAMP Expectation Maximization Non-Negative Approximate Message Passing
% From the M-by-T matrix of noisy linear observations Y = AX + W, EMNNAMP
% returns (MAP or MMSE) estimates of the N-by-T signal matrix X, enforcing
% the known non-negativity constraint X > 0, and P-by-T linear equality
% constraints BX = C.  When B and C are row vectors of ones, we say that 
% the columns of X lie on the simplex, hence the name of the function.
%
% EMNNAMP offers three modes of operation: Non-negative Least squares GAMP 
% (NNLS-AMP), non-negative Lasso GAMP (NNLAMP), and NN Gaussian mixture 
% GAMP (NNGMAMP).  Under NNLS-AMP, the marginals of X are assumed to be the
% improper uniform distribution on [0, infty), i.e.,
%
%   p_X(x) = 1 if x >= 0
%            0 if x < 0.
%
% Under NNL-AMP, it is assumed that the coeeficients of X are drawn
% i.i.d. from the exponential pdf, i.e.,
%
% p(X(n,t)) = lambda*exp(-lambda|X(n,t)|) x >= 0
%             0                           x < 0,
%
% where lambda is the exponential rate parameter.  In NNL-AMP, this rate 
% parameter is learned through EM.
%
% Under NNGM-AMP, it is assumed that the coefficients of X are
% drawn i.i.d. from the bernoulli non-negative Gaussian mixture pdf, i.e.,
%
%   p(X(n,t)) = (1-tau(n,t))*delta(X(n,t))
%                + tau(n,t) sum_k^L [omega(t,k)
%               Normal(X(n,t);theta(n,t,k),phi(n,t,k))
%                / Phi_c(-theta(n,t,k)/ sqrt(phi(n,t,k))]     x >= 0
%               0                                             x < 0,   
%
% where n = 1:N indexes the row of the signal matrix X,  t = 1:T
% indexes the column (timestep) of all matrices, delta() denotes the
% Dirac delta pdf, and Phi_c denotes the complimentary cdf of the N(0,1) 
% distribution.  Above, tau(n,t), omega(n,t,:), theta(n,t,:), and phi(n,t,:)
% are the sparsity rate, active weights, active locations, and
% active scales, respectively, of the coefficients within the 
% n-th row and t-th column of X.  
%
% In the expectation maximization (EM) versions of NNL-AMP and NNGM-AMP the
% free parameters are learned across the rows, columns, or jointly.
%
% In EMNNAMP (NNLSAMP, NNLAMP, NNGMAMP), the noise samples W(m,t) are 
% assumed apriori independent and zero-mean Gaussian distributed:
%
%    p(W(m,t)) = Normal(W(m,t);0,psi(m,t))   m = 1:M
%
% where m = 1:M indexes the rows of the matrices W and Y, and psi(t)
% is the variance within the t-th column of W. In the EM version, the noise 
% variance is also learned through EM. To enforce the P linear
% equality constraints, we augment the matrix via A_aug = [A; B] and the
% measurements as Y_aug = [Y; C], and assume the augmented output prior
%
%    p(W(m,t)) = delta(W(m,t))   m = M+1:P
%
% For heavy-tailed output channels (or measurements with outliers), we 
% recommend assuming that the noise samples W(m,t) are independent and 
% identically distributed as the Laplcian pdf:
%
%    p(W(m,t)) = psi/2*exp(-psi abs(w(m,t))   m = 1:M,
%
% where psi is the Laplacian rate parameter.
%
% The estimate of X is calculated using the Generalized Approximate Message
% Passing (GAMP) algorithm, as described in "Generalized Approximate Message
% Passing for Estimation with Random Linear Mixing" by Sundeep Rangan,
% with certain adaptive-stepsize enhancements.
%
% The EMNNAMP algorithm is described in detail in "An Empirical Bayes
% Approach to Recovering Linearly Constrained Non-Negative Sparse Signals" 
% by Jeremy Vila and Philip Schniter.
%
%Syntax:
% [Xhat, stateFin] = EMNNAMP(Y, A)
% [Xhat, stateFin, estHist, optEMFin] = EMNNAMP(Y, A, optALG, optEM, optGAMP)
%                                         %allows customization
%
%Inputs:
% Y - matrix of observed measurements
%
% A - known mixing matrix (in either explicit or GAMP-object format)
%
% optGAMP - structure containing various GAMP option fields [optional].
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
%   Note: EMNNAMP(Y, A) or EMNNAMP(Y, A, [], []) sets GAMP parameters at defaults
%
% optALG- structure contraining various algorithm options
%   .alg_type           Set to 'NNLSAMP','NNGMAMP', or 'NNLAMP' 
%                       [default = 'NNGMAMP']
%   .laplace_noise      Set to true to assume the Laplacian noise
%                       distribution [default = false]
%   .linEqMat           The linear matrix needed for equality constraint, 
%                       i.e., 'B'.  If no constraints exist set to []. 
%                       [default = [] for NN case]
%   .linEqMeas          The linear measurements needed for equality
%                       constraint, i.e., 'C' [default = []]
%   .LEtol              Stopping tolerance for the linear equality
%                       constraints. [default= 1e-8];
%
% optEM - structure containing various EM option fields [optional]
%   .SNRdB              dB SNR used to initialize noise variance [default=20]
%   .maxEMiter          number of EM iterations [default=200]
%   .EMtol              EMNNAMP terminates when norm change between
%                       sparsity stateFin.outLapRate is less than this tolerance. Set to
%                       -1 to turn off and run max iters [default=1e-5]
%   .learn_tau          Set to true to learn sparsity rate, set to false
%                       to never update sparsity rate (stays at 
%                       initialization) [default=true]
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
%   .learn_outLapRate   Set to true to learn Laplacian stateFin.outLapRate (kappa), set to
%                       false to never update Laplacian stateFin.outLapRate (stays at
%                       initialization) [default=true]
%   .learn_inExpRate    Set to true to learn the exponential rate (lambda), set to
%                       false to never update exponential rate (stays at
%                       initialization) [default=true]
%   .hiddenZ            Set to true to do noise variance learning assuming
%                       hidden Z variables.  Set to false to use hidden X
%                       variables [default = false]
%   .maxBethe           Set to true to learn the output parameters via
%                       maximization of the Bethe Free Energy [default =
%                       true]
%   .inDim              Set to 'col' to allow different EM-learned parameters
%                       for each column in signal matrix X.  Set to 'row' 
%                       to allow different EM-learned parameters for each 
%                       row in signal matrix X. Set to 'joint' to force 
%                       common EM-learned parameters for all rows and 
%                       columns. [default='joint']
%   .outDim             Set to 'col' to allow different EM-learned noise
%                       stateFins for each column in matrix Y.  Set to 'joint'
%                       to force common EM-learned stateFineters for all columns.
%                       [default='joint']
%   .minScale           Minimum value for scales when in EMNNGMAMP mode
%                       [default = 1e-10]
%
%   Note: EMNNAMP(Y, A) or EMNNAMP(Y, A, [], []) sets EM params at defaults
%
%   WARNING! Set the following initializations only if confident they
%   accurately represent the model.  Otherwise, let the defaults preside.
%   Also, ensure that all parameter vectors are of compatible lengths
%   depending on the inDim and OutDim options.
%
%   .L                  Number of mixture components [default=3, max=20]
%   .tau                Initialize overall sparsity rate tau. Must be
%                       either scalar or 1-by-T vector.  [default is
%                       based on the LASSO phase transition curve]
%   .active_weights     Initialize NNGM component active_weights (omega).
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .active_loc         Initialize mean of active components (theta)
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .active_scales      Initialize variance of active components (phi)
%                       Must be L-by-1 or L-by-T vector.
%                       [Defaults based on best L-term uniform-pdf fit]
%   .inExpRate          Rate for the Exponential distribution [default = 1]
%   .noise_var          Initialize the noise variance.  Must be either
%                       scalar or 1-by-T vector [default is based on an
%                       assumed SNR = 20 dB]
%   .outLapRate         outLapRate parameter for the Laplacian
%                       distribution (when enabled) [default = 1]
%
%Outputs:
% Xhat - GAMP-estimate of X (MAP for NNLSAMP and NNLAMP or approximate MMSE
%        for NNGMAMP)
% stateFin - structure containing various parameters and GAMP quantities
%   .tau                Sparsity rate (if alg_type = 'NNGMAMP')
%                       [1-by-T for 'col' mode or 1-by-N for 'row mode']
%   .active_weights     active_weights of NNGM (if alg_type = 'NNGMAMP')
%                       [L-by-T for 'col' mode or L-by-N for 'row mode]
%   .active_loc         Locations of active components (if alg_type = 'NNGMAMP')
%                       [L-by-T for 'col' mode or L-by-N for 'row mode]
%   .active_scales      Scales of active components (if alg_type = 'NNGMAMP')
%                       [L-by-T for 'col' mode or L-by-N for 'row mode]
%   .inExpRate          Exponential rate parameter (if alg_type = 'NNLAMP')
%                       [1-by-T for 'col' mode or 1-by-N for 'row mode']
%   .noise_var          Variance of noise (if laplace_noise = false)
%                       [1-by-T for 'col' mode or 1-by-M for 'row mode']
%   .outLapRate         Laplacian Rate (if laplace_noise = true)
%                       [1-by-T for 'col' mode or 1-by-M for 'row mode']
%   .SNRdB              The learned SNR in dB.  
%                       [1-by-T for 'col' mode or 1-by-M for 'row mode']
%   .Xvar               GAMP quantity
%   .Rhat               GAMP quantity
%   .Rvar               GAMP quantity
%   .Zhat               GAMP quantity
%   .Zvar               GAMP quantity
%   .Phat               GAMP quantity
%   .Pvar               GAMP quantity
%
% estHist - GAMP output structure containing per-iteration GAMP data.  If
%           it is not called, MATLAB does not preallocate memory to the
%           structure, thus reducing complexity.
%   .pass               Per iteration logical value indicating whether 
%                       adaptive stepsize selection was used.
%   .step               Per-iteration step sizes.
%   .val                Per-iteration value of the GAMP cost function.
%
% optEMfin - Contains the final values of EMopt 
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 10/13/14
% Change summary:
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Updated output channel parameter estimation procedure.
%   v 2.1 (JV)- Added option to learn output parameters using hidden Z or X
%               variables.
%   v 2.2 (JV)- Allowed noise variance learning via maximization of bethe
%               free energy
%   v 2.3 (JV)- Improved method to append history.  Changed default to
%               noise variance learning to maxBEthe = true.
%
% Version 2.3
%

function [Xhat, stateFin, estHist, optEMFin] = EMNNAMP(Y, A, optALG, optEM, optGAMP)

%Ensure that matrix is GAMP object
if ~isobject(A)
    A = MatrixLinTrans(A); 
end

%Find problem dimensions
[M,N] = A.size();
T = size(Y, 2);
if size(Y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end

% Set default algorithm options
if nargin <= 2 || isempty(optALG)
    optALG = [];
end

% Set default EM options
if nargin <= 3 || isempty(optEM)
    optEM = [];
end

%Set default GAMP options
if nargin <= 4  || isempty(optGAMP)
    optGAMP = [];
end

% Whatevere options are not specified by users, set to defaults
[optALG, optEM, optGAMP] = set_opts_EMNNAMP(optALG, optEM, optGAMP, N, T);

histFlag = false;

if nargout >=3
    histFlag = true;
end;

%test to see if user supplies empty constraints or measurements.
%If so, do not augment the model
emptyFlag = false;
if isempty(optALG.linEqMat) || isempty(optALG.linEqMeas)
    emptyFlag = true;
end

if emptyFlag
    P = 0;
else
    %get size of augmented row
    P = size(optALG.linEqMeas,1);

    %if simplex gamp shift the mean of matrix to zero
    tmp = optALG.linEqMat(:,1);
    if repmat(tmp,1,N) - optALG.linEqMat == zeros(P,N);
        mn = sum(A.mult(ones(N,1)))/M/N;

        A = A.mult(eye(N)) - mn;
        A = MatrixLinTrans(A); 

        Y = Y - mn;
    end
end

%Redefine appended sensing matrix and measurements for linear equality constraints
if ~emptyFlag
    AugA = LinTransConcat({A;optALG.linEqMat});
end

MAPGAMP = false;
if strcmp(optALG.alg_type,'NNLSAMP')
    MAPGAMP = true;
    optEM.learn_noisevar = false;
    optEM.learn_outLapRate = false; 
elseif strcmp(optALG.alg_type,'NNLAMP')
    MAPGAMP = true;
end

%% Perform NNLS-GAMP, EM-NNL-GAMP, or EM-NNGM-GAMP  

%Set initializations
stateFin = set_inits_EMNNAMP(Y, A, optALG, optEM);

%% Define input channel
%Defin non-informative uniform distribution
if strcmp(optALG.alg_type,'NNLSAMP')
    gX = UnifEstimIn(0,Inf);
%Define NN laplacian input
elseif strcmp(optALG.alg_type,'NNLAMP')
    gX = NNSoftThreshEstimIn(stateFin.inExpRate, 1);
%Define Bernoulli NNGM input
elseif strcmp(optALG.alg_type,'NNGMAMP')
    %stateFin.active_scales = max(optEM.minScale,stateFin.active_scales);
    gX = NNGMEstimIn(stateFin.active_weights, stateFin.active_loc, stateFin.active_scales);
    gX = SparseScaEstim(gX, stateFin.tau);
else 
    error('Invalid algorithm type.  set optALG.alg_type = NNLSAMP, NNGMAMP, or NNLAMP');
end

%% Define output channel
% No linear equality constraints
if emptyFlag
    if ~optALG.laplace_noise
        gOut =  AwgnEstimOut(Y, stateFin.noise_var, MAPGAMP);
    else 
        gOut = LaplaceEstimOut(Y, stateFin.outLapRate, MAPGAMP);
    end

    %Perform EMNNAMP
    if ~histFlag
        estFin = gampEst(gX, gOut, A, optGAMP);
    else
        [estFin, ~, estHist] = gampEst(gX, gOut, A, optGAMP);
    end

    %use linear equality constraints
else
    gOutArray = cell(2,1);
    if ~optALG.laplace_noise
        gOutArray{1} = AwgnEstimOut(Y, stateFin.noise_var, MAPGAMP);
    else
        gOutArray{1} = LaplaceEstimOut(Y, stateFin.outLapRate, MAPGAMP);
    end
    gOutArray{2} = DiracEstimOut(optALG.linEqMeas,1);
    gOut = EstimOutConcat(gOutArray,[M;P]);

%% Perform EMNNAMP
    if ~histFlag
        estFin = gampEst(gX, gOut, AugA, optGAMP);
    else
        [estFin, ~, estHist] = gampEst(gX, gOut, AugA, optGAMP);
    end
end

Xhat = estFin.xhat;
Xvar = estFin.xvar;
Rhat = estFin.rhat;
Rvar = estFin.rvar;
%If doing EM learning of noise with hidden Z use these variables
if ~optEM.maxBethe
    if optEM.hiddenZ
        Zhat = estFin.zhat;
        Zvar = estFin.zvar;
    %If doing EM learning of noise with hidden X use these variables 
    else
        Zhat = A.mult(Xhat);
        Zvar = A.multSq(Xvar);
    end
else 
    Zhat = estFin.shat(1:end-P,:);
    Zvar = estFin.svar(1:end-P,:);
end

XhatPrev = Xhat;
firstStep = optGAMP.step;

lastIter = 0;
t = 0;
stop = 0;
while stop == 0;
%% Perform GAMP portions of algorithm

    %Check to see if amx iterations hit
    t = t + 1;
    if t > optEM.maxEMiter
        break
    end
    
    %% Find EM update for input and output parameters
    %Update input channel parameters
    stateFin = EMNNAMP_in_update(Rhat, Rvar, stateFin, optALG, optEM); 

    %Update output parameters
    stateFin = EMNNAMP_out_update(Y, Zhat, Zvar, stateFin, optALG, optEM, lastIter);
    
    %warm start GAMP
%      optGAMP = optGAMP.warmStart(estFin);
    optGAMP.xhat0 = Xhat;
    optGAMP.xvar0 = Xvar;
    optGAMP.shat0 = estFin.shat;
    optGAMP.svar0 = estFin.svar;
    optGAMP.xhatPrev0 = estFin.xhatPrev;
    optGAMP.scaleFac = estFin.scaleFac;
    optGAMP.step = min(max(estFin.step,firstStep),estFin.stepMax);
    optGAMP.stepMax = estFin.stepMax;

    
    %% Redefine input channels
    %Define new NN laplacian input
    if strcmp(optALG.alg_type,'NNLAMP')
        gX = NNSoftThreshEstimIn(stateFin.inExpRate,1);
    %Define new Bernoulli NNGM input
    elseif strcmp(optALG.alg_type,'NNGMAMP')
        gX = NNGMEstimIn(stateFin.active_weights, stateFin.active_loc, stateFin.active_scales);
        gX = SparseScaEstim(gX, stateFin.tau);
    end

    %% Redefine output channels
    %No linear equality constraints
    if emptyFlag
        if ~optALG.laplace_noise
            gOut =  AwgnEstimOut(Y, stateFin.noise_var, MAPGAMP);
        else 
            gOut = LaplaceEstimOut(Y, stateFin.outLapRate, MAPGAMP);
        end

        %Perform EMNNAMP
        if ~histFlag
            estFin = gampEst(gX, gOut, A, optGAMP);
        else
            [estFin, ~, estHistNew] = gampEst(gX, gOut, A, optGAMP);
        end
    %linear equality constraints
    else
        %Define output channel
        gOutArray = cell(2,1);
        if ~optALG.laplace_noise
            gOutArray{1} = AwgnEstimOut(Y, stateFin.noise_var, MAPGAMP);
        else
            gOutArray{1} = LaplaceEstimOut(Y, stateFin.outLapRate, MAPGAMP);
        end
        gOutArray{2} = DiracEstimOut(optALG.linEqMeas, 1);
        gOut = EstimOutConcat(gOutArray,[M;P]);

        %Perform EMNNAMP
        if ~histFlag
            estFin = gampEst(gX, gOut, AugA, optGAMP);
        else
            [estFin, ~, estHistNew] = gampEst(gX, gOut, AugA, optGAMP);
        end
    end

    %update gamp quantities
    if histFlag
        estHist = appendEstHist(estHist,estHistNew);
    end

    Xhat = estFin.xhat;
    Xvar = estFin.xvar;
    Rhat = estFin.rhat;
    Rvar = estFin.rvar;
    %If doing EM learning of noise with hidden Z use these variables
    if ~optEM.maxBethe
        if optEM.hiddenZ
            Zhat = estFin.zhat;
            Zvar = estFin.zvar;
        %If doing EM learning of noise with hidden X use these variables 
        else
            Zhat = A.mult(Xhat);
            Zvar = A.multSq(Xvar);
        end
    else
        Zhat = estFin.shat(1:end-P,:);
        Zvar = estFin.svar(1:end-P,:);
    end


    %Calculate the change in signal estimates
    norm_change = norm(Xhat-XhatPrev,'fro')^2/norm(Xhat,'fro')^2;
    if emptyFlag
        LE_norm = -Inf;
    else
        LE_norm = norm(optALG.linEqMeas - optALG.linEqMat*Xhat)^2/P;
    end
    
    if lastIter
        stop = 1;
    end

    %Check for estimate tolerance threshold
    if norm_change < optEM.EMtol && LE_norm < optALG.LEtol
        lastIter = 1;
    end

    %Reinitialize GAMP estimates
    XhatPrev = Xhat;

end
    
%% Reformat parameters
% output final input parameters
if strcmp(optALG.alg_type,'NNGMAMP')
    L = size(stateFin.active_weights,3);
    if strcmp(optEM.inDim,'joint')
        stateFin.tau = stateFin.tau(1,1);
        stateFin.active_weights = reshape(stateFin.active_weights(1,1,:),1,L)';
        stateFin.active_loc = reshape(stateFin.active_loc(1,1,:),1,L).';
        stateFin.active_scales = reshape(stateFin.active_scales(1,1,:),1,L)';
    elseif strcmp(optEM.inDim,'col')
        stateFin.tau = stateFin.tau(1,:);
        stateFin.active_weights = reshape(stateFin.active_weights(1,:,:),T,L)';
        stateFin.active_loc = reshape(stateFin.active_loc(1,:,:),T,L).';
        stateFin.active_scales = reshape(stateFin.active_scales(1,:,:),T,L)';
    elseif strcmp(optEM.inDim,'row')
        stateFin.tau = stateFin.tau(:,1);
        stateFin.active_weights = reshape(stateFin.active_weights(:,1,:),N,L)';
        stateFin.active_loc = reshape(stateFin.active_loc(:,1,:),N,L).';
        stateFin.active_scales = reshape(stateFin.active_scales(:,1,:),N,L)';
    end
elseif strcmp(optALG.alg_type,'NNLAMP')
    if strcmp(optEM.inDim,'joint')
        stateFin.inExpRate = stateFin.inExpRate(1,1);
    elseif strcmp(optEM.inDim,'col')
        stateFin.inExpRate = stateFin.inExpRate(1,:);
    elseif strcmp(optEM.inDim,'row')
        stateFin.inExpRate = stateFin.inExpRate(:,1);
    end
end

%output final output parameters
if optALG.laplace_noise
    if strcmp(optEM.outDim,'joint')
        stateFin.outLapRate = stateFin.outLapRate(1,1);
    elseif strcmp(optEM.outDim,'col')
        stateFin.outLapRate = stateFin.outLapRate(1,:);
    elseif strcmp(optEM.outDim,'row')
        stateFin.outLapRate = stateFin.outLapRate(:,1);
    end
else
    if strcmp(optEM.outDim,'joint')
        stateFin.noise_var = stateFin.noise_var(1,1);
    elseif strcmp(optEM.outDim,'col')
        stateFin.noise_var = stateFin.noise_var(1,:);
    elseif strcmp(optEM.outDim,'row')
        stateFin.noise_var = stateFin.noise_var(:,1);
    end
end

%Output final GAMPstates
stateFin.Xvar = estFin.xvar;
stateFin.Zhat = estFin.zhat(1:end-P,:);
stateFin.Zvar = estFin.zvar(1:end-P,:);
stateFin.Rhat = estFin.rhat;
stateFin.Rvar = estFin.rvar;
stateFin.Phat = estFin.phat(1:end-P,:);
stateFin.Pvar = estFin.pvar(1:end-P,:);
if histFlag
    estHist.phat = estHist.phat(1:end-P,:);
    estHist.pvar = estHist.pvar(1:end-P,:);
    estHist.zhat = estHist.zhat(1:end-P,:);
    estHist.zvar = estHist.zvar(1:end-P,:);
    estHist.shat = estHist.shat(1:end-P,:);
    estHist.svar = estHist.svar(1:end-P,:);
end

%Output final options
optEMFin = optEM;

return