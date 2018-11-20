%% Hyperspectral Unmixing via turbo Approximate Message Passing
% Given the hyperspectral data Y of size M by  T (where T = T_1 * T_2), we 
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
% Syntax 
%   [estFin] = HUTAMP(Y, N) or, for full inputs and outputs
%   [estFin, stateFin, estHist] = HUTAMP(Y, N, optALG, optBiGAMP)
%
% Inputs
%  Y            The HUT data cube of size M by T_1 by T_2.
%  N            Number of materials present.  If unknown, assume to be
%               large [default = 5]
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

function [estFin, stateFin, estHist]...
    = HUTAMP(Y, N, optALG, optBiGAMP)

%Find and set dimensions of problem
[M, T1, T2] = size(Y);

if T2 == 1
    T = T1;
else
    T = T1*T2;
end

Y = reshape(Y,M,T);
mn = mean(Y(:));

%Define indeces of non-psuedo row
I = 1:M;

if nargin <=1
    N = 5;
end

%Preset all GAMP defaults
if nargin <= 2
    optALG = [];
end

%Preset all EM defaults
if nargin <= 3
    optBiGAMP = [];
end

%Set default BiGAMP options
[optALG, optBiGAMP] = set_opts_HUTAMP(optALG, optBiGAMP);

%Set default HUTAMP parameters
[Y, Shat, stateFin] = EndExt(Y, N, optALG);

%Initialize the abundances.  This is important!
optBiGAMP.xhat0 = pinv([Shat;ones(1,N)])*[Y;ones(1,T)];
%optBiGAMP.xhat0 = (1./N).*ones(N,T);
optBiGAMP.xvar0 = 10*ones(N,T);

%% Set HUT BiGAMP defaults and parameter initializations
problem = BiGAMPProblem();
problem.N = N;
problem.L = T;
problem.M = M;

scale = 5.*mn;
%Define input and output estimators
gOutArray = cell(2,1);
gOutArray{1} = AwgnEstimOut(Y, stateFin.noise_var);
gOutArray{2} = DiracEstimOut(scale*ones(1,T));
gOut = EstimOutConcat(gOutArray,[M;1]);

%gOut = AwgnEstimOut(Y, stateFin.noise_var);

% %Define input channels
gA = NNGMEstimIn(stateFin.active_weights, stateFin.active_loc, stateFin.active_scales);
gA = SparseScaEstim(gA, stateFin.lambda);
gSArray = cell(2,1);
% Fix endmembers on first BiGAMP iterations
% gSArray{1} = DiracEstimIn(Shat);
gSArray{1} = DiracEstimIn(Shat);
%gSArray{1} = AwgnEstimIn(Shat,1e-4);
%Ensure the last row is not learned and is a row vector of 1's
gSArray{2} = DiracEstimIn(scale*ones(1,N));
gS = EstimInConcat(gSArray,[M;1]);

%Perform BiGAMP
[estFin, ~, estHist, state] = ...
    BiGAMP(gA, gS, gOut, problem, optBiGAMP);

stateFin = set_inits_HUTAMP(Y, Shat, estFin.xhat, N, optALG, stateFin);

%Define noiseless measurements
Zhat = estFin.Ahat*estFin.xhat;
lambda_in = stateFin.lambda;

% if no spectral coherence is exhibited, initialize parameters to something
% non-informative
if ~optALG.spectralCoherence
    eta_in = 0;
    kappa_in = 1;
end

%Initialize loop
t = 0; %iteration
stop = 0; %stop condition
while stop == 0;
%% Perform GAMP portions of algorithm

    %Check to see if max iterations hit
    t = t + 1;
    if t > optALG.maxEMiter
        break
    end  
    
    % If returns a single variance expand to matrix sizes
    if optBiGAMP.uniformVariance
        estFin.qvar = repmat(estFin.qvar,M+1,N);
        estFin.rvar = repmat(estFin.rvar,N,T);
    end
   
%% Perform Gauss Markov Message Passing for endmembers

    %Perform the Gauss Markov message Passing if enabled
    if optALG.spectralCoherence
        [eta_in, kappa_in, stateFin] ...
            = Gauss_Markov(estFin.qhat(I,:), estFin.qvar(I,:), stateFin, optALG);
        kappa_in = max(kappa_in,1e-14);
    end

    %Perform Markov Random Field Message Passing if enabled
    if optALG.spatialCoherence
        %Find posterior parameters on abundances
        beta = zeros(N,T,optALG.L); gamma = zeros(N,T,optALG.L);
        nu = zeros(N,T,optALG.L);

        %Compute quantities needed for extrinsic activity probabilities
        for i = 1:optALG.L
            dummy = stateFin.active_scales(:,:,i) + estFin.rvar + eps;
            beta(:,:,i) = sqrt(estFin.rvar./dummy).*stateFin.active_weights(:,:,i)...
             .*exp(((estFin.rhat-stateFin.active_loc(:,:,i))...
             .^2./abs(dummy)-abs(estFin.rhat).^2./estFin.rvar).*(-.5));
            gamma(:,:,i) = (estFin.rvar.*stateFin.active_loc(:,:,i)...
                +stateFin.active_scales(:,:,i).*estFin.rhat)./dummy;
            nu(:,:,i) = stateFin.active_scales(:,:,i).*estFin.rvar./dummy;
        end;
        alpha = -gamma./sqrt(nu);

        %Compute log-likelihood ratio for activity probability
        LLR = log(sum(beta.*erfc(alpha/sqrt(2))...
            ./erfc(-stateFin.active_loc./sqrt(stateFin.active_scales)/sqrt(2)),3));
        LLR = max(-20,min(20,LLR));
        %Compute extrinsic posterior activity likelihoods
        lambda_out = 1./(1+ exp(-LLR));

        %Perform MRF to leverage spatial coherence
        for i = 1:N
            tmp = reshape(lambda_out(i,:),T1,T2);
            [tmp, stateFin.betaH(i), stateFin.betaV(i), stateFin.alpha(i)] = ...
                    SPD_MRF(tmp, stateFin.betaH(i), ...
                    stateFin.betaV(i), stateFin.alpha(i), 8, optALG);
            lambda_in(i,:) = reshape(tmp,1,T);
        end
        [stateFin] = abun_update(estFin.rhat, estFin.rvar, stateFin, optALG); 
    else
        %If no-spatial coherence, do EM learning for all enabled parameters
        [stateFin] = abun_update(estFin.rhat, estFin.rvar, stateFin, optALG); 
        lambda_in = stateFin.lambda;
    end
    
    % Update noise variance
    if optALG.learn_noisevar
        if ~optALG.maxBethe
         stateFin.noise_var = sum(sum((Y-Zhat(1:end-1,:)).^2))/M/T ...
             +sum(sum(estFin.zvar(1:end-1,:)))/M/T;
        else
 
        Ybar2 = (Y - Zhat(1:end-1,:)).^2;                  %compute quadratic terms
        pvarNot = estFin.Avar(1:end-1,:)*estFin.xvar;    %compute pvar - pvarBar
        %Perform update of AWGN noise variance per band
        %Minimizes the Bethe free energy at the fixed point
        stateFin.noise_var = sum(pvarNot + 0.5.*Ybar2 + 0.5.*sqrt(Ybar2.^2 ...
                 + 4.*(estFin.pvar(1:end-1,:) - pvarNot).*Ybar2),2)./T;
        stateFin.noise_var = resize(stateFin.noise_var,M,T);
        end
    end
    
    

%% ReRun BiGAMP using new messages
    
    %Define input and output estimators
    gOutArray = cell(2,1);
    gOutArray{1} = AwgnEstimOut(Y, stateFin.noise_var);
    gOutArray{2} = DiracEstimOut(scale*ones(1,T));
    gOut = EstimOutConcat(gOutArray,[M;1]);

    %Define input channels
    gA = NNGMEstimIn(stateFin.active_weights, stateFin.active_loc, stateFin.active_scales);
    gA = SparseScaEstim(gA, lambda_in);
    gSArray = cell(2,1);
    gSArray{1} = AwgnEstimIn(eta_in, kappa_in);
    %Ensure the last row is not learned and is a row vector of 1's
    gSArray{2} = DiracEstimIn(scale*ones(1,N));
    gS = EstimInConcat(gSArray,[M;1]);
    
    %Warm start the BiGAMP variables
%     optBiGAMP.xhat0 = estFin.xhat;
%     optBiGAMP.Ahat0 = estFin.Ahat;
%     optBiGAMP.xvar0 = estFin.xvar;
%     optBiGAMP.Avar0 = estFin.Avar;

    %Perform BiGAMP
    [estFin, ~, estHist2, state] = ...
        BiGAMP(gA, gS, gOut, problem, optBiGAMP, state);
    Zhat2 = estFin.Ahat*estFin.xhat;
    
    %Update Histories
    estHist.errZ = [estHist.errZ; estHist2.errZ];
    estHist.errX = [estHist.errX; estHist2.errX];
    estHist.errA = [estHist.errA; estHist2.errA];
    estHist.val = [estHist.val; estHist2.val];
    estHist.step = [estHist.step; estHist2.step];
    estHist.pass = [estHist.pass; estHist2.pass];
    estHist.timing = [estHist.timing; estHist2.timing];
    
    %Calculate the change in signal estimates
    norm_change = norm(Zhat-Zhat2,'fro')^2/norm(Zhat,'fro')^2;
    
    %Reinitialize GAMP estimates
    Zhat = Zhat2;

    %Check for estimate tolerance threshold
    if norm_change < optALG.EMtol
        stop = 1;
    end
    
end

%% Output final estimates
estFin.Shat = estFin.Ahat(1:M,:) + mn;
estFin.Svar = estFin.Avar(1:M,:);
estFin.Ahat = estFin.xhat;
estFin.Avar = estFin.xvar;
estFin = rmfield(estFin,{'xhat','xvar'});

estHist.errS = estHist.errA;
estHist.errA = estHist.errX;
estHist = rmfield(estHist,'errX');

%% Output final parameters
stateFin.specMean = stateFin.specMean(1,:) + mn;
stateFin.specVar = stateFin.specVar(1,:);
stateFin.specCorr = stateFin.specCorr(1,:);

stateFin.active_weights = reshape(stateFin.active_weights(:,1,:),N,optALG.L)';
stateFin.active_loc = reshape(stateFin.active_loc(:,1,:),N,optALG.L).';
stateFin.active_scales = reshape(stateFin.active_scales(:,1,:),N,optALG.L)';
stateFin.lambda = lambda_in;
stateFin.noise_var = stateFin.noise_var(:,1);

return