function [estFin,optFin,EMoptFin,estHist]...
    = EMBiGAMP_RPCA(Y, A2, problem, BiGAMPopt, EMopt)

%EMBiGAMP_RPCA:  Expectation-Maximization Bilinear Generalized AMP for RPCA
%
%EM-BiG-AMP tunes the parameters of the distributions on A, X, and Y|Z
%assumed by BiG-AMP using the EM algorithm. This version of the code
%assumes that the entries of A are N(0,1), while the entries of X are
%Gaussian with zero mean and variance to be learned. The outliers are
%modeled using the X2 component and are assumed to be Bernouli-Gaussian with
%zero mean, activation rate to be learned, and active variance to be learned.
%The noise is assumed to be AWGN with unknown variance. Optionally, a rank
%contraction procedure can be used to learn the rank of the underlying low
%rank component. This function is an example. Similar procedures with
%different choices for the priors can be created with minimal changes to
%the code.
%
% INPUTS:
% -------
% Y: the noisy data matrix. May be passed as a full matrix, a sparse matrix
%   as in sparse(Y), or as a vector containing the observed entries. The
%   locations of these entries are defined in the rowLocations and
%   columnLocations fields of the problem object. (Sparse mode is not yet
%   implemented for RPCA. May be supported in a future release)
% A2: The fixed matrix used to pre-multiply the outliers. The data model is
%     A2*Y = A*X + A2*X2. The matrix A2 should be unitary and can be passed
%     as either a matrix or an object of the LinTrans class
% problem: An objet of the class BiGAMPProblem specifying the problem
%   setup, including the matrix dimensions and observation locations
% opt (optional):  A set of options of the class BiGAMPOpt.
% EMopt (optional): A structure containing several fields. If the input is
%   omitted or empty, the values in defaultOptions defined below will be
%   used. The user may pass a structure containing any or all of these fields
%   to replace the defaults. Fields not included in the user input will be
%   replaced with the defaults
%
% OUTPUTS:
% --------
% estFin: Structure containing final BiG-AMP outputs
% optFin: The BiGAMPOpt object used
% EMoptFin: The EM options used
% estHist: Structure containing per iteration metrics about the run


%% Handle Options

%Create options if not provided
if (nargin < 4)
    BiGAMPopt = [];
end

%Handle empty options object
if (isempty(BiGAMPopt))
    BiGAMPopt = BiGAMPOpt();
end


%If no inputs provided, set user options to empty
if nargin < 5
    EMopt = [];
end

%Get problem dimensions
M = problem.M;
L = problem.L;
N = problem.N;


%Initial values (note that the mean of the X entries is initialized
%with zero)
defaultOptions.noise_var = []; %initial noise variance estimate
defaultOptions.active_var = []; %initial variance estimate of active X elements

%Options to control learning
defaultOptions.learn_noisevar= true; %learn the variance of the AWGN
defaultOptions.learn_var = true; %learn the variance of the X entries
defaultOptions.learn_mean = false; %learn the mean of the X entries
defaultOptions.sig_dim = 'joint'; %learn a single variances for X entries (joint)
%or a different variance per (row) or (column)

%Options to control learning of the outliers
defaultOptions.learn_x2var = 1;
defaultOptions.learn_x2lambda = 1;

%Iteration control
defaultOptions.maxEMiter = 20; %maximum number of EM cycles
defaultOptions.tmax = 20; %first EM iteration to use full expression for noise variance update

%Tolerances
defaultOptions.EMtol = BiGAMPopt.tol; %convergence tolerance
defaultOptions.maxTol = 1e-4;%largest allowed tolerance for a single EM iteration

%Rank learning options
if ~isempty(N)
    defaultOptions.learnRank = false; %if N is provided, by default we do not try to learn the rank
else
    defaultOptions.learnRank = true; %default to rank contraction.
end
defaultOptions.rank_tau = 1.5; %Minimum ratio of singular value gap to average before allowing
%rank to contract
defaultOptions.rankMax = 50; %currently hard coded for RPCA. User may wish to adjust
defaultOptions.nitFirstEM = 50; %Number of BiG-AMP iterations allowed for the first EM iteration.
%Only used for rank learning

%thresholds for detecting bad rows/columns
defaultOptions.Lthresh = 0.8;
defaultOptions.Mthresh = 0.8;


%Combine user options with defaults. Fields not provided will be given
%default values
EMopt = checkOptions(defaultOptions,EMopt);

%% Initial Setup

%Correct N when doing rank learning
if EMopt.learnRank
    N = EMopt.rankMax;
    problem.N = N;
end

%Indices of observed entries
if ~isempty(problem.rowLocations)
    omega = sub2ind([M L],problem.rowLocations,problem.columnLocations);
else
    omega = true(M,L);
end

%Sparse mode is not yet supported for this setup. Note that some of the
%code is in the file for future use, but additional modifications are
%needed to enable it
if BiGAMPopt.sparseMode
    error('Sparse mode is not yet supported')
end

%Create matrix sparse multiply if needed
if BiGAMPopt.sparseMode
    sMult = @(arg1,arg2) sparseMult2(arg1.',arg2,...
        BiGAMPopt.rowLocations,BiGAMPopt.columnLocations);
    
    %Ensure that Y is proper dimensions
    if numel(Y) > numel(BiGAMPopt.rowLocations)
        Y = reshape(Y(omega),1,[]);
    end
end


%Find values above the median
bigSet = abs(Y) > median(abs(Y(:)));

%Set noise variance if requested
if ~isfield(EMopt,'noise_var') || isempty(EMopt.noise_var)
    if ~BiGAMPopt.sparseMode
        EMopt.noise_var = sum(sum(abs(Y(omega & ~bigSet)).^2))/(sum(sum(omega & ~bigSet))*101);
    else
        %Not implemented
    end
end


%History
histFlag = false;
if nargout >=4
    histFlag = true;
    estHist.errZ = [];
    estHist.errX = [];
    estHist.errX2 = [];
    estHist.errA = [];
    estHist.val = [];
    estHist.step = [];
    estHist.pass = [];
    estHist.timing = [];
end;

%Set initial noise variance
nuw = EMopt.noise_var;
meanX = zeros(N,L);

%Set lambda to 1
lambda = ones(N,L);

%Compute X variance, setting nuA = 1
if ~isfield(EMopt,'active_var') || isempty(EMopt.active_var)
    nuX = (norm(Y(~bigSet),'fro')^2/sum(sum(~bigSet)) - nuw)/N;
else
    nuX = EMopt.active_var;
end


%Compute X variance, setting nuA = 1
if ~isfield(EMopt,'tmax') || isempty(EMopt.tmax)
    tmax = EMopt.maxEMiter;
else
    tmax = EMopt.tmax;
end

%Setup initial X2 params
nuX2 = norm(Y(bigSet),'fro')^2/sum(sum(bigSet));
lambdaX2 = 0.1*ones(M,L);
meanX2 = zeros(M,L);

%Initialize loop
t = 0;
stop = 0;
failCounter = 0;

%Initialize xhat and Ahat
%xhat = sqrt(nuX)*randn(N,L);
xhat = zeros(N,L); %seems to perform better
Ahat = randn(M,N);
x2hat = zeros(M,L);

%Set them
BiGAMPopt.xhat0 = xhat;
BiGAMPopt.Ahat0 = Ahat;
BiGAMPopt.Avar0 = 10*ones(M,N);
BiGAMPopt.xvar0 = 10*nuX.*ones(N,L);
BiGAMPopt.x2hat0 = x2hat;
BiGAMPopt.x2var0 = 10*lambdaX2.*nuX2.*ones(M,L);

%Ensure that diagnostics are off to save run time
BiGAMPopt.diagnostics = 0;

%Ensure that EM outputs are calculated
BiGAMPopt.saveEM = 1;

%Original tolerance
tol0 = BiGAMPopt.tol;


%Inits
SNR = 100;
zhatOld = 0;

%Set A2
if isnumeric(A2)
    A2 = MatrixLinTrans(A2);
end

%Build options for learning X and X2
Xopts = EMopt;
Xopts.learn_lambda = false;
X2opts = EMopt;
X2opts.learn_var = EMopt.learn_x2var;
X2opts.learn_mean = 0; %mean learning off for now
X2opts.learn_lambda = EMopt.learn_x2lambda;

%Save iteration count
nitSafe = BiGAMPopt.nit;

%% Main Loop

%EM iterations
%The < 2 condition is so that we can do a final iteration using the full
%noise varaince update without repeating code
while stop < 2
    
    %Start timing
    tstart = tic;
    
    %Estimate SNR
    SNRold = SNR;
    if t > 0
        SNR = norm(zhat,'fro')^2 / norm(Y - zhat,'fro')^2;
    end
    
    %Set tolerance
    %tolNew = 1 / SNR;
    tolNew = min(max(tol0,1/SNR),EMopt.maxTol);
    BiGAMPopt.tol = tolNew;
    
    %Increment time exit loop if exceeds maximum time
    t = t + 1;
    if t >= EMopt.maxEMiter || stop > 0
        stop = stop + 1;
    end
    
    %Set iteration limits
    if (t == 1) && (EMopt.learnRank)
        BiGAMPopt.nit = EMopt.nitFirstEM;
    else
        BiGAMPopt.nit = nitSafe;
    end
    
    %Prior on A
    gA = AwgnEstimIn(0, 1);
    
    %Prior on X
    gX = AwgnEstimIn(meanX, nuX);
    
    %Prior on X2
    %lambdaMatrix(~omega) = 1; %enforce outliers in unknown locations
    inputEst = AwgnEstimIn(0, nuX2);
    gX2 = SparseScaEstim(inputEst,lambdaX2);
    
    %Output log likelihood
    gOut = AwgnEstimOut(A2.mult(Y), nuw);
    
    %Stop timing
    t1 = toc(tstart);
    
    
    %Run BiG-AMP
    [estFin2,~,estHist2,state] = ...
        BiGAMP_X2(gX,gA,gX2,A2,gOut,problem,BiGAMPopt);
    
    %Start timing
    tstart = tic;
    
    %Correct cost function
    estHist2.val = estHist2.val -0.5*numel(omega)*log(2*pi*nuw);
    
    
    %Report progress
    if histFlag
        error_value = estHist2.errZ(end);
    else
        error_value = nan;
    end
    disp(['It ' num2str(t,'%04d')...
        ' nuX = ' num2str(mean(nuX(:)),'%5.3e')...
        ' nuX2 = ' num2str(mean(nuX2(:)),'%5.3e')...
        ' Lam = ' num2str(mean(lambdaX2(:)),'%0.2f')...
        ' tol = ' num2str(BiGAMPopt.tol,'%5.3e')...
        ' SNR = ' num2str(10*log10(SNR),'%03.2f')...
        ' Z_e = ' num2str(error_value,'%05.4f')...
        ' X2_e = ' num2str(estHist2.errX2(end),'%05.4f')...
        ' numIt = ' num2str(length(estHist2.errZ),'%04d')])
    
    %Compute zhat
    if BiGAMPopt.sparseMode
        zhat = sMult(estFin2.Ahat,estFin2.xhat);
    else
        zhat = (A2.multTr(estFin2.Ahat*estFin2.xhat)) + estFin2.x2hat;
    end
    
    
    %Calculate the change in signal estimates
    norm_change = norm(zhat-zhatOld,'fro')^2/norm(zhat,'fro')^2;
    
    %Check for estimate tolerance threshold
    if (norm_change < max(tolNew/10,EMopt.EMtol)) &&...
            ( (norm_change < EMopt.EMtol) ||...
            (abs(10*log10(SNRold) - 10*log10(SNR)) < 1))
        stop = stop + 1;
    end
    
    %Update noise variance. Include only a portion of the Zvar
    %in beginning stages of EMGMAMP because true update may make it
    %unstable.
    %Learn noise variance
    if EMopt.learn_noisevar
        
        %First, just the term based on the residual
        if BiGAMPopt.sparseMode
            nuw = norm(Y - zhat,'fro')^2 ...
                /numel(omega);
        else
            
            nuw = norm(Y(omega) - zhat(omega),'fro')^2 ...
                /numel(omega);
        end
        
        %Then the component based on zvar
        if t >= tmax || stop > 0
            if isscalar(estFin2.zvar)
                nuw = nuw + estFin2.zvar;
            else
                if BiGAMPopt.sparseMode
                    nuw = nuw + sum(estFin2.zvar)/numel(omega);
                else
                    nuw = nuw + sum(estFin2.zvar(omega))/numel(omega);
                end
            end
        end
    end
    
    %Estimate new X parameters
    [lambda, meanX, nuX] =...
        BG_update(estFin2.rhat, estFin2.rvar,...
        lambda, meanX, nuX, Xopts);
    
    %Estimate new X2 parameters
    [lambdaX2,meanX2,nuX2]=...
        BG_update(estFin2.r2hat, estFin2.r2var,...
        lambdaX2, meanX2, nuX2, X2opts);
    
    
    
    %Check to see if we have misconverged
    [~,~,~,p1] = gX2.estim(estFin2.r2hat,estFin2.r2var);
    stuckState = (max(sum(p1)) > EMopt.Mthresh*M || ...
        max(sum(p1.')) > EMopt.Lthresh*L);
    
    %Don't stop if stuck and t is less than 3
    if stuckState && t <= 3
        stop = 0;
    end
    
    %Check for problems
    if t > 3 && stuckState && failCounter < 5
        disp('Converged to bad state, restarting')
        failCounter = failCounter + 1;
        stop = 0;
        estFin2.xhat = zeros(N,L);
        estFin2.Ahat = randn(size(estFin2.Ahat));
        estFin2.xvar = nuX;
        estFin2.Avar = ones(size(estFin2.Avar));
        estFin2.x2hat = zeros(size(estFin2.x2hat));
        estFin2.x2var = nuX2;
        state.shatOpt = [];
    elseif EMopt.learnRank
        
        %Check for rank update
        Nhat = heuristicRank(estFin2.xhat,EMopt.rank_tau);
        
        %If a rank was accepted and it is at least a decrease of 2.
        %Occasionally, the algorithm will incorrectly decide to truncate
        %the rank by 1, so we do not allow it to do this
        if ~isempty(Nhat) && (Nhat < N - 1)
            
            %Inform user
            %if verbose
            disp(['Updating rank estimate from ' num2str(N) ...
                ' to ' num2str(Nhat) ' on iteration ' num2str(t)])
            %end
            
            %Update rank
            N = Nhat;
            problem.N = N;
            
            %Truncate signals
            estFin2.xhat = estFin2.xhat(1:N,:);
            estFin2.xvar = estFin2.xvar(1:N,:);
            estFin2.Ahat = estFin2.Ahat(:,1:N);
            estFin2.Avar = estFin2.Avar(:,1:N);
            lambda = lambda(1:N,:);
            meanX = meanX(1:N,:);
            nuX = nuX(1:N,:);
            
            %Reset X2
            estFin2.x2hat = zeros(M,L);
            estFin2.x2var = nuX2;
            
            %Only do this once
            EMopt.learnRank = false;
            
        end
        
    end
    
    
    %Reinitialize GAMP estimates
    zhatOld = zhat;
    BiGAMPopt.xhat0 = estFin2.xhat;
    BiGAMPopt.xvar0 = estFin2.xvar;
    BiGAMPopt.shat0 = state.shatOpt;
    BiGAMPopt.Ahat0 = estFin2.Ahat;
    BiGAMPopt.Avar0 = estFin2.Avar;
    BiGAMPopt.x2hat0 = estFin2.x2hat;
    BiGAMPopt.x2var0 = estFin2.x2var;
    BiGAMPopt.step = BiGAMPopt.stepMin;
    
    
    %Stop timing
    t2 = toc(tstart);
    
    %Output Histories if necessary
    if histFlag
        estHist.errZ = [estHist.errZ; estHist2.errZ];
        estHist.errX = [estHist.errX; estHist2.errX];
        estHist.errX2 = [estHist.errX2; estHist2.errX2];
        estHist.errA = [estHist.errA; estHist2.errA];
        estHist.val = [estHist.val; estHist2.val];
        estHist.step = [estHist.step; estHist2.step];
        estHist.pass = [estHist.pass; estHist2.pass];
        if t == 1
            estHist.timing = [estHist.timing; t1 + t2 + estHist2.timing];
        else
            estHist.timing = [estHist.timing; t1 + t2 + estHist.timing(end) ...
                + estHist2.timing];
        end
    end
    
end

%% Cleanup

%Update finals
estFin = estFin2;
optFin = BiGAMPopt;
EMoptFin = EMopt;




