function results = trial_rpca(optIn)

%trial_rpca: This function runs several algorithms on a sample
%instance of the RPCA problem. The function can be run with no
%arguments. See the nargin==0 block below for the required fields to call
%the function with optional paramters. The function returns a structure of
%algorithm results.
%This code can be used to produce the results for the noise free phase
%plane plots for RPCA in the BiG-AMP arXiv submission. 
%(This code conducts a single trial at 1 point in the phase plane.)

%Add needed paths for RPCA examples
setup_RPCA

%Test case
if nargin == 0
    clc
    
    %Handle random seed
    defaultStream = RandStream.getGlobalStream;
    if 1
        savedState = defaultStream.State;
        save random_state.mat savedState;
    else
        load random_state.mat %#ok<UNRCH>
    end
    defaultStream.State = savedState;
    
    %Flags to test BiG-AMP variants
    optIn.tryBigamp = 1;
    optIn.tryBigampEM = 1;
    optIn.tryBigampEMcontract = 1;
    
    %Flags to turn on comparison algorithms
    optIn.tryGrasta = 0;
    optIn.tryInexactAlm = 0;
    optIn.tryVbrpca = 0;
    optIn.tryLmafit = 0;
    
    %%%%Inexact ALM search
    %%%%Uncomment this line to search over a grid of lambda values for Inexact
    %%%%ALM. Note that IALM-1 and IALM-2 are identical when this line is
    %%%%commented out
    %optIn.lambda_inexactAlm = 1/sqrt(optIn.M) * [1 logspace(-1,1,50)];
    
    %Problem Parameters
    optIn.M = 200;
    optIn.L = 200;%L >= M for some codes (doesn't matter for BiG-AMP)
    optIn.N = 10; %rank of low-rank component
    optIn.p1 = 1; %fraction of entries in Y observed (typically 1 for RPCA)
    optIn.lambda = 0.25; %fraction of entries corrupted with large outliers
    optIn.nuw = [0 20^2/12]; %First entry is AWGN, second entry is variance of large outliers
    
end

%% Problem Setup

%Flags for BiG-AMP
tryBigamp = optIn.tryBigamp;
tryBigampEM = optIn.tryBigampEM;
tryBigampEMcontract = optIn.tryBigampEMcontract;

%Flags for comparison algorithms
tryGrasta = optIn.tryGrasta;
tryInexactAlm = optIn.tryInexactAlm;
tryVbrpca = optIn.tryVbrpca;
tryLmafit = optIn.tryLmafit;

%Get lambda values for inexact ALM
if isfield(optIn,'lambda_inexactAlm')
    lambda_inexactAlm = optIn.lambda_inexactAlm;
else
    lambda_inexactAlm = 1/sqrt(optIn.M);
end

%Specify SNR
nuw = optIn.nuw;
lambda = optIn.lambda;

%Define the true matrix size (Z and Y are MxL)
M = optIn.M;
L = optIn.L;

%Specify N, which is the rank of Z in this setup
N = optIn.N;

%Specify the fraction of on entries in Y, which we shall dub p1
p1 = optIn.p1;

%Get BiG-AMP options
opt = BiGAMPOpt;

%Define the problem
problem = BiGAMPProblem();
problem.M = M;
problem.N = N;
problem.L = L;

%% Build true low rank matrix

%Compute true input vector
X = randn(N,L);

%Build true A
A = randn(M,N);

%Noise free signal
Z = A*X;


%% Form the output channel


%Noisy output channel
inds = rand(size(Z)) < lambda;

%Uniform errors
errorWidth = sqrt(12*nuw(2));
Y = Z +...
    sqrt(nuw(1))*randn(size(Z)) +...
    (-errorWidth/2 + errorWidth*rand(size(Z))).*inds;

%Build the perfect version of X2- the large outliers
X2 = zeros(size(Y));
X2(inds) = Y(inds) - Z(inds);

%Censor Y
omega = false(M,L);
ind = randperm(M*L);
omega(ind(1:ceil(p1*M*L))) = true;
Y(~omega) = 0;

%Store locations if any are omitted
if p1 < 1
    [problem.rowLocations,problem.columnLocations] = find(omega);
end

%Define the error function
error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
opt.error_function = error_function;


%% Establish the channel objects

%Prior on X
gX = AwgnEstimIn(zeros(size(X)), ones(size(X)));

%Prior on A
gA = AwgnEstimIn(zeros(size(A)), ones(size(A)));


%Output log likelihood
gOutBase = GaussMixEstimOut(Y,nuw(1),nuw(2),lambda);
gOut = MaskedEstimOut(gOutBase,omega);




%% Control initialization

%Use the initial values
opt.xhat0 = zeros(N,L);
opt.Ahat0 = randn(M,N);
opt.Avar0 = 10*ones(M,N);
opt.xvar0 = 10*ones(N,L);


%Initialize results as empty
results = [];



%% Switch to Non-adaptive

%Run with non-adaptive step for BiG-AMP-1
opt.stepMin = 0.25;
opt.stepMax = 0.25;
opt.adaptStep = 0;

%% Run BiGAMP-1


if tryBigamp
    
    %Allow up to 5 attempts
    failCounter = 0;
    tryAgain = 1;
    failTime = 0;
    
    while tryAgain
        
        %Increment fail counter
        failCounter = failCounter + 1;
        
        disp('Starting BiG-AMP-1')
        %Run BGAMP
        tstart = tic;
        [estFin,~,estHist] = ...
            BiGAMP(gX, gA, gOut, problem, opt); 
        tGAMP = toc(tstart);
        
        %Check to see if we have misconverged
        [~,~,p1] = gOutBase.estim(estFin.Ahat*estFin.xhat,estFin.pvar);
        
        if max(sum(p1)) > 0.8*M || max(sum(p1.')) > 0.8*L
            disp('Misconverged, trying again...')
            tryAgain = true;
            opt.Ahat0 = randn(size(opt.Ahat0));
            failTime = failTime + estHist.timing(end);
        else
            tryAgain = false;
        end
        
        %Stp after 5 attemps
        if failCounter >= 5
            tryAgain = false;
        end
    end
    
    %Add in the failed time
    estHist.timing = estHist.timing + failTime;
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'BiG-AMP-1'; %#ok<*AGROW>
    results{loc}.err = estHist.errZ(end);
    results{loc}.time = tGAMP;
    results{loc}.errHist = estHist.errZ;
    results{loc}.timeHist = estHist.timing;
    
end

%% Switch to adaptive

%After we run BiG-AMP-1, switch to adaptive step size for -2 and EM
opt.stepMin = 0.05;
opt.stepMax = 0.5;
opt.adaptStep = 1;

%% Specify Q

%Try a unitary matrix
Q = orth(randn(M));

%Define Q error function
error_functionQ = @(qval) 20*log10(norm(Q'*qval - Z,'fro') / norm(Z,'fro'));

%Change error function
opt.error_function = error_functionQ;

%% Run alternative BiGAMP


%Define gOut for this version
gOut2 = AwgnEstimOut(Q*Y,nuw(1));

%Make A2
A2 = MatrixLinTrans(Q);


%Create gX2
lambdaMatrix = lambda*ones(M,L);
lambdaMatrix(~omega) = 1; %enforce outliers in unknown locations
inputEst = AwgnEstimIn(0, nuw(2));
gX2 = SparseScaEstim(inputEst,lambdaMatrix);

%Create error function
opt.error_functionX2 =...
    @(q) 20*log10(norm(q(omega) - X2(omega),'fro')/norm(X2(omega),'fro'));

%Define initilizations
opt.x2hat0 = zeros(M,L);
opt.x2var0 = 10*lambda*nuw(2)*ones(M,L);
opt.xhat0 = zeros(N,L);
opt.xvar0 = 10*ones(N,L);
opt.Avar0 = 10*ones(M,N);

if tryBigamp
    
    %Allow up to 5 attempts
    failCounter = 0;
    tryAgain = 1;
    failTime = 0;
    
    while tryAgain
        
        %Increment fail counter
        failCounter = failCounter + 1;
        
        disp('Starting BiG-AMP-2')
        %Run BGAMP
        tstart = tic;
        [estFin2,~,estHist2] = ...
            BiGAMP_X2(gX, gA, gX2, A2, gOut2, problem, opt);
        tGAMP2 = toc(tstart);
        
        %Check to see if we have misconverged
        [~,~,~,p1] = gX2.estim(estFin2.r2hat,estFin2.r2var);
        
        if max(sum(p1)) > 0.8*M || max(sum(p1.')) > 0.8*L
            disp('Misconverged, trying again...')
            tryAgain = true;
            opt.Ahat0 = randn(size(opt.Ahat0));
            failTime = failTime + estHist2.timing(end);
        else
            tryAgain = false;
        end
        
        %Stp after 5 attemps
        if failCounter >= 5
            tryAgain = false;
        end
    end
    
    %Add in the failed time
    estHist2.timing = estHist2.timing + failTime;
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'BiG-AMP-2'; %#ok<*AGROW>
    results{loc}.err = estHist2.errZ(end);
    results{loc}.time = tGAMP2;
    results{loc}.errHist = estHist2.errZ;
    results{loc}.timeHist = estHist2.timing;
    
end

%% EM BiG AMP


if tryBigampEM
    
    %Silence
    opt.verbose = false;
    
    disp('Starting EM-BiG-AMP-2')
    %Run BGAMP
    tstart = tic;
    [estFinEM,~,~,estHistEM] = ...
        EMBiGAMP_RPCA(Y,A2,problem,opt); %#ok<ASGLU>
    tEMGAMP = toc(tstart);
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-BiG-AMP-2'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    
end

%% EM BiG AMP with rank contraction


if tryBigampEMcontract
    
    %Silence
    opt.verbose = false;
    
    %Enable rank learning with rank contraction
    EMopt.learnRank = true;
    EMopt.rankMax = 90;

    
    disp('Starting EM-BiG-AMP-2 with rank contraction')
    %Run BGAMP
    tstart = tic;
    [estFinEM,~,~,estHistEM] = ...
        EMBiGAMP_RPCA(Y,A2,problem,opt,EMopt); %#ok<ASGLU>
    tEMGAMP = toc(tstart);
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-BiG-AMP-2 (Rank Contraction)'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    results{loc}.rank = size(estFin.xhat,1);
end

%% Try LMaFit

if tryLmafit
    
    %Inform user
    disp('Starting LMaFit')
    
    %Build LMaFit options
    Lmafit_opts.tol = opt.tol;
    Lmafit_opts.maxit = 6000;
    
    Lmafit_opts.est_rank = 0; %don't estimate rank
    
    %Do it
    tstart = tic;
    [Almafit,Xlmafit,~,~,timingLmafit,estHistLmafit] = lmafit_sms_v1_timing(Y,N,Lmafit_opts,[],error_function);
    tLmafit = toc(tstart);
    
    %Compute error
    ZhatLMaFit = Almafit*Xlmafit;
    errLMaFit = 20*log10(norm(ZhatLMaFit(:) - Z(:)) / norm(Z(:)));
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'LMaFit'; %#ok<*AGROW>
    results{loc}.err = errLMaFit;
    results{loc}.time = tLmafit;
    results{loc}.errHist = estHistLmafit.errZ;
    results{loc}.timeHist = timingLmafit;
    
end

%% Try GRASTA

if tryGrasta
    disp('Starting GRASTA')
    
    maxCycles                   = 20;    % the max cycles of robust mc
    OPTIONS.QUIET               = opt.verbose;     % suppress the debug information
    
    OPTIONS.MAX_LEVEL           = 20;    % For multi-level step-size,
    OPTIONS.MAX_MU              = 15;    % For multi-level step-size
    OPTIONS.MIN_MU              = 1;     % For multi-level step-size
    
    OPTIONS.DIM_M               = M;  % your data's ambient dimension
    OPTIONS.RANK                = N; % give your estimated rank
    
    OPTIONS.ITER_MIN            = 20;    % the min iteration allowed for ADMM at the beginning
    OPTIONS.ITER_MAX            = 20;    % the max iteration allowed for ADMM
    OPTIONS.rho                 = 2;   % ADMM penalty parameter for acclerated convergence
    OPTIONS.TOL                 = 1e-8;   % ADMM convergence tolerance
    OPTIONS.stopTol             = opt.tol; %stop tolerance
    OPTIONS.USE_MEX             = 0;     % If you do not have the mex-version of Alg 2
    % please set Use_mex = 0.
    CONVERGE_LEVEL              = 20;    % If status.level >= CONVERGE_LEVLE, robust mc converges
    
    
    %Build the inputs
    [I,J] = find(omega);
    S = reshape(Y(omega),[],1);
    
    %Call it
    tstart = tic;
    [Usg, Vsg, ~,timingGrasta,estHistGrasta] =...
        grasta_mc_timing(I,J,S,M,L,maxCycles,CONVERGE_LEVEL,OPTIONS,error_function);
    tGrasta = toc(tstart);
    
    
    
    %Compute Zhat
    ZhatGrasta = Usg*Vsg';
    
    %Compute error
    errGrasta = 20*log10(norm(ZhatGrasta(:) - Z(:)) / norm(Z(:)));
    
    
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'GRASTA'; %#ok<*AGROW>
    results{loc}.err = errGrasta;
    results{loc}.time = tGrasta;
    results{loc}.errHist = estHistGrasta.errZ;
    results{loc}.timeHist = timingGrasta;
    
end

%% Inexact Alm


if tryInexactAlm
    
    
    %Check if we are running with the IALM-1 lambda
    magicLam = sqrt(1/M);
    [magicError,magicLoc] = min(abs(magicLam - lambda_inexactAlm));
    if magicError < 1e-3*magicLam
        magicFlag = 1;
    else
        magicFlag = 0;
    end
    
    
    for counter = 1:length(lambda_inexactAlm)
        display('Starting Inexact ALM')
        %Call it
        %         tstart = tic;
        %         [~,~,~,timingInexactAlm,estHistInexactAlm] =...
        %             inexact_alm_rpca_timing(Y, lambda_inexactAlm(counter), opt.tol, 200);
        %         tInexactAlm = toc(tstart);
        tstart = tic;
        [~,~,~,timingInexactAlm,estHistInexactAlm] =...
            inexact_alm_rpca_tasos_timing(...
            Y, lambda_inexactAlm(counter), opt.tol, 200,N,error_function);
        tInexactAlm = toc(tstart);
        
        
        
        %Compute best error
        errInexactAlm = estHistInexactAlm.errZ(end);
        
        %Save it
        inexactAlmResults{counter}.errInexactAlm = errInexactAlm; %#ok<AGROW>
        inexactAlmResults{counter}.tInexactAlm = tInexactAlm;%#ok<AGROW>
        inexactAlmResults{counter}.errInexactAlmHist = estHistInexactAlm.errZ;%#ok<AGROW>
        inexactAlmResults{counter}.tInexactAlmHist = timingInexactAlm;%#ok<AGROW>
        
    end
    
    %Store best result in the structure
    myfun = @(q) q.errInexactAlm;
    yada = cellfun(myfun,inexactAlmResults);
    [~,best_lambda] = min(yada);
    
    
    %Save the results for IALM-1 if it was run
    if magicFlag
        loc = length(results) + 1;
        results{loc}.name = 'IALM-1'; %#ok<*AGROW>
        results{loc}.err = inexactAlmResults{magicLoc}.errInexactAlm;
        results{loc}.time = inexactAlmResults{magicLoc}.tInexactAlm;
        results{loc}.errHist = inexactAlmResults{magicLoc}.errInexactAlmHist;
        results{loc}.timeHist = inexactAlmResults{magicLoc}.tInexactAlmHist;
    end
    
    %Save results for IALM-2
    finalTimes = cellfun(@(q) q.tInexactAlmHist(end),inexactAlmResults);
    
    loc = length(results) + 1;
    results{loc}.name = 'IALM-2'; %#ok<*AGROW>
    results{loc}.err = inexactAlmResults{best_lambda}.errInexactAlm;
    results{loc}.time = sum(cellfun(@(q) q.tInexactAlm,inexactAlmResults));
    results{loc}.errHist = inexactAlmResults{best_lambda}.errInexactAlmHist;
    %Add times for other lambda values
    results{loc}.timeHist = inexactAlmResults{best_lambda}.tInexactAlmHist...
        + sum(finalTimes) - finalTimes(best_lambda);
    
    
end

%% Try VBRPCA

if tryVbrpca
    options.thr = opt.tol;
    options.verbose = opt.verbose;
    options.initial_rank = N; % or we can use a value.
    options.DIMRED = 0; %don't allow it to reduce dimensions
    options.inf_flag = 2; % inference flag for the sparse component
    % 1 for standard VB, 2 for MacKay. MacKay generally converges faster.
    options.MAXITER = 300;
    %Estimate noise variance? (beta is inverse noise variance)
    options.UPDATE_BETA = 1;
    % If the noise inv. variance is not to be estimated, set
    % options.UPDATE_BETA = 0; % and set beta using
    %options.beta = 1/nuw(1);
    
    % Select the optimization mode:
    % 'VB': fully Bayesian inference (default)
    % 'VB_app': fully Bayesian with covariance approximation
    % 'MAP': maximum a posteriori (covariance is set to 0)
    options.mode = 'VB';
    
    %Turn this on to enable random init.
    %options.init = 'rand';
    
    %Run it
    disp('Starting VBRPCA');
    tstart = tic;
    [timingVbrpca,estHistVbrpca,Zvbrpca,~,~,X2vbrpca] =...
        VBRPCA_timing(Y, options,error_function);
    tVbrpca = toc(tstart);
    
    %Display errors
    if opt.verbose
        X1_error_vbrpca = error_function(Zvbrpca) %#ok<NOPRT,NASGU>
        X2_error_vbrpca = opt.error_functionX2(X2vbrpca) %#ok<NOPRT,NASGU>
    end
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'VSBL'; %#ok<*AGROW>
    results{loc}.err = estHistVbrpca.errZ(end);
    results{loc}.time = tVbrpca;
    results{loc}.errHist = estHistVbrpca.errZ;
    results{loc}.timeHist = timingVbrpca;
    
end



%% Store the options structures in results
results{1}.optIn = optIn;



%% Show Results

if nargin == 0
    
    %Plot results
    plotUtilityNew(results,[-80 0],200,201)
    
    %Show results
    results{:} %#ok<NOPRT>
end

