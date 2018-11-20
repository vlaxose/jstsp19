function results = trial_matrixCompletion_UnknownRank(optIn)

%trial_matrixCompletion_UnknownRank: This function runs several algorithms on a sample
%instance of the matrix completion problem. The function can be run with no
%arguments. See the nargin==0 block below for the required fields to call
%the function with optional paramters. The function returns a structure of
%algorithm results. This version assumes that the underlying matrix rank is
%unknown by the algorithms under test. Note that the settings for all
%algorithms could probably be adjusted to reduce run time. Large iteration
%counts are currently specified. 
%This code can be used to produce the results for the Approximately Low
%Rank matrix completion examples in the BiG-AMP arXiv submission. 

%Add needed paths for MC examples
setup_MC

%Test case
if nargin == 0
    clc
    
    %Handle random seed
    defaultStream = RandStream.getGlobalStream;
    if 1 %change to zero to try the same random draw repeatedly
        savedState = defaultStream.State;
        save random_state.mat savedState;
    else
        load random_state.mat %#ok<UNRCH>
    end
    defaultStream.State = savedState;
    
    
    %Handle random seed
    defaultStream = RandStream.getGlobalStream;
    if 1 %change to zero to try the same random draw repeatedly
        savedState = defaultStream.State;
        save random_state.mat savedState;
    else
        load random_state.mat %#ok<UNRCH>
    end
    defaultStream.State = savedState;
    
    %Flags to test BiG-AMP variants
    optIn.tryBigampEM = 1;
    
    %Flags to turn on comparison algorithms
    optIn.tryLmafit = 1;
    optIn.tryVbmc = 1;
    
    %Problem parameters
    optIn.SNR = 20; %Signal to Noise Ratio in dB (set to inf for noise free)
    optIn.M = 500; %Data matrix is M x L
    optIn.L = 500; %L >= M for some codes (doesn't matter for BiG-AMP)
    optIn.N = 500; %Inner dimension, true rank set by type of problem
    optIn.p1 = 0.2; %fraction of observed entries
    
    %Specify type of problem
    optIn.type = 'exponential'; %exponential decaying singular values
    %optIn.type = 'power'; %power lay decaying singular values
    %optIn.type = 4; %Set type to an integer for a matrix of fixed rank
end

%% Problem Setup

%Turn OptSpace on/off
tryLmafit = optIn.tryLmafit;
tryVbmc = optIn.tryVbmc;
tryBigampEM = optIn.tryBigampEM;


%Specify SNR
SNR = optIn.SNR;

%Define the true matrix size (Z and Y are MxL)
M = optIn.M;
L = optIn.L;

%Specify N, which is the rank of Z in this setup
N = optIn.N;

%Specify the fraction of on entries in Y, which we shall dub p1
p1 = optIn.p1;


%Create Options object
opt = BiGAMPOpt();
opt.nit = 100; %limit iterations



%Override EM defaults
EMopt.maxEMiter = 200;
EMopt.maxEMiterInner = 5;
EMopt.learnRank = true;
EMopt.rankStep = 1;

%NOTE: The maximum rank was limited to 30 for this experiment to limit
%computation time of the various tests across all the methods. If this
%parameter is not set, then EM-BiG-AMP will automatically select a
%reasonable maximum, which may be larger than 30 and appropriate for some
%applications. When copying this example, this parameter may need to be
%omitted or adjusted depending on the application.
EMopt.rankMax = 30;




%% Build the data based on problem type


%Generate full rank matrices
if N < min(M,L)
    U = orth(randn(M,N));
    V = orth(randn(L,N)).';
else
    U = orth(randn(M));
    V = orth(randn(L));
end

%Choose singular values
switch optIn.type
    
    case 'exponential',
        svec(1:M) = exp(-0.3*(1:M));
        
    case 'power',
        svec(1:M) = (1:M).^(-3);
        
        
    otherwise,
        if ~isnumeric(optIn.type)
            error('must be numeric')
        end
        
        %Round it
        optIn.type = round(optIn.type);
        svec = zeros(M,1);
        svec(1:optIn.type) = 1;
end

%Build A and X based on specified singular values
X = sqrt(diag(svec))*V;
A = U*sqrt(diag(svec));
Z = A*X;

%Define the error function
error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
opt.error_function = error_function;


%Determine nuw
nuw = norm(reshape(Z,[],1))^2/M/L*10^(-SNR/10);

%Noisy output channel
Y = Z + sqrt(nuw)*randn(size(Z));

%Censor Y
omega = false(M,L);
ind = randperm(M*L);
omega(ind(1:ceil(p1*M*L))) = true;
Y(~omega) = 0;

%Specify the problem setup for BiG-AMP, including the matrix dimensions and
%sampling locations. Notice that the rank N can be learned by the EM code
%and does not need to be provided in that case. We set it here for use by
%the low-level codes which assume a known rank
problem = BiGAMPProblem();
problem.M = M;
problem.N = 1;
problem.L = L;
[problem.rowLocations,problem.columnLocations] = find(omega);

%Initialize results as empty
results = [];





%% Try VBMC

if tryVbmc
    
    options.verbose = 1;
    options.MAXITER = 100;
    options.DIMRED = 1; % Reduce dimensionality during iterations?
    % you can also set the threshold to reduce dimensions
    options.DIMRED_THR = 1e3;
    %Estimate noise variance? (beta is inverse noise variance)
    %options.UPDATE_BETA = 1;
    % options.UPDATE_BETA_START = 1;% iteration number to start estimating noise variance
    % If the noise inv. variance is not to be estimated, set
    options.UPDATE_BETA = 1; % and set beta using
    %options.beta = 1e9;
    % Manually tuning this parameter can give significantly better results.
    options.thr = opt.tol;
    
    % options.initial_rank = 'auto'; % This sets to the maximum possible rank
    options.initial_rank = EMopt.rankMax; % or we can set a value.
    %options.init = 'rand';
    
    
    
    %Run it
    disp('Starting VBMC');
    tstart = tic;
    [ZVbmc,timingVbmc,estHistVbmc] = VBMC_timing(omega, Y, options,error_function);
    tVbmc = toc(tstart);
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'VSBL';
    results{loc}.err = estHistVbmc.errZ(end);
    results{loc}.time = tVbmc;
    results{loc}.errHist = estHistVbmc.errZ;
    results{loc}.timeHist = timingVbmc;
    %results{loc}.rank = sum(svd(ZVbmc) > 1e-4*max(svd(ZVbmc)));
    results{loc}.rank = rank(ZVbmc);
end





%% Try LMaFit

if tryLmafit
    
    %Inform user
    disp('Starting LMaFit')
    
    %Get input data
    Known = find(omega);
    data = reshape(Y(omega),[],1);
    
    %Options- these are the values from the LMaFit paper
    Lmafit_opts.est_rank = 2; %fixed rank
    Lmafit_opts.tol = 1e-4;%
    Lmafit_opts.print = 1; %no printout
    Lmafit_opts.maxit = 1000;
    Lmafit_opts.rk_inc = 1;
    Lmafit_opts.rank_max = EMopt.rankMax; %impose rank limit, different from theirs
    
    %Do it
    tstart = tic;
    [XLMaFit,YLMaFit,~,timingLmafit,estHistLmafit] =...
        lmafit_mc_adp_timing(M,L,1,Known,data,Lmafit_opts,error_function); %#ok<ASGLU>
    tLmafit = toc(tstart);
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'LMaFit';
    results{loc}.err = estHistLmafit.errZ(end);
    results{loc}.time = tLmafit;
    results{loc}.errHist = estHistLmafit.errZ;
    results{loc}.timeHist = timingLmafit;
    results{loc}.rank = size(XLMaFit,2);
    disp(['LMaFit Error was ' num2str(results{loc}.err)])
end



%% Try EM-BiGAMP


if tryBigampEM
    
    
    
    disp('Starting EM-BiG-AMP')
    %Run BGAMP
    tstart = tic;
    [estFin,~,~,estHistEM] = ...
        EMBiGAMP_MC(Y,problem,opt,EMopt);
    tGAMP = toc(tstart);
    
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = opt.error_function(estFin.Ahat*estFin.xhat);
    results{loc}.time = tGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    results{loc}.rank = size(estFin.xhat,1);
    
end





%% Store the options structures in results
results{1}.optIn = optIn;





%% Show Results

if nargin == 0
    
    %Show error plots
    plotUtilityNew(results,[-100 0],200,201)
    
 
    %Show results
    results{:}  %#ok<NOPRT>
    
end


