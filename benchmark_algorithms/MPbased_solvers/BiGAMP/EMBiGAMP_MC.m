function [estFin,optFin,EMoptFin,estHist]...
    = EMBiGAMP_MC(Y, problem, BiGAMPopt, EMopt)

%EMBiGAMP_MC:  Expectation-Maximization Bilinear Generalized AMP for matrix
%completion.
%
%EM-BiG-AMP tunes the parameters of the distributions on A, X, and Y|Z
%assumed by BiG-AMP using the EM algorithm. This version of the code
%assumes that the entries of A are N(0,1), while the entries of X are
%Gaussian with variance (and possibly mean) to be learned. The noise is
%assumed to be AWGN with unknown variance. Optionally, a procedure based on
%AICc may be used to learn the underlying matrix rank. This function is an
%example. Similar procedures with different choices for the priors can be
%created with minimal changes to the code.
%
% INPUTS:
% -------
% Y: the noisy data matrix. May be passed as a full matrix, a sparse matrix
%   as in sparse(Y), or as a vector containing the observed entries. The
%   locations of these entries are defined in the rowLocations and
%   columnLocations fields of the problem object.
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

%Get problem dimensions from problem object
M = problem.M;
L = problem.L;
N = problem.N;

%Indices of observed entries
if ~isempty(problem.rowLocations)
    omega = sub2ind([M L],problem.rowLocations,problem.columnLocations);
else
    warning('Sampling locations not specified- assuming all entries are observed.') %#ok<WNTAG>
    omega = 1:(M*L);
end
Ns = numel(omega); %number of measurements

%Create options if not provided
if (nargin < 3)
    BiGAMPopt = [];
end

%Handle empty options object
if (isempty(BiGAMPopt))
    BiGAMPopt = BiGAMPOpt();
    
    %Use sparse mode for low sampling rates
    if Ns / (M*L) < 0.2
        BiGAMPopt.sparseMode = true;
    end
end


%Check sparse mode option
if numel(Y) ~= M*L
    if (numel(Y) == numel(omega))
        BiGAMPopt.sparseMode = true;
    else
        error('Size of Y is not consistent. Must either provide the full matrix or the set of observed entries')
    end
end

%Determine the maximum rank that can be uniquely determined from the number
%of provided measurements (based on the number of degrees of freedom in the
%SVD of a rank N MxL matrix)
p1 = Ns/(M*L);
Nmax1 = (M + L - sqrt((M+L)^2 - 4*p1*M*L))/2;
Nmax2 = (M + L + sqrt((M+L)^2 - 4*p1*M*L))/2;
Nmax = min(Nmax1,Nmax2);

%Define default values
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

%Iteration control. Note that for rank method 2, only a single set of EM
%iterations is run. Thus, the difference between maxEMiter and
%maxEMiterInner is only relevant for rank method 1. In methods 0 or 2, the
%maximum possible number of EM iterations will be maxEMiterInner.
defaultOptions.maxEMiter = 20; %maximum number of EM cycles
defaultOptions.maxEMiterInner = 20; %maximum number of EM iterations for a given rank estimate
defaultOptions.tmax = 20; %first EM iteration to use full expression for noise variance update
%typically works well to set this equal to maxEMiter
%for MC



%Tolerances
defaultOptions.EMtol = BiGAMPopt.tol; %convergence tolerance
defaultOptions.maxTol = 1e-4;%largest allowed tolerance for a single EM iteration

%Rank learning mode
%Set to 0 to disable rank learning
%Set to 1 to use the AICc rank learning method. This approach tends to be
%more robust, particularly when the singular values of the true matrix tail
%off smoothly.
%Set to 2 for rank contraction. This approach works well when the singular
%values of the true matrix exhibit a distinct gap. It is also generally
%much faster than method 1.
if ~isempty(N)
    defaultOptions.learnRank = 0; %if N is provided, by default we do not try to learn the rank
else
    defaultOptions.learnRank = 2; %default to rank contraction.
end

%This option applies in both modes 1 and 2. In mode 1, this is the maximum
%the rank can increase to. In mode 2, this is the starting rank. Note that
%in mode 2 BiGAMPopt.N gets overriden, while in mode 1 BiGAMPopt.N is used
%as the starting rank.
%By default, the maximum rank is set to the maximum rank for which the
%matrix can be uniquely determined based on the number of provided
%measurements
defaultOptions.rankMax = floor(Nmax); %maximum allowed rank
defaultOptions.nitFirstEM = 50; %Number of BiG-AMP iterations allowed for the first EM iteration.
%Only used in mode 2!

%Optionts for rank method 1 (no effect in methods 0 or 2)
defaultOptions.rankStep = 1; %amount to increase rank
defaultOptions.rankStart = 1; %initial rank for rank expansion

%Options for rank method 2 (no effect in methods 0 or 1)
defaultOptions.rank_tau = 1.5; %Minimum ratio of singular value gap to average
%before allowing rank to contract

%If no inputs provided, set user options to empty
if nargin < 4
    EMopt = [];
end

%Combine user options with defaults. Fields not provided will be given
%default values
EMopt = checkOptions(defaultOptions,EMopt);


%% Initial Setup

%Correct N when in mode 1. We start at the initial rank in this mode
if EMopt.learnRank == 1
    N = EMopt.rankStart;
    problem.N = N;
end

%Correct N when in mode 2. We start at the max allowed rank in this mode
if EMopt.learnRank == 2
    N = EMopt.rankMax;
    problem.N = N;
end

%Create matrix sparse multiply if needed
if BiGAMPopt.sparseMode
    sMult = @(arg1,arg2) sparseMult2(arg1.',arg2,...
        problem.rowLocations,problem.columnLocations);
    
    %Ensure that Y is proper dimensions
    if numel(Y) > numel(problem.rowLocations)
        Y = reshape(Y(omega),1,[]);
    end
end

%Include additional fields that are required by the BG code
EMopt.learn_lambda = false;

%Set noise variance if requested
if ~isfield(EMopt,'noise_var') || isempty(EMopt.noise_var)
    if ~BiGAMPopt.sparseMode
        EMopt.noise_var = sum(sum(abs(Y(omega)).^2))/(numel(omega)*101);
    else
        EMopt.noise_var = sum(sum(abs(Y).^2))/...
            (numel(omega)*101);
    end
end


%History
histFlag = false;
if nargout >=4
    histFlag = true;
    
    estHist.errZ = [];
    estHist.errX = [];
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
    nuX = (norm(Y,'fro')^2/numel(omega) - nuw)/N;
else
    nuX = EMopt.active_var;
end


%Compute X variance, setting nuA = 1
if ~isfield(EMopt,'tmax') || isempty(EMopt.tmax)
    tmax = EMopt.maxEMiter;
else
    tmax = EMopt.tmax;
end

%Initialize loop
t = 0;
tInner = 0;
stop = 0;


%Initialize xhat and Ahat
%xhat = sqrt(nuX)*randn(N,L);
xhat = zeros(N,L); %seems to perform better
Ahat = randn(M,N);

%Set initial step size small
BiGAMPopt.step = BiGAMPopt.stepMin;

%Set them
BiGAMPopt.xhat0 = xhat;
BiGAMPopt.Ahat0 = Ahat;
BiGAMPopt.Avar0 = ones(M,N);
BiGAMPopt.xvar0 = nuX*ones(N,L);

%Original tolerance
tol0 = BiGAMPopt.tol;

%Ensure that diagnostics are off to save run time
BiGAMPopt.diagnostics = 0;

%Ensure that EM outputs are calculated
BiGAMPopt.saveEM = 1;

%Outer loop for rank learning
bestVal = -inf;
rankStop = false;
zhatLast = 0;
SNR = 100;
outerCounter = 0;
zhatOld = 0;

%Save iteration count
nitSafe = BiGAMPopt.nit;

%% Main loop

while ~rankStop
    
    outerCounter = outerCounter + 1;
    
    %EM iterations
    %The < 2 condition is so that we can do a final iteration using the full
    %noise variance update without repeating code
    while stop < 2
        
        %Start timing
        tstart = tic;
        
        %Estimate SNR
        SNRold = SNR;
        
        %Compute SNR
        if t > 0
            if BiGAMPopt.sparseMode
                SNR = norm(zhat,'fro')^2/norm(Y - zhat,'fro')^2;
                
            else
                SNR = norm(zhat(omega),'fro')^2/...
                    norm(Y(omega) - zhat(omega),'fro')^2;
            end
        end
        
        
        %Set tolerance for this iteration
        %tolNew = 1 / SNR;
        tolNew = min(max(tol0,1/SNR),EMopt.maxTol);
        BiGAMPopt.tol = tolNew;
        
        %Increment time exit loop if exceeds maximum time
        t = t + 1;
        tInner = tInner + 1;
        if tInner >= EMopt.maxEMiterInner || stop > 0
            stop = stop + 1;
        end
        
        %Set iteration limits
        if (t == 1) && (EMopt.learnRank == 2)
            BiGAMPopt.nit = EMopt.nitFirstEM;
        else
            BiGAMPopt.nit = nitSafe;
        end
        
        %Prior on A
        gA = AwgnEstimIn(0, 1);
        
        %Prior on X
        gX = AwgnEstimIn(meanX, nuX);
        
        
        %Output log likelihood
        gOut = AwgnEstimOut(Y, nuw);
        
        
        %Stop timing
        t1 = toc(tstart);
        
        %Run BiG-AMP
        [estFin2,~,estHist2,state] = ...
            BiGAMP(gX, gA, gOut, problem, BiGAMPopt);
        
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
            ' meanX = ' num2str(mean(meanX(:)),'%5.3e')...
            ' tol = ' num2str(BiGAMPopt.tol,'%5.3e')...
            ' nuw = ' num2str(nuw,'%5.3e')...
            ' SNR = ' num2str(10*log10(SNR),'%03.2f')...
            ' Error = ' num2str(error_value,'%05.4f')...
            ' numIt = ' num2str(length(estHist2.errZ),'%04d')])
        
        
        
        %Compute zhat
        if BiGAMPopt.sparseMode
            zhat = sMult(estFin2.Ahat,estFin2.xhat);
        else
            zhat = estFin2.Ahat*estFin2.xhat;
        end
        
        
        %Calculate the change in signal estimates
        norm_change = norm(zhat - zhatOld,'fro')^2/norm(zhat,'fro')^2;
        %norm_change = norm(xhat-xhat2,'fro')^2/norm(xhat,'fro')^2;
        zhatOld = zhat;
        
        %Check for estimate tolerance threshold
        if (norm_change < max(tolNew/10,EMopt.EMtol)) &&...
                ( (norm_change < EMopt.EMtol) ||...
                (abs(10*log10(SNRold) - 10*log10(SNR)) < 1))
            stop = stop + 1;
        end
        
        
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
            if tInner >= tmax || stop > 0
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
            lambda, meanX, nuX, EMopt);
        
        %Rank estimation method 2
        if EMopt.learnRank == 2
            
            %Check for rank update
            Nhat = heuristicRank(estFin2.xhat,EMopt.rank_tau);
            
            %If a rank was accepted and it is at least a decrease of 2.
            %Occasionally, the algorithm will incorrectly decide to truncate
            %the rank by 1, so we do not allow it to do this
            if ~isempty(Nhat) && (Nhat < N - 1)
                
                %Inform user
                disp(['Updating rank estimate from ' num2str(N) ...
                    ' to ' num2str(Nhat) ' on iteration ' num2str(t)])
                
                
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
                
                %Only do this once
                EMopt.learnRank = 0;
                
            end
        end
        
        %Reinitialize GAMP estimates
        BiGAMPopt.xhat0 = estFin2.xhat;
        BiGAMPopt.xvar0 = estFin2.xvar;
        BiGAMPopt.shat0 = state.shatOpt;
        BiGAMPopt.Ahat0 = estFin2.Ahat;
        BiGAMPopt.Avar0 = estFin2.Avar;
        BiGAMPopt.step = BiGAMPopt.stepMin;
        
        %Stop timing
        t2 = toc(tstart);
        
        %Output Histories if necessary
        if histFlag
            estHist.errZ = [estHist.errZ; estHist2.errZ];
            estHist.errX = [estHist.errX; estHist2.errX];
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
    
    %Start timing
    tstart = tic;
    
    %Save several quantities for this iteration
    rank_hist(outerCounter) = N; %#ok<AGROW>
    cost_hist(outerCounter) = estHist.val(end); %#ok<AGROW>
    error_hist(outerCounter) = estHist.errZ(end);%#ok<AGROW>
    
    %Compute the residual
    if BiGAMPopt.sparseMode
        residualHist(outerCounter) = norm(Y - zhat,'fro')^2; %#ok<AGROW>
    else
        residualHist(outerCounter) = norm(Y(omega) - zhat(omega),'fro')^2; %#ok<AGROW>
    end
    
    %Compute number of free parameters
    Ne = N*(M + L - N);
    
    %Compute the various penalty functions
    AICc(outerCounter) = -Ns*log(residualHist(outerCounter)/Ns) - 2*Ns./(Ns - Ne - 1).*Ne; %#ok<AGROW>
    
    
    %Stop rank inflation if we are doing worse
    if AICc(outerCounter) < bestVal
        rankStop = true;
        disp(['Terminating, AICc was ' num2str(AICc(outerCounter),'%5.3e') ...
            ', estimated rank was ' num2str(rank_hist(outerCounter - 1))])
    else %otherwise, update estimate
        estFin.xhat = estFin2.xhat;
        estFin.xvar = estFin2.xvar;
        estFin.Ahat = estFin2.Ahat;
        estFin.Avar = estFin2.Avar;
        %Save final options
        optFin = BiGAMPopt;
        EMoptFin = EMopt;
        
    end
    bestVal = AICc(outerCounter);
    
    %Check max rank
    if N >= EMopt.rankMax
        rankStop = true;
    end
    
    %Reset stop
    stop = 0;
    tInner = 0;
    
    %Calculate the change in signal estimates
    norm_change = norm(zhat-zhatLast,'fro')^2/norm(zhat,'fro')^2;
    if norm_change < EMopt.EMtol
        rankStop = true;
    end
    zhatLast = zhat;
    
    %Check total EM iterations
    if t >= EMopt.maxEMiter
        rankStop = true;
    end
    
    %Check to see if rank learning is enabled
    if EMopt.learnRank ~= 1
        rankStop = true;
    end
    
    %Stop if we are over-parameterized
    if Ne > numel(omega)
        rankStop = true;
    end
    
    %If we are not stopping, increase the rank
    if ~rankStop
        
        %Increase by rank step
        N = N + EMopt.rankStep;
        problem.N = N;
        
        disp(['Increasing rank to ' num2str(N) ...
            ' AICc was ' num2str(AICc(outerCounter),'%5.3e')])
        
        %Expand
        xhat = [estFin2.xhat ; zeros(EMopt.rankStep,L)];
        Ahat = [estFin2.Ahat sqrt(var(estFin2.Ahat(:,end)))*randn(M,EMopt.rankStep)];
        if ~BiGAMPopt.uniformVariance
            Avar = [estFin2.Avar mean(estFin2.Avar(:))*ones(M,EMopt.rankStep)];
            xvar = [estFin2.xvar; mean(estFin2.xvar(:))*ones(EMopt.rankStep,L)];
        else
            %Temporary solution. This avoids local minima but causes
            %transients
            Avar = 10;
            xvar = 10*mean(nuX(:));
        end
        %Set
        BiGAMPopt.xhat0 = xhat;
        BiGAMPopt.xvar0 = xvar;
        BiGAMPopt.Ahat0 = Ahat;
        BiGAMPopt.Avar0 = Avar;
        
        %Expand nuX
        nuX = [nuX; mean(nuX(end,:))*ones(EMopt.rankStep,L)]; %#ok<AGROW>
        meanX = [meanX; mean(meanX(end,:))*ones(EMopt.rankStep,L)]; %#ok<AGROW>
        
        %Fix lambda
        lambda = ones(N,L);
        
        
    end
    
    %Stop timing
    t3 = toc(tstart);
    
    %Add the time in
    estHist.timing(end) = estHist.timing(end) + t3;
    
end


%% Cleanup

%Save the history
estHist.AICc = AICc;
estHist.cost_hist = cost_hist;
estHist.residualHist = residualHist;
estHist.rank_hist = rank_hist;
estHist.error_hist = error_hist;



