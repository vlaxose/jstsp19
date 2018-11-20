function results = trial_calibration(optIn)
%Trial code for calibration experiment.

%% Test Case

if nargin == 0
    
    %Fresh slate
    clc
    
    %Handle random seed
    if 1 %change to zero to try the same random draw repeatedly
        savedState = rng;
        save random_state.mat savedState;
    else
        load random_state.mat %#ok<UNRCH>
    end
    rng(savedState);
    
    %Control algorithms
    optIn.tryPbigamp = 1;
    optIn.tryEMPbigamp = 1;
    optIn.trySparseLift = 0;
    
    %Specify a problem size
    optIn.Nb = 4; %size of call vector
    optIn.Nc = 256; %length of signal
    optIn.K = 10; %number of non-zeros in c
    optIn.M = 128; %number of measurements
    
    
    %Specify SNR
    optIn.SNR = inf;
    
    %Control uniformVariance
    optIn.uniformVariance = 1;
    
else
    
    %Don't run stupid cases
    if optIn.M < (optIn.Nb + optIn.K)
        error('No reason to waste computer time')
    end
    
end


%% Problem setup

%Algorithm settings
tryPbigamp = optIn.tryPbigamp;
tryEMPbigamp = optIn.tryEMPbigamp;
trySparseLift = optIn.trySparseLift;

%Get params
Nb = optIn.Nb; %size of call vector
Nc = optIn.Nc; %length of signal
K = optIn.K; %number of non-zeros in c
M = optIn.M; %number of measurements
SNR = optIn.SNR;

%Fix L
L = 1;


%% Build true signals

%Build the two matrix factors
%b = sqrt(1/2)*complex(randn(Nb,1),randn(Nb,1));
%c = sqrt(1/2)*complex(randn(Nc,1),randn(Nc,1));

%Build the two matrix factors
b = randn(Nb,1);
c = randn(Nc,1);

%sparsify c
locs = randperm(Nc);
c(locs((K+1):end)) = 0;


%% Build the Z object

%Draw A0 matrix
%A0 = sqrt(1/2)*complex(randn(M,Nc),randn(M,Nc));
A0 = randn(M,Nc);

%Build H from the DFT
H = dftmtx(M)/sqrt(M);
H = H(:,1:Nb);
%H = sqrt(1/2)*complex(randn(M,Nb),randn(M,Nb));

%Build the A operator
Aop = Calibration_PLinTrans(A0,H);

%Build the ParametricZ objet
zObject = Multiple_Snapshot_ParametricZ(Aop,L);


%% Continue setup

%Save the true Z
z = zObject.computeZ(b,c);

%Determine nuw
nuw = norm(reshape(z,[],1))^2/M*10^(-SNR/10);

%Noisy output channel
y = z + sqrt(nuw/2)*complex(randn(size(z)),randn(size(z)));


%% Establish the channel objects for P-BiG-AMP

%Prior on B
gB = CAwgnEstimIn(0, 1);

%Prior on C
gC = CAwgnEstimIn(0, 1);
gC = SparseScaEstim(gC,K/Nc);

%Output log likelihood
gOut = CAwgnEstimOut(y, nuw);


%% Options and Problem objects

%Setup the problem
problem = PBiGAMPProblem();
problem.M = M;
problem.Nb = Nb;
problem.Nc = Nc;
problem.zObject = zObject;

%Setup the options
opt = PBiGAMPOpt();

%Uniform variance
opt.uniformVariance = optIn.uniformVariance;

%Iteration settings
opt.verbose = (nargin == 0);
opt.nit = 500;
opt.tol = 1e-8;

%Specify maximum number of trials
maxTrials = 15;
threshNMSE = -110;

%Use the initial values
opt.bhat0 = sqrt(1/2)*(randn(Nb,1) + 1j*randn(Nb,1));
opt.chat0 = sqrt(1/2)*(randn(Nc,1) + 1j*randn(Nc,1));

%Variances
[~,holder,~] = gB.estimInit();
opt.bvar0 = 10*holder;
[~,holder,~] = gC.estimInit();
opt.cvar0 = 10*holder;

%Specify error functions
opt.error_functionB = @(qval) 0;
opt.error_functionC = @(qval) 0;
opt.error_function = @(qval) 20*log10(norm(qval - z,'fro') / norm(z,'fro'));

%Overall error function
error_function_full = @(bhat,chat) 20*log10(norm(bhat*chat.' - b*c.','fro')/norm(b*c.','fro'));


%% Init results

%Initialize results as empty
results = [];



%% P-BiG-AMP

if tryPbigamp
    
    %Count failures
    failCounter = 0;
    stop = false;
    
    tstart = tic;
    while ~stop
        
        %Increment counter
        failCounter = failCounter + 1;
        
        disp('Starting P-BiG-AMP')
        
        [estFin, ~, estHist] = ...
            PBiGAMP(gB, gC, gOut, problem, opt);
        
        
        if (estHist.errZ(end) < threshNMSE) || failCounter > maxTrials
            stop = true;
        else
            opt.bhat0 = sqrt(1/2)*(randn(Nb,1) + 1j*randn(Nb,1));
            opt.chat0 = sqrt(1/2)*(randn(Nc,1) + 1j*randn(Nc,1));
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHist.errZ(end))])
    end
    tGAMP = toc(tstart);
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHist.errZ(end);
    results{loc}.errFull = error_function_full(estFin.bhat,estFin.chat);
    results{loc}.time = tGAMP;
    results{loc}.errHist = estHist.errZ;
    results{loc}.timeHist = estHist.timing;
    results{loc}.fails = failCounter;
end

%% Try EM-P-BiG-AMP

if tryEMPbigamp
    
    %Specify options
    EMopt.B_type = 'CG';
    EMopt.C_type = 'CBG';
    EMopt.C_var_init = 'fixed';
    EMopt.C_learn_lambda = 1;
    EMopt.B_learn_lambda = 0;
    EMopt.C_lambda = 0.01; %start smaller
    EMopt.maxEMiter = 10;
    opt.nit = 500;
    
    %Coding
    disp('Starting EM-P-BiG-AMP')
    
    %Count failures
    failCounter = 0;
    stop = false;
    
    tstart = tic;
    while ~stop
        
        %Increment counter
        failCounter = failCounter + 1;
        
        %Run EM-P-BiG-AMP
        [estFinEM, ~, ~,estHistEM] = ...
            EMPBiGAMP(y,problem,opt,EMopt);
        
        %Check
        if (estHistEM.errZ(end) < threshNMSE) || failCounter > maxTrials
            stop = true;
        else
            opt.bhat0 = sqrt(1/2)*(randn(Nb,1) + 1j*randn(Nb,1));
            opt.chat0 = sqrt(1/2)*(randn(Nc,1) + 1j*randn(Nc,1));
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHistEM.errZ(end))])
        
    end
    tEMGAMP = toc(tstart);
    
    
    disp(['True nuw = ' num2str(nuw)])
    disp(['True C_lambda = ' num2str(K/Nc)])
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.errFull = error_function_full(estFinEM.bhat,estFinEM.chat);
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    results{loc}.fails = failCounter;
end



%% Try SparseLift

if trySparseLift
    
    %Notify
    disp('Starting SparseLift')
    
    %Build the required Phi matrix
    Phi = zeros(M,Nb*Nc);
    for kk = 1:M
        Phi(kk,:) = kron((A0(kk,:)),(H(kk,:)));
    end
    
    %Do a sanity check
    sanity_check = norm(z - Phi*vec(b*c.'),'fro');
    disp(['SparseLift sanity check: ' num2str(sanity_check)])
    
    %Set EPS using truth
    EPS = norm(y - z,'fro');
    
    %Use CVX
    tstart = tic;
    cvx_begin
    
    cvx_precision best
    variable xSL(Nc*Nb,1)
    minimize norm(xSL,1)
    subject to
    %Phi*xSL == z %exact
    norm(Phi*xSL - y,2) <= EPS %#ok<NOPRT>
    
    cvx_end
    time_SL = toc(tstart);
    
    
    
    %Get errors
    err_SL = opt.error_function(Phi*xSL);
    errFull_SL = 20*log10(norm(xSL - vec(b*c.'),'fro')/norm(vec(b*c'),'fro'));
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'SparseLift'; %#ok<*AGROW>
    results{loc}.err = err_SL;
    results{loc}.errFull = errFull_SL;
    results{loc}.time = time_SL;
    
    
end


%% Analyze results

if nargin == 0
    
    %Show Z error
    figure(901)
    clf
    if tryPbigamp
        plot(estHist.errZ,'b-x')
    end
    hold on
    if tryEMPbigamp
        plot(estHistEM.errZ,'r-+')
    end
    hold on
    grid
    xlabel('iteration')
    ylabel('Z NMSE (dB)')
    
    
    %Show results
    results{:} %#ok<NOPRT>
end



