function results = trial_LowRank_Plus_Sparse_Matrix_Recovery(optIn)
%Trial code for generalized low rank + sparse matrix recovery

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
    optIn.tryTfocs = 0;
    
    %Specify a problem size
    optIn.R = 5;
    optIn.Mq = 100;
    optIn.Nq = 100;
    
    %Control uniformVariance
    optIn.uniformVariance = 1;
    
    
    %Specify sparsity of Phi- number of non-zeroes per matrix
    %Set to negative for dense Phi
    optIn.phiSparsity = 50;
    
    optIn.lambda = 0.1; %fraction of entries corrupted with large outliers
    optIn.nuw = [0 20^2/12]; %First entry is AWGN, second entry is variance of large outliers
    
    
    %Decide on measurements
    df = optIn.R*(optIn.Mq + optIn.Nq - optIn.R) + ...
        optIn.lambda*optIn.Mq*optIn.Nq;
    optIn.M = min(optIn.Mq*optIn.Nq,ceil(3*df));
    
    %Report status
    disp(['M=' num2str(optIn.M) '; DoF=' num2str(df) ...
        '; MqNq=' num2str(optIn.Mq*optIn.Nq)])
    
end

%Do a sanity check
df = optIn.R*(optIn.Mq + optIn.Nq - optIn.R) + ...
    optIn.lambda*optIn.Mq*optIn.Nq;

if df > optIn.M
    error('Above counting bound, no reason to waste computer time')
end


%% Problem setup

%Algorithm settings
tryPbigamp = optIn.tryPbigamp;
tryEMPbigamp = optIn.tryEMPbigamp;
tryTfocs = optIn.tryTfocs;

%Get parameters
Nq = optIn.Nq;
Mq = optIn.Mq;
R = optIn.R;
M = optIn.M;
phiSparsity = optIn.phiSparsity;
nuw = optIn.nuw;
lambda = optIn.lambda;

%Derived sizes
Nb = Mq*R;
Nc = Nq*R + Mq*Nq;



%% Build true signals

%Build the two matrix factors
Bmat = sqrt(1/2)*complex(randn(R,Mq),randn(R,Mq));
Cmat1 = sqrt(1/2)*complex(randn(R,Nq),randn(R,Nq));
B = Bmat(:);
C1 = Cmat1(:);

%Build the sparse matrix
errorWidth = sqrt(12*nuw(2));
inds = rand(Mq,Nq) < lambda;
Cmat2 = (-errorWidth/2 + errorWidth*rand(Mq,Nq)).*inds.*exp(1j*2*pi*rand(Mq,Nq));
C2 = Cmat2(:);

%C
C = [C1;C2];

%Build Q1
Q1 = Bmat.'*Cmat1;

%Build Q
%Q = Q1 + Cmat2;

%% Build the Z object

%Efficient method for generating very large sparse Phi matrices with the
%same number of non-zeros in each row, chosen uniformly at random. This is
%MUCH faster than more straightforward methods.
if phiSparsity > 0
    
    %Preallocate sparsity locations
    row = zeros(phiSparsity*M,1);
    col = kron((1:M)',ones(phiSparsity,1));
    val = sqrt(1/2)*complex(randn(size(row)),randn(size(row)));
    
    %Compute Random row locations
    startVal = 1;
    for m = 1:M
        
        %Get unique locations for this row. This becomes substantially
        %faster than using randperm for large Mq*Nq.
        locs = 0;
        while length(locs) < phiSparsity
            locs = unique(randi(Mq*Nq,phiSparsity,1));
        end
        
        %Assign them
        row(startVal:(startVal+phiSparsity-1)) = locs;
        startVal = startVal + phiSparsity;
    end
    
    %Build Phi
    Phi = sparse(row,col,val,Mq*Nq,M);
else
    Phi = sqrt(1/2)*complex(randn(Mq*Nq,M),randn(Mq*Nq,M));
end


%Build the parametric Z
zObject = LowRank_Plus_Sparse_Matrix_Recovery_ParametricZ(...
    Mq,Nq,R,Phi);

%Save the true Z
Z = zObject.computeZ(B,C);

%Add white noise
Y = Z + sqrt(nuw(1)/2)*complex(randn(size(Z)),randn(size(Z)));



%% Establish the channel objects for P-BiG-AMP

%Prior on B
gB = CAwgnEstimIn(0, 1);

%Prior on C
gC1 = CAwgnEstimIn(0, 1);
gC2 = CAwgnEstimIn(0, nuw(2));
gC2 = SparseScaEstim(gC2,lambda);
gC = EstimInConcat({gC1 gC2},[Nq*R Mq*Nq]);

%Output log likelihood
gOut = CAwgnEstimOut(Y, nuw(1));

%% Options and Problem objects

%Setup the problem
problem = PBiGAMPProblem();
problem.M = M;
problem.Nb = Nb;
problem.Nc = Nc;
problem.zObject = zObject;

%Setup the options
opt = PBiGAMPOpt();

%Iteration settings
opt.verbose = (nargin == 0);
opt.nit = 500;
opt.tol = 1e-8;

%Uniform variance
opt.uniformVariance = optIn.uniformVariance;

%Specify maximum number of trials
maxTrials = 15;
threshNMSE = -80;

%Set initial values
opt.bhat0 = sqrt(1/2)*(randn(Nb,1) + 1j*randn(Nb,1));
opt.chat0 = [sqrt(1/2)*(randn(Nq*R,1) + 1j*randn(Nq*R,1)); ...
    zeros(Mq*Nq,1)];

%Variances
[~,holder,~] = gB.estimInit();
opt.bvar0 = 10*holder;
[~,holder,~] = gC.estimInit();
opt.cvar0 = 10*holder;


%Specify error functions
opt.error_functionB = @(qval) 0;
opt.error_functionC = @(qval) 0;
opt.error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
error_functionQ = @(qval) 20*log10(norm(qval - Q1,'fro') / norm(Q1,'fro'));
error_functionC2 = @(qval) 20*log10(norm(qval(:) - C2(:),'fro') / norm(C2(:),'fro'));




%% Init results

%Initialize results as empty
results = [];

%% Try TFOCS

if tryTfocs
    
    %Specify problem
    Atfocs = @(varargin) tfocs_linop( Mq, Nq, M, Phi, varargin{:} );
    epsilon = norm(Y - Z,'fro'); %use the truth
    lambda_tfocs = sqrt(1/Mq);
    mu = 0.05;
    contOpts = [];
    contOpts.maxIts = 30;
    contOpts.tol = opt.tol;
    
    %Options from web site example
    optsTfcos = [];
    optsTfcos.stopCrit       = 4;
    optsTfcos.printEvery     = 0; %changed
    optsTfcos.tol            = opt.tol; %changed!
    optsTfcos.maxIts         = 50; %doubled
    
    %Call TFOCS
    tstart = tic;
    if epsilon > 0
        tfocs_res = ...
            tfocs_SCD({prox_nuclear(1), prox_l1(lambda_tfocs)}, { Atfocs, Atfocs, -1*Y },...
            prox_l2(epsilon), mu, [], [], optsTfcos,contOpts);
    else
        tfocs_res = ...
            tfocs_SCD({prox_nuclear(1), prox_l1(lambda_tfocs)}, { Atfocs, Atfocs, -1*Y },...
            prox_0, mu, [], [], optsTfcos,contOpts);
    end
    tTfocs = toc(tstart);
    
    %Grab component
    Qtfocs = tfocs_res{1};
    C2tfocs = tfocs_res{2};
    
    % %Truncate rank
    % [U,S,V] = svd(Qtfocs);
    % S = diag(S);
    % S((R+1):end) = 0;
    % Qtfocs = U*diag(S)*V';
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'TFOCS'; %#ok<*AGROW>
    results{loc}.err = opt.error_function(Atfocs(Qtfocs,1) + Atfocs(C2tfocs,1));
    results{loc}.errQ = error_functionQ(Qtfocs);
    results{loc}.errC2 = error_functionC2(C2tfocs);
    results{loc}.time = tTfocs;
    
    %Show it
    results{loc} %#ok<NOPRT>
end

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
            opt.bhat0 = complex(randn(Nb,1),randn(Nb,1));
            opt.chat0 = [complex(randn(Nq*R,1),randn(Nq*R,1));zeros(Mq*Nq,1)];
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHist.errZ(end))])
    end
    tGAMP = toc(tstart);
    
    %Compute estimate of Q
    Qhat = zObject.computeQ1(estFin.bhat,estFin.chat);
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHist.errZ(end);
    results{loc}.errQ = error_functionQ(Qhat);
    results{loc}.errC2 = error_functionC2(estFin.chat((Mq*R+1):end));
    results{loc}.time = tGAMP;
    results{loc}.errHist = estHist.errZ;
    results{loc}.timeHist = estHist.timing;
    
    %Show it
    results{loc} %#ok<NOPRT>
    
end

%% Try EM-P-BiG-AMP

if tryEMPbigamp
    
    %Specify options
    EMopt.B_type = 'CG';
    EMopt.C_type = 'LowRankPlusSparse';
    EMopt.Csizes = [Nq*R Mq*Nq];
    EMopt.C_var_init = 'fixed';
    EMopt.C_learn_lambda = 1;
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
            EMPBiGAMP(Y,problem,opt,EMopt);
        
        %Check
        if (estHistEM.errZ(end) < threshNMSE) || failCounter > maxTrials
            stop = true;
        else
            opt.bhat0 = complex(randn(Nb,1),randn(Nb,1));
            opt.chat0 = [complex(randn(Nq*R,1),randn(Nq*R,1));zeros(Mq*Nq,1)];
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHistEM.errZ(end))])
        
    end
    tEMGAMP = toc(tstart);
    
    %Compute estimate of Q
    QhatEM = zObject.computeQ1(estFinEM.bhat,estFinEM.chat);
    
    disp(['True nuw = ' num2str(nuw)])
    disp(['True lamba = ' num2str(lambda)])
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.errQ = error_functionQ(QhatEM);
    results{loc}.errC2 = error_functionC2(estFinEM.chat((Mq*R+1):end));
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    
    %Show it
    %results{loc} %#ok<NOPRT>
end

%% Oracle results
%
% %Compute Oracle estimate assuming knowledge of B and dense C
% Atrue = zObject.getAOperator(B);
% fhand = @(x) pcgHelper(Atrue,nuw/1,true(size(C)),x);
% chat_oracle = Atrue.multTr(pcg(fhand,Y,1e-10,1e6));
% Qoracle = zObject.computeQ(B,chat_oracle);
%
%
% %Save results
% loc = length(results) + 1;
% results{loc}.name = 'B Oracle'; %#ok<*AGROW>
% results{loc}.err = opt.error_function(zObject.computeAX(B,chat_oracle));
% results{loc}.errQ = error_functionQ(Qoracle);


%% Analyze results

%Save the options
results{1}.optIn = optIn;
results{1}.df = R*(Mq + Nq - R) + nnz(Cmat2);

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
    grid
    xlabel('iteration')
    ylabel('Z NMSE (dB)')
    
    %     %Show final Q and truth
    %     figure(902)
    %     clf
    %     plot(real(Q(:)),'ko')
    %     hold on
    %     plot(real(Qhat(:)),'bx')
    %     plot(real(QhatEM(:)),'r+')
    %     title('Real part of Q')
    %     grid
    %
    %     %Show final Q and truth
    %     figure(903)
    %     clf
    %     plot(imag(Q(:)),'ko')
    %     hold on
    %     plot(imag(Qhat(:)),'bx')
    %     plot(imag(QhatEM(:)),'r+')
    %     title('Imag part of Q')
    %     grid
    %
    
%     figure(905)
%     clf
%     plot(real(Cmat2(:)),'bo')
%     hold on
%     plot(real(estFinEM.chat(R*Nq+1:end)),'r+')
    
    %Show results
    results{:} %#ok<NOPRT>
end



