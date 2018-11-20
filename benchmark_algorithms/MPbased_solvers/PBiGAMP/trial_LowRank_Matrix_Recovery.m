function results = trial_LowRank_Matrix_Recovery(optIn)
%Trial code for generalized low rank matrix recovery

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
    optIn.R = 4;
    optIn.Mq = 80;
    optIn.Nq = 80;
    df = optIn.R*(optIn.Mq + optIn.Nq - optIn.R);
    optIn.M = 3*df;
    
    %Specify SNR
    optIn.SNR = 50;
    
    %Specify sparsity of Phi- number of non-zeroes per matrix
    optIn.phiSparsity = 10;
    
    %Control uniformVariance
    optIn.uniformVariance = 1;
    
    %Report status
    disp(['M=' num2str(optIn.M) '; DoF=' num2str(df) ...
        '; MqNq=' num2str(optIn.Mq*optIn.Nq)])
    
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
SNR = optIn.SNR;

%Derived sizes
Nb = Mq*R;
Nc = Nq*R;


%% Build true signals

%Build the two matrix factors
Bmat = sqrt(1/2)*complex(randn(R,Mq),randn(R,Mq));
Cmat = sqrt(1/2)*complex(randn(R,Nq),randn(R,Nq));
B = Bmat(:);
C = Cmat(:);

%Build Q
Q = Bmat.'*Cmat;


%% Build Phi


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

%% Compare two methods for Z object

%Specify
whichMethod = 0;

switch whichMethod
    
    case 0,
        
        %Build the parametric Z
        zObject = LowRank_Matrix_Recovery_ParametricZ(Mq,Nq,R,Phi);
        
        
    case 1,
        
        %The following is a way of computing the resulting 4D tensor,
        %which is M x 1 x Nb x Nc, that satisfies
        %zij(m,1,:,:) = kron(conj(reshape(Phi(:,m),Mq,Nq)),eye(R));
        %without needing to explicitly construct the kron products.
        
        %Determine linear indices of elements
        inds = sub2ind(size(Phi),row,col);
        
        %Convert to indices for 3D matrix
        [rr,cc,pp] = ind2sub([Mq Nq M],inds);
        
        %Group by page
        rr = reshape(rr,[],M);
        cc = reshape(cc,[],M);
        
        %Shift starting locations
        rr = (rr - 1)*R;
        cc = (cc - 1)*R;
        
        %Expand
        rr = kron(rr,ones(R,1));
        cc = kron(cc,ones(R,1));
        pp = kron(pp,ones(R,1));
        vals = kron(conj(val),ones(R,1));
        
        %Create offset and add
        offset = repmat(kron(ones(phiSparsity,1),(1:R)'),1,M);
        rr = vec(rr + offset);
        cc = vec(cc + offset);
        
        %Create tensor
        zij = sptensor([pp rr cc ],vals,[M Nb Nc]);
        
        %Build parametric Z
        zObject = Affine_ParametricZ(zij,[],[],[]);
        
end


%% Continue setup

%Save the true Z
Z = zObject.computeZ(B,C);

%Determine nuw
nuw = norm(reshape(Z,[],1))^2/M*10^(-SNR/10);

%Noisy output channel
Y = Z + sqrt(nuw/2)*complex(randn(size(Z)),randn(size(Z)));


%% Establish the channel objects for P-BiG-AMP

%Prior on B
gB = CAwgnEstimIn(0, 1);

%Prior on C
gC = CAwgnEstimIn(0, 1);

%Output log likelihood
gOut = CAwgnEstimOut(Y, nuw);


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
opt.nit = 200;
opt.tol = 1e-6;

%Specify maximum number of trials
maxTrials = 5;
threshNMSE = -30;

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
opt.error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
error_functionQ = @(qval) 20*log10(norm(qval - Q,'fro') / norm(Q,'fro'));



%% Init results

%Initialize results as empty
results = [];

%% Try TFOCS

if tryTfocs
    
    %Specify problem
    Atfocs = @(varargin) tfocs_linop( Mq, Nq, M, Phi, varargin{:} );
    epsilon = norm(Y - Z,'fro'); %use the truth
    mu = 0.05;
    contOpts.maxIts = 20;
    contOpts.tol = opt.tol;
    
    %Options from web site example
    optsTfcos = [];
    optsTfcos.stopCrit       = 4;
    optsTfcos.printEvery     = 0; %changed
    optsTfcos.tol            = opt.tol;
    optsTfcos.maxIts         = 25;
    
    %Call TFOCS
    tstart = tic;
    Qtfocs = ...
        tfocs_SCD( prox_nuclear, { Atfocs, -1*Y },...
        prox_l2(epsilon), mu, [], [], optsTfcos,contOpts);
    tTfocs = toc(tstart);
    
    
    % %Truncate rank
    % [U,S,V] = svd(Qtfocs);
    % S = diag(S);
    % S((R+1):end) = 0;
    % Qtfocs = U*diag(S)*V';
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'TFOCS'; %#ok<*AGROW>
    results{loc}.err = opt.error_function(Atfocs(Qtfocs,1));
    results{loc}.errQ = error_functionQ(Qtfocs);
    results{loc}.time = tTfocs;
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
            opt.chat0 = complex(randn(Nc,1),randn(Nc,1));
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHist.errZ(end))])
    end
    tGAMP = toc(tstart);
    
    %Compute estimate of Q
    Qhat = reshape(estFin.bhat,R,Mq).'*reshape(estFin.chat,R,Nq);
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHist.errZ(end);
    results{loc}.errQ = error_functionQ(Qhat);
    results{loc}.time = tGAMP;
    results{loc}.errHist = estHist.errZ;
    results{loc}.timeHist = estHist.timing;
    
end

%% Try EM-P-BiG-AMP

if tryEMPbigamp
    
    %Specify options
    EMopt.B_type = 'CG';
    EMopt.C_type = 'CG';
    EMopt.C_var_init = 'fixed';
    EMopt.C_learn_lambda = 0;
    EMopt.maxEMiter = 10;
    opt.nit = 200;
    
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
            opt.chat0 = complex(randn(Nc,1),randn(Nc,1));
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHistEM.errZ(end))])
        
    end
    tEMGAMP = toc(tstart);
    
    %Compute estimate of Q
    QhatEM = zObject.computeQ(estFinEM.bhat,estFinEM.chat);
    
    disp(['True nuw = ' num2str(nuw)])
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.errQ = error_functionQ(QhatEM);
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    
end

%% Oracle results

if whichMethod == 0
    %Compute Oracle estimate assuming knowledge of B and dense C
    Atrue = zObject.getAOperator(B);
    fhand = @(x) pcgHelper(Atrue,nuw/1,true(size(C)),x);
    chat_oracle = Atrue.multTr(pcg(fhand,Y,1e-10,1e6));
    Qoracle = zObject.computeQ(B,chat_oracle);
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'B Oracle'; %#ok<*AGROW>
    results{loc}.err = opt.error_function(zObject.computeZ(B,chat_oracle));
    results{loc}.errQ = error_functionQ(Qoracle);
    
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
    plot(xlim, results{end}.errQ*[1 1],'k-');
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
    %Show results
    results{:} %#ok<NOPRT>
end



