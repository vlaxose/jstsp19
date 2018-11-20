function results = trial_affine(optIn)
%Trial code for our general model

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
    optIn.tryPbigamp = true;
    optIn.tryEMPbigamp = true;
    
    %Specify a problem size
    alpha = 0.75; % ratio of measurements-to-unknowns
    rho = 0.25; % sparsity rate
    optIn.N = 80; %size of b and c
    optIn.K = rho*optIn.N; %number of non-zeros in b and c
    optIn.M = alpha*(2*optIn.N); %number of measurements
    
    %Specify SNR
    optIn.SNR = inf;
    
    %Control uniformVariance
    optIn.uniformVariance = true;
   
    %Control complex-valued
    optIn.cmplx = true;
   
    %Control affine offset
    optIn.affine = true;

else
    
    %Don't run stupid cases
    if optIn.K > optIn.M / 2
        error('No reason to waste computer time')
    end
    
end


%% Problem setup

%Algorithm settings
tryPbigamp = optIn.tryPbigamp;
tryEMPbigamp = optIn.tryEMPbigamp;


%Get parameters
%Specify problem dimensions
N = optIn.N; %length of b and c
K = optIn.K; %number of non-zeros in b and c
M = optIn.M; %number of measurements

%Derived lengths
Nb = N; %length of b
Nc = N; %length of c

%Other problem parameters
SNR = optIn.SNR;
cmplx = optIn.cmplx;
affine = optIn.affine;


%% Build true signals

%Draw Bernoulli-Gaussian signal vectors
if cmplx
    b = sqrt(1/2)*complex(randn(Nb,1),randn(Nb,1));
    c = sqrt(1/2)*complex(randn(Nc,1),randn(Nc,1));
else
    b = randn(Nb,1);
    c = randn(Nc,1);
end

%sparsify b and c
locs = randperm(Nb);
b(locs((K+1):end)) = 0;
locs = randperm(Nc);
c(locs((K+1):end)) = 0;


%% Build the Z object

%Draw measurement coefficients iid Gaussian
if cmplx
    zij = tensor(complex(randn(M,Nb,Nc),randn(M,Nb,Nc)));
else
    zij = tensor(randn(M,Nb,Nc));
end
if affine
    if cmplx
        zi0 = sqrt(Nc)*tensor(complex(randn(M,Nb),randn(M,Nb)));
        z0j = sqrt(Nb)*tensor(complex(randn(M,Nc),randn(M,Nc)));
    else
        zi0 = sqrt(Nc)*tensor(randn(M,Nb));
        z0j = sqrt(Nb)*tensor(randn(M,Nc));
    end
end

%Build parametric Z
if affine
  zObject = Affine_ParametricZ(zij,zi0,z0j);
else
  zObject = Affine_ParametricZ(zij);
end

%% Continue setup

%Save the true Z
z = zObject.computeZ(b,c);

%Determine nuw
nuw = norm(reshape(z,[],1))^2/M*10^(-SNR/10);
nuw = max(nuw,1e-10);

%Noisy output channel
if cmplx
    Y = z + sqrt(nuw/2)*complex(randn(size(z)),randn(size(z)));
else
    Y = z + sqrt(nuw)*randn(size(z));
end


%% Establish the channel objects for P-BiG-AMP

%Prior on B
if cmplx
    gB = CAwgnEstimIn(0, 1);
else
    gB = AwgnEstimIn(0, 1);
end
gB = SparseScaEstim(gB,K/Nb);

%Prior on C
if cmplx
   gC = CAwgnEstimIn(0, 1);
else
   gC = AwgnEstimIn(0, 1);
end
gC = SparseScaEstim(gC,K/Nc);

%Output log likelihood
if cmplx
    gOut = CAwgnEstimOut(Y, nuw);
else
    gOut = AwgnEstimOut(Y, nuw);
end


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
opt.tol = 1e-7;

%Specify maximum number of trials
maxTrials = 5;
threshNMSE = max(-30,-SNR);

%Use the initial values
if cmplx
    opt.bhat0 = sqrt(1/2)*(randn(Nb,1) + 1j*randn(Nb,1));
    opt.chat0 = sqrt(1/2)*(randn(Nc,1) + 1j*randn(Nc,1));
else
    opt.bhat0 = randn(Nb,1);
    opt.chat0 = randn(Nc,1);
end

%Variances
[mB,vB,~] = gB.estimInit(); holder=abs(mB).^2+vB;
opt.bvar0 = 10*holder;
[mC,vC,~] = gC.estimInit(); holder=abs(mC).^2+vC;
opt.cvar0 = 10*holder;

%Specify error functions
opt.error_function = @(qval) 20*log10(norm(qval - z,'fro') / norm(z,'fro'));
if affine
    opt.error_functionB = @(qval) 20*log10(norm(qval - b)/norm(b));
    opt.error_functionC = @(qval) 20*log10(norm(qval - c)/norm(c));
else
    % scale first to circumvent ambiguity
    opt.error_functionB = @(qval) 20*log10(norm(qval*((qval'*b)/norm(qval)^2) - b)/norm(b));
    opt.error_functionC = @(qval) 20*log10(norm(qval*((qval'*c)/norm(qval)^2) - c)/norm(c));
end

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
        
        [~, ~, estHist] = ...
            PBiGAMP(gB, gC, gOut, problem, opt);
        
        
        if (estHist.errZ(end) < threshNMSE) || failCounter > maxTrials
            stop = true;
        else % start from a new random initialization
            if cmplx
               opt.bhat0 = complex(randn(Nb,1),randn(Nb,1));
               opt.chat0 = complex(randn(Nc,1),randn(Nc,1));
            else
               opt.bhat0 = randn(Nb,1);
               opt.chat0 = randn(Nc,1);
            end
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHist.errZ(end))])
    end
    tGAMP = toc(tstart);
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHist.errZ(end);
    results{loc}.errB = estHist.errB(end);
    results{loc}.errC = estHist.errC(end);
    results{loc}.time = tGAMP;
    results{loc}.errHist = estHist.errZ;
    results{loc}.timeHist = estHist.timing;
    
end

%% Try EM-P-BiG-AMP

if tryEMPbigamp
    
    %Specify options
    if cmplx
        EMopt.B_type = 'CBG';
        EMopt.C_type = 'CBG';
    else
        EMopt.B_type = 'BG';
        EMopt.C_type = 'BG';
    end
    EMopt.C_var_init = 'fixed';
    EMopt.C_learn_lambda = 1;
    EMopt.B_learn_lambda = 1;
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
        [~, ~, ~,estHistEM] = ...
            EMPBiGAMP(Y,problem,opt,EMopt);
        
        %Check
        if (estHistEM.errZ(end) < threshNMSE) || failCounter > maxTrials
            stop = true;
        else
            if cmplx
               opt.bhat0 = complex(randn(Nb,1),randn(Nb,1));
               opt.chat0 = complex(randn(Nc,1),randn(Nc,1));
            else
               opt.bhat0 = randn(Nb,1);
               opt.chat0 = randn(Nc,1);
            end
        end
        disp(['Attempts completed: ' num2str(failCounter) ' Final Z error= ' ...
            num2str(estHistEM.errZ(end))])
        
    end
    tEMGAMP = toc(tstart);
    
    
    disp(['True nuw = ' num2str(nuw)])
    disp(['True B_lambda = ' num2str(K/Nb)])
    disp(['True C_lambda = ' num2str(K/Nc)])
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-P-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.errB = estHistEM.errB(end);
    results{loc}.errC = estHistEM.errC(end);
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    
end



%% Analyze results

if nargin == 0
    
    %Show Z error
    figure(901)
    clf
    if tryPbigamp
        plot(estHist.errZ,'b-x')
    end
    if tryEMPbigamp
        hold on
        plot(estHistEM.errZ,'r-+')
        hold off
        legend('PBiG-AMP','EM-PBiG-AMP')
    else
        legend('PBiG-AMP')
    end
    grid
    xlabel('iteration')
    ylabel('Z NMSE (dB)')
    
    
    %Show results
    results{:} %#ok<NOPRT>
end



