function results = trial_DL(optIn)

%trial_DL: This function runs several algorithms on a sample
%instance of the dictionary learning problem. The function can be run with no
%arguments. See the nargin==0 block below for the required fields to call
%the function with optional parameters. The function returns a structure of
%algorithm results.
%This code can be used to produce the results for the noise free phase
%plane plots for dictionary learning in the BiG-AMP arXiv submission. 
%(This code conducts a single trial at 1 point in the phase plane.)

%Add paths
setup_DL

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
    optIn.tryBigampEM = 1;
    
    %Flags to turn on comparison algorithms
    optIn.tryKsvd = 0;
    optIn.tryErspud = 0;
    optIn.trySpams = 0;

    
    %Problem dimensions
    optIn.M = 20; %size of signal
    optIn.N = optIn.M; %size of dictionary
    optIn.L = ceil(5*optIn.N*log(optIn.N)); %From Spielman et al.
    
    
    %Specify dictionary usage
    optIn.K = 5;
    
    %Specify maximum allowed trials
    optIn.maxTrials = 1;
    
    %SNR
    optIn.SNR = inf;
    
    %Specify coding mode (0 for OMP, 1 for TST)
    %The specified function is used to compute a coding error for the
    %resulting dictionary for each algorithm. TST may be significantly
    %slower.
    optIn.useTST = 0;
    
    %Precondition option (enable to use a preconditioner for EM-BiG-AMP)
    optIn.precondition = 0;
    
    
end

%% Problem Setup

%Turn algs on/off
tryBigampEM = optIn.tryBigampEM;
tryKsvd = optIn.tryKsvd;
tryErspud = optIn.tryErspud;
trySpams = optIn.trySpams;

%SNR
SNR = optIn.SNR;

%Coding
useTST = optIn.useTST;

%Precondition
precondition = optIn.precondition;

%Max trials
maxTrials = optIn.maxTrials;

%Define problem dimensions
M = optIn.M;
L = optIn.L;
N = optIn.N;
K = optIn.K;

%Set options
opt = BiGAMPOpt; %initialize the options object
opt.nit = 500; %limit iterations

%Set sizes
problem = BiGAMPProblem();
problem.M = M;
problem.N = N;
problem.L = L;


%% Build the dictionary

%Draw randomly
A = randn(M,N);

%Normalize the columns
A = A*diag(1 ./ sqrt(abs(diag(A'*A))));

%Dictionary error function
dictionary_error_function =...
    @(q) 20*log10(norm(A -...
    q*find_permutation(A,q),'fro')/norm(A,'fro'));



%% Compute coefficient vectors

%Compute true vectors with exactly K non-zeroes in each
X = randn(N,L);
for ll = 1:L
    yada = randperm(N);
    yada2 = zeros(N,1);
    yada2(yada(1:K)) = 1;
    X(:,ll) = X(:,ll) .* yada2;
end



%% Form the output channel

%Compute noise free output
Z = A*X;

%Define the error function
error_function = @(qval) 20*log10(norm(qval - Z,'fro') / norm(Z,'fro'));
opt.error_function = error_function;


%Determine nuw
nuw = norm(reshape(Z,[],1))^2/M/L*10^(-SNR/10);

%Noisy output channel
Y = Z + sqrt(nuw)*randn(size(Z));

%Coding error
coding_error_function = @(q) 20*log10(coding_error(Y,q,K,useTST));


%Initialize results as empty
results = [];


%% EM BiG AMP

if tryBigampEM
    
    %Silence
    opt.verbose = false;
    disp('Starting EM-BiG-AMP')
    
    %Coding
    bestError = inf;
    bestSparsity = inf;
    
    %Compute preconditioner
    if precondition
        Q = chol(inv(Y*Y'));
    else
        Q = 1;
    end
    
    %Define the error function
    QZ = Q*Y; %Using (possibly) noisy data for error function
    error_function2 = @(qval) 20*log10(norm(qval - QZ,'fro') / norm(QZ,'fro'));
    opt.error_function = error_function2;
    
    tstart = tic;
    for trial = 1:maxTrials
        
        %Run EM-BiGAMP  
        [estFinTemp,~,~,estHistEMtemp] = ...
            EMBiGAMP_DL(Q*Y,problem,opt);
        
        %Correct the dictionary
        estFinTemp.Ahat = Q \ estFinTemp.Ahat;
        
        %Reset error function
        opt.error_function = error_function;
        
        %If the sparisty is better and the error is very small or better
        if (sum(sum(estHistEMtemp.p1)) < bestSparsity) && ...
                ( (estHistEMtemp.errZ(end) < bestError)  ||...
                (estHistEMtemp.errZ(end) < -100)  )
            
            %Update
            bestSparsity = sum(sum(estHistEMtemp.p1));
            bestError = estHistEMtemp.errZ(end);
            AhatOptEM = estFinTemp.Ahat;
            estHistEM = estHistEMtemp;
            p1EM = estHistEMtemp.p1;
            
            %Notify user
            disp(['Accepting new result. Error: ' num2str(bestError)...
                '  Average sparsity: ' num2str(bestSparsity/L)...
                '  Max Sparsity: ' num2str(max(sum(p1EM)))])
        end
        %keyboard
    end
    tEMGAMP = toc(tstart);
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'EM-BiG-AMP'; %#ok<*AGROW>
    results{loc}.err = estHistEM.errZ(end);
    results{loc}.time = tEMGAMP;
    results{loc}.errHist = estHistEM.errZ;
    results{loc}.timeHist = estHistEM.timing;
    results{loc}.dict = AhatOptEM;
    results{loc}.dictError = dictionary_error_function(results{loc}.dict);
    results{loc}.codingError =...
        coding_error_function(results{loc}.dict);
end




%% SPAMS


if trySpams
    
    %Build params
    spams_param = [];
    spams_param.K = N;
    spams_param.mode = 2;
    spams_param.lambda = 0.1/sqrt(N);
    spams_param.iter = 1000;
    
    
    %Trials
    bestSPAMSerror = inf;
    
    tstart = tic;
    for trial = 1:maxTrials
        
        %Do it   
        A_spamsTemp = mexTrainDL(Y,spams_param);
        
        
        SPAMSerrorTemp = dictionary_error_function(A_spamsTemp);
        
        %Update the estimate if this result was superior
        if SPAMSerrorTemp < bestSPAMSerror
            SPAMSerror = SPAMSerrorTemp;
            bestSPAMSerror = SPAMSerror;
            A_spams = A_spamsTemp;
            disp(['Updating Solution. Error was: '...
                num2str(SPAMSerror) ' dB'])
        end
        
    end
    tspams = toc(tstart);
    
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'SPAMS'; %#ok<*AGROW>
    results{loc}.err = dictionary_error_function(A_spams);
    results{loc}.time = tspams;
    results{loc}.errHist = results{loc}.err;
    results{loc}.timeHist = zeros(size(results{loc}.errHist));
    results{loc}.dict = A_spams;
    results{loc}.dictError = dictionary_error_function(results{loc}.dict);
    results{loc}.codingError =...
        coding_error_function(results{loc}.dict);
    
    
end



%% ER-SpUD


if tryErspud
    
    
    %Call it
    tic
    %[Aerspud,Xerspud] = ER_SpUD_SC(Y);
    [Aerspud,Xerspud]=dl_spud(Y);   %#ok<NASGU>
    tErspud = toc;
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'ER-SpUD (proj)'; %#ok<*AGROW>
    results{loc}.err = dictionary_error_function(Aerspud);
    results{loc}.time = tErspud;
    results{loc}.errHist = results{loc}.err;
    results{loc}.timeHist = zeros(size(results{loc}.errHist));
    results{loc}.dict = Aerspud;
    results{loc}.dictError = dictionary_error_function(results{loc}.dict);
    results{loc}.codingError =...
        coding_error_function(results{loc}.dict);
    
    
    
end


%% K-SVD

if tryKsvd
    
    %Build params
    ksvd_params = [];
    ksvd_params.data = Y;
    ksvd_params.Tdata = K;
    ksvd_params.dictsize = N;
    ksvd_params.iternum = 100;
    ksvd_params.exacty = 1;
    
    %Trials
    bestKSVDerror = inf;
    tstart = tic;
    for trial = 1:maxTrials
        
        %Do it
        
        [A_ksvdtemp,~,err_ksvdtemp] = ksvd(ksvd_params);
        
        
        %Fix error
        err_ksvdtemp = err_ksvdtemp.^2 * numel(Y);
        
        %Update the estimate if this result was superior
        if err_ksvdtemp(end) < bestKSVDerror
            bestKSVDerror = err_ksvdtemp(end);
            err_ksvd = err_ksvdtemp;
            A_ksvd = A_ksvdtemp;
            disp(['Updating Solution. Error was: '...
                num2str(10*log10(err_ksvd(end)/norm(Z,'fro')^2))])
        end
        
    end
    tksvd = toc(tstart);
    
    %Save results
    loc = length(results) + 1;
    results{loc}.name = 'K-SVD'; %#ok<*AGROW>
    results{loc}.err = 10*log10(err_ksvd(end)/norm(Z,'fro')^2);
    results{loc}.time = tksvd;
    results{loc}.errHist = 10*log10(err_ksvd/norm(Z,'fro')^2);
    results{loc}.timeHist = zeros(size(err_ksvd));
    results{loc}.dict = A_ksvd;
    results{loc}.dictError = dictionary_error_function(results{loc}.dict);
    results{loc}.codingError =...
        coding_error_function(results{loc}.dict);
    
    
end




%% Store the options structures in results
results{1}.optIn = optIn;
results{1}.trueDict = A;
results{1}.trueEncoding = X;


%% Show Results

if nargin == 0
    results{:} %#ok<NOPRT>
    disp('Note that dictError is the normalized error in dB for recovering')
    disp('the dictionary for each algorithm')
end

