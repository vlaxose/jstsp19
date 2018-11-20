% TestSVMGAMP
%
% This script will run a single test of the performance of GAMP on a sparse
% binary classification problem using the HingeLossEstimOut class.  In this
% test, it is assumed that a length-M binary vector, y (y(m) \in {0,1}), 
% is associated with a group of N features, organized into an M-by-N 
% feature matrix, A.  The entries of A are drawn i.i.d. Normal(0, 1/M).  
% The relationship between the feature matrix, A, and the class labels, y, 
% is determined by a synthetically generated length-N "true" weight vector,
% x.  Specifically, this relationship is governed by the equation
%               y = (1/2) * ( sgn(A*x + w) + 1 ),
% i.e., y(m) = 1 when A(m,:)*x + w(m) is greater than 0, otherwise y(m) =
% 0.  The weight vector, x, is a sparse vector whose elements are drawn as
% i.i.d. Bernoulli-Gaussian random variables, with standard-normal non-zero
% entries, i.e.,
%       p(x(n)) = (1 - K/N)*delta(x(n)) + (K/N)*Normal(x(n); 0, 1),
% where K < M is the number of important (discriminative) features.
% Entries of additive class label noise, w(m), are i.i.d. 
% Normal(0, sigma2).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 10/25/12
% Change summary: 
%       - Created (10/25/12; JAZ)
% Version 0.2
%

Stream = RandStream.getGlobalStream;
SavedState = Stream.State;      % Save the current RNG state?
Stream.State = SavedState;      % Reset the RNG to previous initial state?

%% Define test parameters

% Problem setup
N = 512;            % # of features
NoverM = 3;         % Ratio of # of features-to-# of training samples
MoverK = 20;         % Ratio of # of training samples-to-# of "active" features
sigma2 = 0;         % Variance of corrupting label noise
varyK = false;      % Draw K from binomial dist. (true) or keep fixed (false)
Mtest = 256;        % # of test set samples for computing error rate

map_gamp = false;    % Run MMSE (false) or MAP (true) version of GAMP?

% EstimIn parameters
sigprior = 'bg';    % Choice of EstimIn object type.  Valid options are
                    % 'gaussian' for an iid Gaussian prior (the traditional
                    % SVM assumption), 'bg' for a Bernoulli-Gaussian 
                    % object, 'laplace' for a SoftThreshEstimIn (Laplacian)
                    % object, 'ellp'  an EllpEstimIn (ell-p minimization) 
                    % object, or 'elastic' for an Elastic Net 
                    % regularization penalty
svm_param = .15;      % Under a Gaussian signal prior, this parameter acts as
                    % a regularization parameter by controlling the
                    % (inverse) variance of the Gaussian prior
lagrangian = 1e0;   % Lagrangian penalty term for SoftThreshEstimIn and
                    % EllpEstimIn (lambda)
elastic_lam = [1e-2, 1e-2];   % [ell-1, ell-2] penalty terms for elastic net
lagrangians = logspace(-2, 1, 15);      % Cross-val choices
CrossVal = false;    % Crossval to select value of Lagrangian?
pnorm = 0.75;       % Choice of p-norm for EllpEstimIn


% Specify GAMP options
GAMPopts = GampOpt();
GAMPopts.adaptStep = true;
GAMPopts.nit = 150;
switch map_gamp
    case false
        GAMPopts.stepMin = 0;
    case true
        GAMPopts.stepMin = 1e-1;
end
GAMPopts.tol = 0;
GAMPopts.varNorm = true;

% Options for comparison against MATLAB's SVM algorithm
RunSVM = true;                  % Include comparison against MATLAB SVM?

% L1-LR cross-val options
RunL1LR = false;                 % Include L1-LR in comparison?
CrossValL1LR = false;            % Use x-validation?
Gammas = logspace(-2, 1, 20);   % Lagrangian values to try
Nfolds = 10;                    % # of x-val folds to avg error over

% TFOCS cross-val options
RunTFOCS = false;                % Include TFOCS in comparison?
CrossValTFOCS = false;           % Use x-validation? (Same x-val params as L1-LR)

RunSLR = false;                 % Include SLR in comparison?


%% Generate a synthetic trial

Mtrain = round(N / NoverM);      % # of training samples
K = round(Mtrain / MoverK);      % # of "active" (discriminative) features

% First construct the "true" weight vector that yields the class labels
switch varyK
    case true
        x = zeros(N,1);
        K_true = binornd(N, K/N);   % Draw K from binomial distribution
        active_features = randperm(N, K_true);  % Indices of discriminative features
        x(active_features) = randn(K_true,1);
    case false
        x = zeros(N,1);
        K_true = K;     % Fix K_true to equal average K
        active_features = randperm(N, K_true);  % Indices of discriminative features
        x(active_features) = randn(K_true,1);
    otherwise
        error('Unrecognized option: varyK')
end


% Now construct the label noise vector, the random feature matrix, and true
% class labels for the training set
A_train = (1/sqrt(Mtrain))*randn(Mtrain,N);         % Feature matrix
w_train = sqrt(sigma2)*randn(Mtrain,1);             % Label noise
y_train = (1/2)*(sign(A_train*x + w_train) + 1);	% True binary {0,1} class labels

% Finally, construct the label noise vector, the random feature matrix, and
% true class labels for the training set
A_test = (1/sqrt(Mtrain))*randn(Mtest,N);       % Feature matrix (use training variance)
w_test = sqrt(sigma2)*randn(Mtest,1);           % Label noise
y_test = (1/2)*(sign(A_test*x + w_test) + 1);	% True binary {0,1} class labels


%% Recover with GAMP

switch lower(sigprior)
    case 'gaussian'
        % Build the EstimIn object for a Gaussian signal prior
        EstimIn = AwgnEstimIn(zeros(N,1), 1/2/svm_param*ones(N,1), map_gamp);
    case 'bg'
        % Build the EstimIn object for a Bernoulli-Gaussian signal prior
        EstimIn = AwgnEstimIn(zeros(N,1), 1/2/svm_param*ones(N,1), map_gamp);
        EstimIn = SparseScaEstim(EstimIn, K/N);
    case 'laplace'
        % Build the EstimIn object for a Laplacian signal prior
        EstimIn = SoftThreshEstimIn(lagrangian, map_gamp);
    case 'ellp'
        % Build the EstimIn object for ell-p (0 < p <= 1) minimization
        EstimIn = EllpEstimIn(lagrangian, pnorm);
    case 'elastic'
        % Build the EstimIn object for an elastic net prior
        EstimIn = ElasticNetEstimIn(elastic_lam(1), elastic_lam(2), map_gamp);
    otherwise
        error('Unrecognized choice: sigprior')
end

% Build the EstimOut object for a hinge loss channel
EstimOut = HingeLossEstimOut(y_train, map_gamp);

% Set additional GAMP options
switch map_gamp
    case false
        GAMPopts.shat0 = ones(Mtrain,1);
    case true
        GAMPopts.shat0 = 2*(y_train - 1/2);
end

% Execute GAMP
if CrossVal
    GAMPopts.shat0 = [];
    [OptLambda, XValErrs] = LearnLaplaceLambda(A_train, y_train, ...
        lagrangians, TurboOpt('Signal', Laplacian('lambda', lagrangian), ...
        'Observation', LogitChannel('Scale', alpha, 'Version', 'map'), ...
        'RunOptions', RunOptions('smooth_iters', 1, 'min_iters', 1)), ...
        GAMPopts, Nfolds);
    lagrangian = OptLambda;
    EstimIn.lambda = lagrangian;
end
[x_hat, x_hat_var, rhatFinal, rvarFinal, shatFinal, svarFinal, ...
    zhatFinal,zvarFinal,estHist] = ...
    gampEst(EstimIn, EstimOut, A_train, GAMPopts);
% x_hat = x_hat(1:N);


%% Train SVM using MATLAB's built-in SVM learning algorithm

if RunSVM
    % Run MATLAB SVM training algorithm
    SVMstruct = svmtrain(A_train, y_train, 'autoscale', false, ...
        'boxconstraint', 1/2/svm_param, 'kernel_function', 'linear', ...
        'method', 'QP');
end


%% Train L1-LR algorithm, using cross-validation to select Lagrangian

if RunL1LR
    if CrossValL1LR
        % Cross-validate over choice of gamma
        [GammaOpt, XValErrsL1LR] = LearnGammaL1LR(A_train, y_train, ...
            Gammas, Nfolds);
    else
        GammaOpt = lagrangian^2;    % Use same value of gamma as turboGAMP
    end
    
    % Use optimal gamma to train a final classifier on the full training set
    [X_l1lr, ~, ~, ~, ~, Ptr_l1lr, Pte_l1lr] = ...
        biclsfy_l1slrc(A_train, y_train, A_test, y_test, GammaOpt, ...
        'usebias', 0, 'displaytext', 0);
end


%% Train TFOCS algorithm, using cross-validation to select Lagrangian

if RunTFOCS
    if CrossValTFOCS
        % Cross-validate over choice of gamma
        [GammaOptTfox, XValErrsTfox] = LearnGammaTFOCS(A_train, y_train, ...
            Gammas, Nfolds);
    else
        GammaOptTfox = lagrangian;	% Use same value of gamma as turboGAMP
    end
    
    % Use optimal gamma to train a final classifier on the full training set
    SmoothFunHan = tfunc_scale(smooth_logLLogistic(y_train), -1);
    AtrainFunHan = linop_matrix(A_train);
    NonSmoothFunHan = prox_l1(GammaOptTfox);
    X_tfox = tfocs(SmoothFunHan, AtrainFunHan, NonSmoothFunHan, zeros(N,1));
end


%% Train SLR algorithm

if RunSLR
    [X_slr, ~, ~, ~, ~, ~, Ptr_slr, Pte_slr] = ...
        biclsfy_slrvar(A_train, y_train, A_test, y_test, ...
        'usebias', 0, 'displaytext', 0);
end


%% Compute error metrics

y_hat_train = double(sign(A_train*x_hat) > 0);
y_hat_test = double(sign(A_test*x_hat) > 0);

train_error = sum(y_hat_train ~= y_train) / Mtrain;
test_error = sum(y_hat_test ~= y_test) / Mtest;

fprintf('%% of class 1 in training set: %1.1f\n', 100*sum(y_train == 1)/Mtrain)
fprintf('%% training samples misclassified: %1.1f\n', 100*train_error)
fprintf('%% test samples misclassified: %1.1f\n', 100*test_error)


%% Check to see if GAMP sol'n satisfies ell-2 convergence criterion

if strcmpi(sigprior, 'gaussian')
    PMonesY = 2*(y_train - 1/2);    % {0,1} -> {-1,1}
    vec = (A_train' * (PMonesY.*(PMonesY.*(A_train*x_hat) < 1)));
    criterion = Mtrain/N * sqrt(norm(vec,2)^2 / 4 / norm(x_hat,2)^2);
    
    fprintf('ell-2 convergence criterion value: %g\n', criterion)
end


%% Compute error metrics of SVM

if RunSVM
    % Have MATLAB estimate class labels
    y_train_svm = svmclassify(SVMstruct, A_train);
    y_test_svm = svmclassify(SVMstruct, A_test);
    
    train_error_svm = sum(y_train_svm ~= y_train) / Mtrain;
    test_error_svm = sum(y_test_svm ~= y_test) / Mtest;
    
    fprintf('%% training samples misclassified (SVM): %1.1f\n', 100*train_error_svm)
    fprintf('%% test samples misclassified (SVM): %1.1f\n', 100*test_error_svm)
end


%% Compute error metrics of L1-LR

if RunL1LR
    % Estimate class labels (since L1-LR used feature scaling)
    [~, y_train_l1lr] = max(Ptr_l1lr, [], 2);
    [~, y_test_l1lr] = max(Pte_l1lr, [], 2);
    y_train_l1lr = y_train_l1lr - 1; y_test_l1lr = y_test_l1lr - 1;
    
    train_error_l1lr = sum(y_train_l1lr ~= y_train) / Mtrain;
    test_error_l1lr = sum(y_test_l1lr ~= y_test) / Mtest;
    
    fprintf('%% training samples misclassified (L1-LR): %1.1f\n', 100*train_error_l1lr)
    fprintf('%% test samples misclassified (L1-LR): %1.1f\n', 100*test_error_l1lr)
end


%% Compute error metrics of TFOCS

if RunTFOCS
    % Estimate class labels
    y_train_tfox = double(sign(A_train*X_tfox) > 0);
    y_test_tfox = double(sign(A_test*X_tfox) > 0);
    
    train_error_tfox = sum(y_train_tfox ~= y_train) / Mtrain;
    test_error_tfox = sum(y_test_tfox ~= y_test) / Mtest;
    
    fprintf('%% training samples misclassified (TFOCS): %1.1f\n', 100*train_error_tfox)
    fprintf('%% test samples misclassified (TFOCS): %1.1f\n', 100*test_error_tfox)
end


%% Compute error metrics of SLR

if RunSLR
    % Estimate class labels (since L1-LR used feature scaling)
    [~, y_train_slr] = max(Ptr_slr, [], 2);
    [~, y_test_slr] = max(Pte_slr, [], 2);
    y_train_slr = y_train_slr - 1; y_test_slr = y_test_slr - 1;
    
    train_error_slr = sum(y_train_slr ~= y_train) / Mtrain;
    test_error_slr = sum(y_test_slr ~= y_test) / Mtest;
    
    fprintf('%% training samples misclassified (SLR): %1.1f\n', 100*train_error_slr)
    fprintf('%% test samples misclassified (SLR): %1.1f\n', 100*test_error_slr)
end


%% Plot the truth and the recovery

figure(1); clf
stem(x_hat, 'r'); hold on;
if RunL1LR, stem(X_l1lr, 'g'); end; 
if RunTFOCS, stem(X_tfox, 'c'); end
if RunSLR, stem(X_slr, 'k'); end;  stem(x, 'b'); hold off
title('Plot of weight vectors (Train Error | Test Error)')
xlabel('Feature [n]'); ylabel('Feature weight')
leg_str = {sprintf('GAMP weight vector (%1.1f | %1.1f)', 100*train_error, ...
    100*test_error)};
if RunL1LR, leg_str(end+1) = {sprintf('L1-LR weight vector (%1.1f | %1.1f)', ...
        100*train_error_l1lr, 100*test_error_l1lr)}; end
if RunTFOCS, leg_str(end+1) = {sprintf('TFOCS weight vector (%1.1f | %1.1f)', ...
        100*train_error_tfox, 100*test_error_tfox)}; end
if RunSLR, leg_str(end+1) = {sprintf('SLR weight vector (%1.1f | %1.1f)', ...
        100*train_error_slr, 100*test_error_slr)}; end
leg_str(end+1) = {'"True" weight vector'};
legend(leg_str, 'Location', 'Best')

figure(2); clf
semilogy(-estHist.val);
title(sprintf(['%% training samples misclassified: %1.1f | ' ...
    '%% testing samples misclassified: %1.1f'], 100*train_error, ...
    100*test_error))
xlabel('Iteration'); ylabel('(Negative) GAMP "val" variable')

