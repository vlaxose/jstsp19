% TestKernelGAMP
%
% This script will run a single test of the performance of GAMP on a sparse
% binary classification problem using the KernelLinTrans class.  In this
% test, it is assumed that a length-M binary vector, y (y(m) \in {0,1}), is
% associated with a group of N features, organized into an M-by-N feature
% matrix, A.  In each row of A, the columns corresponding to K
% predetermined features will have entries drawn from class-conditional
% i.i.d. normal distributions, with a mean that depends on the particular
% class of each row.  The remaining columns will be drawn from a
% class-independent zero-mean i.i.d. normal distribution.
%
% Instead of training a classifier on the training feature matrix itself,
% the KernelLinTrans class will be used to transform the feature matrix
% into an inner product space defined by a user-specified kernel function.
% A linear classifier will then be trained on this kernel matrix.
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/10/12
% Change summary: 
%       - Created (12/10/12; JAZ)
% Version 0.2
%

Stream = RandStream.getGlobalStream;
SavedState = Stream.State;      % Save the current RNG state?
Stream.State = SavedState;      % Reset the RNG to previous initial state?

%% Define test parameters

% Problem setup
N = 512;                % # of features
NoverM = 4;             % Ratio of # of features-to-# of training samples
MoverK = 5;            % Ratio of # of training samples-to-# of "active" features
varyK = false;          % Draw K from binomial dist. (true) or keep fixed (false)
Mtest = 256;            % # of test set samples for computing error rate
classmeans = [-1, 1];   % Mean of class 0/1 informative features

% Kernel function parameters
kernel = 'gaussian';    % Choice of kernel ('linear', 'gaussian', or 'polynomial')
gamma = 0.05;            % Gaussian kernel scale parameter
offset = 1;             % Polynomial kernel offset parameter
degree = 2;             % Polynomial kernel degree

% EstimIn parameters
sigprior = 'bg';    % Choice of EstimIn object type.  Valid options are
                    % 'bg' for a Bernoulli-Gaussian object, 'laplace' for a
                    % SoftThreshEstimIn (Laplacian) object, 'ellp'  an
                    % EllpEstimIn (ell-p minimization) object, or 'elastic'
                    % for an Elastic Net regularization penalty
lagrangian = 1e0;   % Lagrangian penalty term for SoftThreshEstimIn and
                    % EllpEstimIn (lambda)
elastic_lam = [1e-2, 1e-2];   % [ell-1, ell-2] penalty terms for elastic net
lagrangians = logspace(-1, 1, 15);      % Cross-val choices
CrossVal = false;    % Crossval to select value of Lagrangian?
Nfolds = 5;         % # of cross-validation folds
pnorm = 0.75;       % Choice of p-norm for EllpEstimIn
    
% Probit regression function parameters
% The output channel is specified as p(y = 1 | z) = Phi((z-m)/sqrt(v)),
% where Phi(*) is the CDF of a normal distribution with mean m and variance
% v.
probit_mean = 0;    % Mean m of probit regression function
probit_var = 1e-2;  % Variance v of probit regression function
map_gamp = false;   % Run MMSE (false) or MAP (true) version of GAMP?

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


%% Generate a synthetic trial

Mtrain = round(N / NoverM);      % # of training samples
K = round(Mtrain / MoverK);      % # of "active" (discriminative) features
if varyK, K = binornd(N, K/N); end

% First construct the training and test binary class label vectors
y_train = (rand(Mtrain,1) > 1/2);
y_test = (rand(Mtest,1) > 1/2);

% WLOG, generate the training and test feature matrices with the first K
% columns containing the "informative" features
A_train = NaN(Mtrain,N);    A_test = NaN(Mtest,N);
A_train(y_train,1:K) = randn(sum(y_train),K) + classmeans(2);   % Class 1 informative features
A_test(y_test,1:K) = randn(sum(y_test),K) + classmeans(2);      % Class 1 informative features
A_train(~y_train,1:K) = randn(sum(~y_train),K) + classmeans(1); % Class 0 informative features
A_test(~y_test,1:K) = randn(sum(~y_test),K) + classmeans(1);    % Class 1 informative features
A_train(:,K+1:N) = randn(Mtrain,N-K);                           % Uninformative features
A_test(:,K+1:N) = randn(Mtest,N-K);                             % Uninformative features

% Now "kernelify" the training and test feature matrices
A_train_kern = KernelLinTrans(A_train, kernel, gamma, offset, degree);
A_test_kern = A_train_kern.feat2kern(A_test);


%% Recover with GAMP

switch lower(sigprior)
    case 'bg'
        % Build the EstimIn object for a Bernoulli-Gaussian signal prior
        EstimIn = AwgnEstimIn(zeros(Mtrain,1), ones(Mtrain,1), map_gamp);
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

% Build the EstimOut object for a probit regression channel
EstimOut = ProbitEstimOut(y_train, probit_mean, probit_var, map_gamp);

% Set additional GAMP options
switch map_gamp
    case false
        GAMPopts.shat0 = zeros(Mtrain,1);
        ver = 'mmse';
    case true
        GAMPopts.shat0 = 2*(y_train - 1/2);
        ver = 'map';
end

% Execute GAMP
[x_hat, x_hat_var, rhatFinal, rvarFinal, shatFinal, svarFinal, ...
    zhatFinal,zvarFinal,estHist] = ...
    gampEst(EstimIn, EstimOut, A_train_kern, GAMPopts);


%% Compute error metrics

y_hat_train = double(sign(A_train_kern.mult(x_hat)) > 0);
y_hat_test = double(sign(A_test_kern*x_hat) > 0);

train_error = sum(y_hat_train ~= y_train) / Mtrain;
test_error = sum(y_hat_test ~= y_test) / Mtest;

fprintf('%% of class 1 in training set: %1.1f\n', 100*sum(y_train == 1)/Mtrain)
fprintf('%% training samples misclassified: %1.1f\n', 100*train_error)
fprintf('%% test samples misclassified: %1.1f\n', 100*test_error)


%% Check to see if GAMP sol'n satisfies ell-infinity convergence criterion

if strcmpi(sigprior, 'laplace')
    PMonesY = 2*(y_train - 1/2);    % {0,1} -> {-1,1}
    C = PMonesY .* (A_train_kern.mult(x_hat)) ./ sqrt(probit_var);
    % Now compute the ratio normpdf(C)/normcdf(C)
    ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
    % Compute convergence vector
    cvec = A_train_kern.multTr(PMonesY.*ratio./sqrt(probit_var));

    fprintf('ell-infinity convergence norm: %g\n', norm(cvec, Inf))
elseif strcmpi(sigprior, 'elastic') && map_gamp
    PMonesY = 2*(y_train - 1/2);    % {0,1} -> {-1,1}
    C = PMonesY .* (A_train_kern.mult(x_hat)) ./ sqrt(probit_var);
    % Now compute the ratio normpdf(C)/normcdf(C)
    ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
    % Compute convergence vector
    deriv = PMonesY .* ratio ./ sqrt(probit_var);
    cvec = A_train_kern.multTr(deriv) - 2*elastic_lam(2)*x_hat;

    fprintf('ell-infinity convergence norm: %g\n', norm(cvec, Inf))
end


%% Plot the truth and the recovery

figure(1); clf
stem(x_hat, 'r');
title(sprintf(['%% training samples misclassified: %1.1f | ' ...
    '%% testing samples misclassified: %1.1f'], 100*train_error, ...
    100*test_error))
xlabel('Feature [n]'); ylabel('Feature weight')
legend('GAMP weight vector', 'Location', 'Best')

figure(2); clf
plot(estHist.val);
title(sprintf(['%% training samples misclassified: %1.1f | ' ...
    '%% testing samples misclassified: %1.1f'], 100*train_error, ...
    100*test_error))
xlabel('Iteration'); ylabel('GAMP "val" variable')