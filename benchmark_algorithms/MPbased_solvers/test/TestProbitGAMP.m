% TestProbitGAMP
%
% This function runs a single test of the performance of GAMP on a sparse
% binary classification problem using the ProbitEstimOut class.  In this
% test, it is assumed that a length-M binary vector, y (y(m) \in {0,1}), is
% associated with a group of N features, organized into an M-by-N feature
% matrix, A.  The entries of A are drawn i.i.d. Normal(0, 1/M).  The
% relationship between the feature matrix, A, and the class labels, y, is
% determined by a synthetically generated length-N "true" weight vector, x.
% Specifically, this relationship is governed by the equation
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
% If run without any arguments, this function runs a default test and plots
% the recovery.
%
% SYNTAX
% [test_error, train_error] = TestProbitGAMP(N, NoverM, MoverK, ...
%                                            Mtest, probit_var, BGmean, ...
%                                            BGvar)
%
% INPUTS
% N             # of features
% NoverM        Ratio of #-of-features-to-#-of-training-examples
% MoverK        Ratio of #-of-training-examples-to-#-of-discriminative-features
% Mtest         # of test set examples to use
% probit_var    ProbitEstimOut variance
% BGmean        Active Bernoulli-Gaussian mean
% BGvar         Active Bernoulli-Gaussian variance
%
% OUTPUTS
% test_error    Percentage of misclassified test set examples
% train_error   Percentage of misclassified training set examples
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/08/13
% Change summary: 
%       - Created (05/29/12; JAZ)
%       - Rewrote script as a function (01/03/13; JAZ)
% Version 0.1
%

function [test_error, train_error] = TestProbitGAMP(N, NoverM, MoverK, ...
    Mtest, probit_var, BGmean, BGvar)

if nargin == 0

    Stream = RandStream.getGlobalStream;
    SavedState = Stream.State;      % Save the current RNG state?
    Stream.State = SavedState;      % Reset the RNG to previous initial state?

    % Define default test parameters
    N = 1000;            % # of features
    NoverM = 2;         % Ratio of # of features-to-# of training samples
    MoverK = 20;         % Ratio of # of training samples-to-# of "active" features
    Mtest = 256;        % # of test set samples for computing error rate

    % Probit regression function parameters
    % The output channel is specified as p(y(m) = 1 | z) = Phi((z-m)/sqrt(v)),
    % where Phi(*) is the CDF of a normal distribution with mean m and variance
    % v.
    probit_var = .01;     % Variance v of probit regression function
    
    % Bernoulli-Gaussian weight vector prior parameter defaults
    BGmean = 0;     % Mean of active Gaussian
    BGvar = 1;      % Variance of active Gaussian
elseif nargin == 7
    % Correct # of arguments provided
    if isempty(N) || isempty(NoverM) || isempty(MoverK) || isempty(Mtest) ...
            || isempty(probit_var) || isempty(BGmean) || isempty(BGvar)
        error('TestProbitGAMP is missing argument(s)')
    end
else
    error('TestProbitGAMP is missing argument(s)')
end


%% Generate a synthetic trial

sigma2 = 0;         % Variance of corrupting label noise
varyK = true;      % Draw K from binomial dist. (true) or keep fixed (false)
probit_mean = 0;    % Mean m of probit regression function
Mtrain = round(N / NoverM);      % # of training samples
K = round(Mtrain / MoverK);      % # of "active" (discriminative) features

% First construct the "true" weight vector that yields the class labels
switch varyK
    case true
        x = zeros(N,1);
        K_true = binornd(N, K/N);   % Draw K from binomial distribution
        active_features = randperm(N, K_true);  % Indices of discriminative features
        x(active_features) = BGmean + sqrt(BGvar)*randn(K_true,1);
    case false
        x = zeros(N,1);
        K_true = K;     % Fix K_true to equal average K
        active_features = randperm(N, K_true);  % Indices of discriminative features
        x(active_features) = BGmean + sqrt(BGvar)*randn(K_true,1);
    otherwise
        error('Unrecognized option: varyK')
end

% Now construct the label noise vector, the random feature matrix, and true
% class labels for the training set
A_train = (1/sqrt(Mtrain))*randn(Mtrain,N);         % Feature matrix
if nargin == 0
    w_train = sqrt(sigma2)*randn(Mtrain,1);             % Label noise
    y_train = (1/2)*(sign(A_train*x + w_train) + 1);	% True binary {0,1} class labels
else
    % Generate according to assumed discriminative model
    y_train = double(normcdf(A_train*x, 0, sqrt(probit_var)) > rand(Mtrain,1));
end

% Finally, construct the label noise vector, the random feature matrix, and
% true class labels for the training set
A_test = (1/sqrt(Mtrain))*randn(Mtest,N);       % Feature matrix (use training variance)
if nargin == 0
    w_test = sqrt(sigma2)*randn(Mtest,1);           % Label noise
    y_test = (1/2)*(sign(A_test*x + w_test) + 1);	% True binary {0,1} class labels
else
    % Generate according to assumed discriminative model
    y_test = double(normcdf(A_test*x, 0, sqrt(probit_var)) > rand(Mtest,1));
end


%% Recover with GAMP

% Build the EstimIn object for a Bernoulli-Gaussian signal
EstimIn = AwgnEstimIn(BGmean*ones(N,1), BGvar*ones(N,1));
EstimIn = SparseScaEstim(EstimIn, K/N);

% Build the EstimOut object for a probit regression channel
EstimOut = ProbitEstimOut(y_train, probit_mean, probit_var);
% EstimOut = LogitEstimOut(y_train);

% Specify GAMP options
GAMPopts = GampOpt();
GAMPopts.adaptStep = false;
GAMPopts.nit = 25;

% Execute GAMP
[x_hat, x_hat_var, ~, ~, ~, ~, zhat, zvar] = ...
    gampEst(EstimIn, EstimOut, A_train, GAMPopts);


%% Compute error metrics

y_hat_train = (1/2) * (sign(A_train*x_hat) + 1);
y_hat_test = (1/2) * (sign(A_test*x_hat) + 1);

train_error = sum(y_hat_train ~= y_train) / Mtrain;
test_error = sum(y_hat_test ~= y_test) / Mtest;

if nargin == 0
    fprintf('%% of class 1 in training set: %1.1f\n', 100*sum(y_train == 1)/Mtrain)
    fprintf('%% training samples misclassified: %1.1f\n', 100*train_error)
    fprintf('%% test samples misclassified: %1.1f\n', 100*test_error)
end


%% Plot the truth and the recovery

if nargin == 0
    figure(1); clf
    stem(x, 'b'); hold on; stem(x_hat, 'r'); hold off
    title(sprintf(['%% training samples misclassified: %1.1f | ' ...
        '%% testing samples misclassified: %1.1f'], 100*train_error, ...
        100*test_error))
    xlabel('Feature [n]'); ylabel('Feature weight')
    legend('"True" weight vector', 'GAMP weight vector', 'Location', 'Best')
end

end     % EOF