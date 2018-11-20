%% Demonstration of AMP for SMLR
% This example shows how to use GAMP for multiclass linear classification
% via multinomial logistic regression.  SPA-SHyGAMP and MSA-SHyGAMP can be
% invoked using the "mcGAMP" function.


%% Data generation
% For this example, the data is generated as follows:
%
% 1.) $M$ $D$-ary class labels are drawn uniformly, i.e. $y_m \sim \mathcal{U}\{1,...,D\}$.
%
% 2.) The set of class means $\{\textbf{u}_d\}_{d=1}^D$, each in $R^N$, are
% drawn mutually orthogonal, $K$-sparse, have uniform support, and each has
% norm $c$.  This is achieved by zero-padding the columns of the left singular vectors
% of a $K\times K$ Gaussian random matrix, and then scaling by $c$.
%
% 3.) The feature variance $v$ is set in order
% for the data to have a desired Bayes error rate, and is a function of $c$ and $D$. 
%
% 4.) Finally, each feature vector $\textbf{a}_m | y_m \sim
% \mathcal{N}(\textbf{u}_{y_m}, v \textbf{I}_N)$.
%
% We note two things:
% 
% 1.) This data model is matched to the multinomial logistic activation function.
% 
% 2.) For a given weight matrix $\textbf{X}$, the test-error-rate can be
% computed using a procedure that involves MATLAB's $\texttt{mvncdf}$ command 
% (see eq. (82)-(83) in the paper for details).


% data parameters
N = 30000;     % number of features
M = 300;       % number of training samples
K = 30;        % number of discriminatory features
D = 4;         % number of classes
Pbayes = .15;  % Bayes error rate

[y_train, A_train, mu, v] = buildDatasets(D,M,N,K,Pbayes);

% A_train is an M x N matrix, where each row is a training feature vector.
% y_train is an M x 1 vector, where each element is the class label
% corresponding to a row of A_train.
% mu is an N x D matrix, where each column is the mean of the D'th class.
% v is the cloud variance. 

xBayes = mu; % Bayes optimal classifier

%% SPA-SHyGAMP 

% SPA-SHyGAMP finds an approximation to the probability or error minimizing
% classifier.


% the function mcGAMP runs SPA-SHyGAMP by defualt.  
% calling mcGAMP with only y and A as input arguments uses the default
% algorithm parameters.  Optionally, one can use
% mcGAMP(y,A,opt_mc,opt_gamp), where opt_mc = MCOpt() and opt_gamp is a
% structure of gamp options.  Note that the default GAMP options in mcGAMP
% (eg the stepsize) are not the same as the default GAMP options.
estFin = mcGAMP(y_train,A_train);

% evaluate test-error-rate
test_error_SPA = testErrorRate(estFin.xhat, mu, v);

fprintf('SPA-SHyGAMP test error rate is %.3f\n',test_error_SPA)

% rerun SPA-SHyGAMP, but with plots on
opt_mc = MCOpt();
opt_mc.plot_hist = 1; % number indicates figure number
opt_mc.plot_wgt = 2; 

% must also specify model in order to compute error within mcGAMP
opt_mc.x_bayes = mu; 
opt_mc.mu = mu;
opt_mc.v = v;
opt_mc.Pbayes = Pbayes;

% call mcGAMP, now with the structure opt_mc as the thrid input
mcGAMP(y_train, A_train, opt_mc);


%% Plot descriptions 
% The first set of plots show the first 500 elements of selected weight vectors, with their indices
% ordered according to the sorted order of the absolute values of $\textbf{x}_d$-simple. 
% Quick descriptions of the weight vectors are:
%
% 1.) $\textbf{X}$-simple is the class-conditional empirical mean of 
% $\textbf{A}$.  In other words, $\textbf{x}_d = \frac{1}{M_d} \sum_{m:y_m=d} \textbf{a}_m$, 
% where $M_d$ is the number of training examples in class $d$.
%  
% 2.) $\textbf{X}$-kterm is $\textbf{X}$-simple where the $\widehat{K}$
% largest elements (by absolute value) of each column are retained, with the remaining
% set to zero. $\widehat{K}$ is chosen based on $N$, $M$, and $D$, and is
% an estimate for the largest number of elements in $\textbf{X}$ that can
% be accurately learned (for details, see pg. 46 in the thesis).
%
% 3.) $\textbf{X}$-GAMP is the weight matrix returned by GAMP.
% 
% 4.) $\textbf{X}$-Bayes is the Bayes optimal weight matrix. 
%
% 
% The second plot shows the test-error-rate and the training-error-rate vs
% GAMP iteration for the various weight matrices shown in the first plot
% (although, note $\textbf{X}$-GAMP changes with each GAMP iteration, and 
% values plotted in the first figure are the final values of
% $\textbf{X}$-GAMP).


%% MSA-SHyGAMP 

% MSA-SHyGAMP solves the traditional ell_1 regularized objective.

opt_mc = MCOpt();
opt_mc.SPA = false; % set this option to false to run MSA-SHyGAMP
estFin = mcGAMP(y_train,A_train,opt_mc);

% evaluate test-error-rate
test_error_MSA = testErrorRate(estFin.xhat, mu, v);

fprintf('MSA-SHyGAMP test error rate is %.3f\n',test_error_MSA)

% rerun SPA-SHyGAMP with plots on 
opt_mc = MCOpt();
opt_mc.SPA = false;
opt_mc.plot_hist = 3; 
opt_mc.plot_wgt = 4; 
opt_mc.x_bayes = mu; 
opt_mc.mu = mu;
opt_mc.v = v;
opt_mc.Pbayes = Pbayes;

mcGAMP(y_train, A_train, opt_mc);


%% SPA-SHyGAMP with emprical test data

% Now we will demonstrate using mcGAMP with empirical test data.

% first, generate test data
M_test = 1e4;
A_test = repmat(mu,1,ceil(M_test/D));
A_test = A_test(:,1:M_test)' + sqrt(v) * randn(M_test,N);
y_test = repmat(1:D,1,ceil(M_test/D));
y_test = y_test(1:M_test)';

opt_mc = MCOpt();
opt_mc.plot_hist = 5; 
opt_mc.plot_wgt = 6; 
opt_mc.A_test = A_test; % put the test features in opt_mc.A_test
opt_mc.y_test = y_test; % put the test features in opt_mc.y_test

mcGAMP(y_train, A_train, opt_mc);





