% ProbitPhasePlane
%
% The purpose of this test is to compare empirical misclassification rates
% against those calculated from GAMP's state evolution across the
% sparsity-undersampling plane, i.e., training and test set error rates are
% calculated and averaged over independent trials for varying undersampling
% rates (Mtrain/N) and sparsity rates (E[K]/Mtrain), where E[K] denotes the
% expected number of "true" discriminative features (the number of non-zero
% entries of the synthetic "true" separating hyperplane normal vector, x).
%
% This test will generate synthetic training and testing datasets for a
% binary (0,1) classification problem. In particular, this script will 
% generate a training dataset in which a binary vector of length 
% Mtrain-by-1, Y_train (Y_train(m) \in {0,1}), is associated with a group
% of N features, organized into an Mtrain-by-N feature matrix, A_train.  
% The entries of A_train are drawn i.i.d. Normal(0, 1/Mtrain).  The 
% relationship between the feature matrix, A_train, and the class labels, 
% Y_train, is determined by a synthetically generated N-by-1 "true" weight 
% vector, X_true, whose entries are drawn i.i.d. Bernoulli-Gaussian, i.e.,
%     p(X(n)) = (1 - K/N)*delta(X(n)) + (K/N)*Normal(X(n); m, v),
% where K < M is the number of important (discriminative) features. This 
% relationship is governed by the equation:
%   p(Y_train(m) = 1 | Atrain, X_true) = Phi( A_train(m,:)*X_true / NU),
% where Phi(.) denotes the CDF of the standard normal distribution and NU
% is a variance parameter that controls the shape of the probit sigmoid.
% A companion testing dataset, with an Mtest-by-1 binary class label 
% vector, called Y_test, and a corresponding testing feature matrix of 
% dimension Mtest-by-N, called A_test, is also produced according to 
% the same relationship described above, using the same statistics to 
% generate A_test as was used to generate A_train.
%
% SYNTAX:
% ProbitPhasePlane(N, Mtest, probit_var, BGmean, BGvar, N_trials, fid)
%
% INPUTS:
%   N               Number of features in the datasets [Default: 512]
%   Mtest           Number of testing samples for performance evaluation
%                   [Default: 256]
%   probit_var      Variance of the probit channel (NU) [Default: 1e-1]
%   BGmean          Active mean of Bernoulli-Gaussian (m) [Default: 0]
%   BGvar           Active variance of Bernoulli-Gaussian (v) [Default: 1]
%   N_trials     	Number of trials to avg. results over per (beta,delta)
%                   [Default: 5]
%   fid             Set == 1 to write progress to screen, or anything else
%                   to write progress to file [Default: 1]
%
% OUTPUTS:
% The empirical and state evolution computed training and test 
% misclassification rates for each algorithm, stored in a .MAT file whose 
% filename is created at runtime based on input parameters and the 
% time/date of the start of execution
% Suffix keywords:
%   _SE             State evolution error rates
%   _Emp            Empirical error rates
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/08/13
% Change summary: 
%		- Created (01/03/13; JAZ)
% Version 0.1
%

function ProbitPhasePlane(N, Mtest, probit_var, BGmean, BGvar, N_trials, fid)

%% Declare the parameters for the signal model and test setup

if nargin == 0
    N = 1024;             	% # features                                [dflt=1024]
    Mtest = 256;         	% # of testing samples                      [dflt=256]
    probit_var = 1e-2;    	% Variance v of probit regression function  [dflt=1e-2]
    BGmean = 0;             % Active Bernoulli-Gaussian mean            [dflt=0]
    BGvar = 1;              % Active Bernoulli-Gaussian variance        [dflt=1]
    N_trials = 5;         	% Number of trials to avg. results over per (beta,delta)
    fid = 0;                % Write progress to screen?                 [dflt=1]
elseif nargin ~= 7
    error('ProbitPhasePlane: Incorrect # of arguments supplied')
end

% Probit regression function parameters
% The output channel is specified as p(y(m) = 1 | z) = Phi((z-m)/sqrt(v)),
% where Phi(*) is the CDF of a normal distribution with mean m and variance
% v.
probit_mean = 0;                % Mean m of probit regression function

% GAMP options
nit = 25;                       % # of GAMP iterations
adaptStep = false;              % No adaptive step-sizing

% Test setup parameters
Q = 20;                            	% Grid sampling density
beta = linspace(0.10, 0.95, Q);  	% Sparsity ratio E[K]/M
delta = linspace(0.05, 0.95, Q);  	% Undersampling ratio M/N
N_beta = length(beta);
N_delta = length(delta);

% Filename for saving data
savefile = [datestr(now, 'ddmmmyyyy_HHMMSSFFF') '_ProbitPhasePlane_N_' ...
    num2str(N) '_Mtest_'  num2str(Mtest) '_probit_var_' num2str(probit_var) ...
    '_BGmean_' num2str(BGmean) '_BGvar_' num2str(BGvar) '.mat'];

% Randomly seed the RNG and save its state for reference purposes
NewStream = RandStream.create('mrg32k3a', 'Seed', sum(100*clock));
RandStream.setGlobalStream(NewStream);
savedState = NewStream.State;


%% Execute the phase plane test

% Add required directories to path
addpath('../../main');
addpath('../../stateEvo');
addpath('../../test');

% Create arrays to store the error rates, TNMSEs and NSERs
TrainErr_SE = NaN(N_beta, N_delta, 1);          % Train error rate for state evo
TrainErr_Emp = NaN(N_beta, N_delta, N_trials);	% Train error rate for empirical
TestErr_SE = NaN(N_beta, N_delta, 1);           % Test error rate for state evo
TestErr_Emp = NaN(N_beta, N_delta, N_trials); 	% Test error rate for empirical
Run_SE = NaN(N_beta, N_delta, 1);               % Runtime for state evo
Run_Emp = NaN(N_beta, N_delta, N_trials);     	% Runtime for empirical

% GAMP options
GAMPopt = GampOpt();
GAMPopt.adaptStep = adaptStep;
GAMPopt.varNorm = false;        % Don't normalize variance messages
GAMPopt.uniformVariance = false;% Use vector message variances
GAMPopt.nit = nit;              % Set # of GAMP iterations
GAMPopt.tol = -1;           	% No early termination

try % Matlabpool opening
    matlabpool close force local
    % Open a matlabpool, if one is not already open
    if matlabpool('size') == 0
        matlabpool open
        fprintf('Opened a matlabpool with %d workers\n', matlabpool('size'));
    end
    
    % Set the RNGs of the parallel workers based on the common random seed
    % obtained above
    parfor i = 1 : matlabpool('size')
        RandStream.setGlobalStream(NewStream);
        pause(3);
    end
catch ME
    warning('matlabpool:open:fail', 'Unable to open a matlabpool: %s', ...
        ME.message)
end

time = clock;   % Start the stopwatch
for n = 1:N_trials
    for b = 1:N_beta
        parfor d = 1:N_delta
            % If running in parallel, for reproducibility we should have
            % each worker move the RNG stream to a predefined substream
            % based on the iteration index
            Stream = RandStream.getGlobalStream;
            Stream.Substream = (n-1)*N_beta*N_delta + (b-1)*N_delta + d;
            
            % First determine the ratios for N/M and M/K
            NoverM = 1 / delta(d);
            MoverK = 1 / beta(b);
            
            % *************************************************************
            % ***        Run empirical synthetic probit recovery        ***
            % *************************************************************
            tic;
            [TestErr_Emp(b,d,n), TrainErr_Emp(b,d,n)] = ...
                TestProbitGAMP(N, NoverM, MoverK, Mtest, probit_var, ...
                BGmean, BGvar);
            Run_Emp(b,d,n) = toc;
            
            
            
            % *************************************************************
            % ***          Compute state evolution predictions          ***
            % *************************************************************
            if n == 1,  % Only need to compute SE predictions once per (beta,delta)
                % Calculate certain quantities based on model parameters
                Mtrain = round(N / NoverM);     % # of training samples
                K = round(Mtrain / MoverK);     % # of "active" (discriminative) features
                sparseRat = K/N;                % Bernoulli-Gaussian activity probability
                
                % Generate input estimation class
                inputEst0 = AwgnEstimIn(BGmean, BGvar);
                inputEst = SparseScaEstim(inputEst0, sparseRat);
                
                % SE options
                SEopt = struct('beta', []); % Create empty structure
                SEopt.beta = N/Mtrain;   	% Measurement ratio
                SEopt.Avar = Mtrain/Mtrain; % Mtrain*var(A(i,j))
                SEopt.tauxieq = true;       % Forces taur(t) = xir(t)
                SEopt.verbose = false;      % Work silently
                SEopt.nit = nit;            % Number of state evolution iterations
                
                % Construct input estimator averaging function for the Gauss-Bernoulli source
                Nx = 400;                           % # of discrete integration points
                Nw = 400;                           % # of "noise" points
                umax = sqrt(2*log(Nx/2));
                u = linspace(-umax, umax, Nx)';
                px1 = exp(-u.^2/2);                 % Compute p(x | x ~= 0)...
                px1 = px1 / sum(px1);               % ...and normalize
                x1 = BGmean + sqrt(BGvar)*u;        % Discrete non-zero x points
                x = [0; x1];                        % Discrete x points
                px = [1-sparseRat; sparseRat*px1];  % p(x)
                inAvg = IntEstimInAvg(inputEst, x, px, Nw);     % Build SE EstimInAvg object
                
                % Construct output estimator averaging function for probit channel
                Np = 500;       % # of discrete P values
                Ny = 2;         % # of discrete Y values (only {0,1} matter)
                Nz = 500;       % # of discrete Z values
                outAvg = ProbitStateEvoEstimOut(Np, Ny, Nz, 0, probit_var, false);
                
                % Run GAMP SE analysis
                tic;
                [~, histSE] = gampSE(inAvg, outAvg, SEopt);
                Run_SE(b,d,n)  = toc;
                
                % Use computed SE quantities to numerically integrate the
                % expression for test set error rate
                zt = linspace(-2,2,1e4);
                zht = linspace(-2,2,1e4);
                dz = zt(2) - zt(1);
                Avar = SEopt.Avar / Mtrain;
                xcovSE = histSE.xcov;
                varZt = N*Avar*xcovSE(1,1,end);
                varZht = N*Avar*xcovSE(2,2,end);
                varZtZht = N*Avar*xcovSE(1,2,end);
                Kz = [varZt, varZtZht; varZtZht, varZht];
                err = 0;
                for i = 1:numel(zht)
                    if zht(i) < 0
                        err = err + (dz^2) * normcdf(zt', probit_mean, sqrt(probit_var))' * ...
                            mvnpdf([zt', repmat(zht(i), numel(zt), 1)], [0, 0], Kz);
                    else
                        err = err + (dz^2) * normcdf(-zt', probit_mean, sqrt(probit_var))' * ...
                            mvnpdf([zt', repmat(zht(i), numel(zt), 1)], [0, 0], Kz);
                    end
                end
                
                % Store SE-estimated test set error rate
                TestErr_SE(b,d,n) = err;
            end
            
            
            if fid ~= 1, 
                fid1 = fopen([savefile(1:end-3) 'txt'], 'w')
                fprintf(fid1, 'Progress: %g\n', (N_beta*(b-1) + d) / N_beta / N_delta);
                fclose(fid1);
            else
                fprintf('Progress: %g\n', (N_beta*(b-1) + d) / N_beta / N_delta);
            end
        end     % delta (M/N)
    end         % beta (E[K]/M)
    
    % Having completed one sweep of the (beta, delta) grid, estimate the
    % time remaining and save the temporary data
    est_time = (N_trials - n)*(etime(clock, time)/n)/60/60;
    if fid ~= 1, 
        fid2 = fopen([savefile(1:end-3) 'txt'], 'w');
        fprintf(fid2, '(%d/%d) Estimated time remaining: %3.2f hours\n', ...
            n, N_trials, est_time);
    	fclose(fid2);
    else
        fprintf('(%d/%d) Estimated time remaining: %3.2f hours\n', ...
            n, N_trials, est_time);
    end
    save(savefile);
end

if fid ~= 1, 
    fid3 = fopen([savefile(1:end-3) 'txt'], 'w');
    fprintf(fid3, 'Total time elapsed for test: %3.2f hours\n', ...
        etime(clock, time)/60/60);
	fclose(fid3); 
else
    fprintf('Total time elapsed for test: %3.2f hours\n', ...
        etime(clock, time)/60/60);
end

% Close the matlabpool
try
    matlabpool close
catch
    % No matlabpool to close
end

% Final save of data
save(savefile);
