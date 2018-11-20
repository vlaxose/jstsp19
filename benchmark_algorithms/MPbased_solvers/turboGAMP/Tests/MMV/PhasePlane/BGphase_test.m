% BGPHASE_TEST        A test of EMturboGAMP for the MMV signal model,
% that, for a fixed choice of signal model parameters, will sweep the
% expected-number-of-active-coefficients-per-measurement ratio, (a.k.a. the
% normalized sparsity ratio (E[K]/M)), beta, and the
% number-of-measurements-per-number-of-unknowns ratio, (a.k.a. the
% undersampling ratio (M/N)), delta over ranges of values, while storing
% the timestep-averaged normalized mean squared error (TNMSE) at each
% (beta, delta) pair.  The TNMSE will also be recorded for a 
% support-aware Kalman smoother using genie_multi_frame_fxn, as well
% as a naive GAMP estimator that independently recovers the unknown signals 
% at each timestep without considering any relationship between the 
% recovered signals of adjacent timesteps.
%
% Amplitudes will be assumed to be zero-mean, and correlated by way of a
% Gauss-Markov process for each row, i.e., 
% GAMMA(n,t) = (1 - Alpha)*GAMMA(n,t-1) + Alpha*E(n,t),
% where pdf(E(n,t)) = Normal(0, Rho), where Rho) is chosen to maintain a
% steady-state variance of Sigma2.
%
% The noise will be AWGN.
%
% The signal coefficients will be marginally Bernoulli-Gaussian.
%
%
% SYNTAX:
% BGphase_test(N, T, alpha, CommonA, SNRmdB, N_trials, fid)
%
% INPUTS:
%   N               Number of unknowns at each timestep
%   T              	Number of timesteps - 1
%   Alpha         	Innovation rate of GAMMAs (0 < Alpha < 1)
%   CommonA         Use a common transform matrix at all timesteps?
%   SNRmdB         	Per-measurement SNR (in dB)
%   N_trials     	Number of trials to avg. results over per (beta,delta)
%   fid             Set == 1 to write progress to screen, or anything else
%                   to write progress to file
%
% OUTPUTS:
% This function will save the timestep-averaged NMSEs and BERs for a
% support-aware genie smoother, a naive BP recovery scheme, a naive 
% support-aware MMSE estimator, and the proposed multi-timestep recovery 
% algorithm in a .MAT file whose filename is created at runtime based on 
% input parameters and the time/date of the start of execution
% Suffix keywords:
%   _sks     	The support-aware genie smoother
%   _naive      The support-unaware, timestep-unaware naive GAMP estimator
%   _noem       EMturboGAMP w/o EM parameter estimation (perfect knowledge)
%   _turbo      EMturboGAMP estimator
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 02/22/12
% Change summary: 
%		- Created (02/20/12; JAZ)
% Version 0.2
%

function BGphase_test(N, T, Alpha, CommonA, SNRmdB, N_trials, fid)

%% Declare the parameters for the signal model and test setup

if nargin == 0
    N = 1024;             	% # parameters                              [dflt=1024]
    T = 6;                  % # of timesteps                            [dflt=5]
    Alpha = 0.05;           % Innovation rate of thetas (1 = total)     [dflt=0.10]
    CommonA = false;      	% Use a single A matrix for all timesteps?  [dflt=true]
    SNRmdB = 25;           	% Per-measurement SNR (in dB)               [dflt=15]
    N_trials = 50;         	% Number of trials to avg. results over per (beta,delta)
    fid = 1;                % Write progress to screen?                 [dflt=1]
end

warning('Using AMP for smoother instead of ABP');

% Declare additional static signal model parameters
A_type = 1;             % 1=iid normal, 2=rademacher, 3=subsampled DFT	[dflt=1]
Theta = 0;            	% Mean of active coefficients                   [dflt=0]
Sigma2 = 1;       	% Steady-state amplitude variance               [dflt=1]
eps = 1e-6;             % "Squelch" parameter for inactives             [dflt=1e-6]

% Algorithm execution parameters
smooth_iters = 15;    	% Number of forward/backward "smoothing" passes 
                        % to perform [dflt: 5]
min_iters = 5;          % Minimum number of smoothing iterations to perform [dflt: 5]
inner_iters = 25;      	% Number of inner GAMP iterations to perform,
                        % per forward/backward pass, at each timestep [dflt: 25]
alg = 2;                % Type of BP algorithm to use during equalization 
                        % at each timestep for SKS: (1) for standard BP, 
                        % (2) for AMP [dflt: 2]

% Test setup parameters
Q = 20;                            	% Grid sampling density
beta = linspace(0.10, 0.95, Q);  	% Sparsity ratio E[K]/M
delta = linspace(0.05, 0.95, Q);  	% Undersampling ratio M/N
N_beta = length(beta);
N_delta = length(delta);

% Filename for saving data
savefile = [datestr(now, 'ddmmmyyyy_HHMMSSFFF') '_BGphase_test_N_' ...
    num2str(N) '_T_' num2str(T) '_Alpha_'  num2str(Alpha) ...
    '_CommonA_' num2str(CommonA) '_SNRm_' num2str(SNRmdB) '.mat'];

% Randomly seed the RNG and save its state for reference purposes
NewStream = RandStream.create('mt19937ar', 'Seed', sum(100*clock));
RandStream.setDefaultStream(NewStream);
savedState = NewStream.State;


%% Execute the phase plane test

% Add SKS to path
path(path, '../../../../../../SparseSolvers/gampmatlab/trunk/code/main')
path(path, '../../../ClassDefs')
path(path, '../../../')
path(path, '../../../../../TimeEvolveSparse/Code/ClassDefs')
path(path, '../../../../../TimeEvolveSparse/Code/Functions')

% Create arrays to store the NMSEs and BERs
TNMSE_sks = NaN*ones(N_beta, N_delta, N_trials);  	% TNMSE for SKS
TNMSE_naive = NaN*ones(N_beta, N_delta, N_trials);	% TNMSE for naive GAMP
TNMSE_noem = NaN*ones(N_beta, N_delta, N_trials);	% TNMSE for turboGAMP
TNMSE_turbo = NaN*ones(N_beta, N_delta, N_trials); 	% TNMSE for EMturboGAMP
NSER_sks = NaN*ones(N_beta, N_delta, N_trials);     % NSER for sks
NSER_naive = NaN*ones(N_beta, N_delta, N_trials);  	% NSER for naive GAMP
NSER_noem = NaN*ones(N_beta, N_delta, N_trials);  	% NSER for turboGAMP
NSER_turbo = NaN*ones(N_beta, N_delta, N_trials);  	% NSER for EMturboGAMP
Run_sks = NaN*ones(N_beta, N_delta, N_trials);  	% Runtime for SKS
Run_naive = NaN*ones(N_beta, N_delta, N_trials);	% Runtime for naive GAMP
Run_noem = NaN*ones(N_beta, N_delta, N_trials);     % Runtime for turboGAMP
Run_turbo = NaN*ones(N_beta, N_delta, N_trials); 	% Runtime for EMturboGAMP

% % Open a matlabpool, if one is not already open
% if matlabpool('size') == 0
%     matlabpool open
%     disp(sprintf('Opened a matlabpool with %d workers', matlabpool('size')));
% end

time = clock;   % Start the stopwatch
for n = 1:N_trials
    for b = 1:N_beta
        for d = 1:N_delta
            % First determine the values of E[K] and M, and things that
            % depend on them
            M = round(delta(d)*N);      % Number of measurements at each timestep
            lambda = beta(b)*M/N;       % Pr{s_n(t) = 1}
            
            % *************************************************************
            % Generate a signal/measurement pair realization for this suite
            % *************************************************************
            % Create a structure to hold the problem suite parameters
            GenParamObj = GenParams(N, T, M, SNRmdB, A_type, 'real', CommonA);
            
            % ----------------------------------------------------------
            % Don't forget to echo change made here in the configuration
            % for the support-aware genie smoother, the naive GAMP method,
            % and the turbo-no-EM GAMP method!!!
            %-----------------------------------------------------------
            
            % Create a Signal class object for BG prior
            SignalObj = BernGauss('sparsity_rate', lambda, 'active_mean', ...
                Theta, 'active_var', Sigma2, 'learn_sparsity_rate', ...
                'scalar', 'learn_active_mean', 'scalar', 'learn_active_var', ...
                'scalar', 'init_params', 'true');
            D = 1;      % 1 mixture component
            
%             % Create a Signal class object for a BGM prior
%             D = 2;
%             SignalObj = GaussMix('sparsity_rate', cat(3, lambda/D, lambda/D), ...
%                 'active_mean', cat(3, Theta, -Theta), 'active_var', ...
%                 cat(3, Sigma2, Sigma2), 'learn_sparsity_rate', ...
%                 'scalar', 'learn_active_mean', 'scalar', ...
%                 'learn_active_var', 'scalar', 'init_params', 'false');
            
            % Create a SupportStruct class object for MMV model
            SupportObj = JointSparse();
            
            % Create an AmplitudeStruct class object for Gauss-Markov
            % process
            AmplitudeObj = GaussMarkov('Alpha', Alpha*ones(1,1,D), 'learn_alpha', ...
                'true', 'dim', 'row', 'init_params', 'true');
            
            % Create Noise class object for AWGN
            NoiseObj = GaussNoise('learn_prior_var', 'scalar', ...
                'init_params', 'true');
            
            % EMturboGAMP runtime options
            RunOptionsObj = RunOptions('smooth_iters', smooth_iters, ...
                'min_iters', min_iters, 'verbose', false, 'tol', 1e-6);
            
            % Place signal model objects into TurboOpt container
            TurboObj = TurboOpt('Signal', SignalObj, 'Noise', NoiseObj, ...
                'SupportStruct', SupportObj, 'AmplitudeStruct', ...
                AmplitudeObj, 'RunOptions', RunOptionsObj);
            
            % Create the signal, measurements, matrices, etc.
            [Y, A, X, S, NewNoiseObj] = signal_gen_fxn(GenParamObj, TurboObj);
            support = find(S(:,1) ~= 0);
            
            % Replace initial NoiseObj with NewNoiseObj, since it contains
            % the true noise variance
            NoiseObj = NewNoiseObj;
            TurboObj.Noise = NoiseObj;
            sig2e = NoiseObj.prior_var;
            
            
            % *************************************************************
            %       Solve using the support-aware genie smoother
            % *************************************************************            
            % Create the TurboOpt object for the genie smoother
            % Create a Signal class object for a BG prior
            D = 1;          % 1 mixture component
            GenieLambda = zeros(N,T,D);
            for i = 1:D
                [locR, locC] = find(S == i);
                GenieLambda(locR,locC,i) = 1;
            end
            GenieSignalObj = BernGauss('sparsity_rate', GenieLambda, ...
                'active_mean', Theta, 'active_var', Sigma2, ...
                'learn_sparsity_rate', 'false', 'learn_active_mean', ...
                'false', 'learn_active_var', 'false', 'init_params', 'false');   
            
%             % Create a Signal class object for a BGM prior
%             D = 2;      % Two active Gaussian components
%             GenieLambda = zeros(N,T,D);
%             for i = 1:D
%                 [locR, locC] = find(S == i);
%                 GenieLambda(locR,locC,i) = 1;
%             end
%             GenieSignalObj = GaussMix('sparsity_rate', GenieLambda, ...
%                 'active_mean', cat(3, Theta, -Theta), 'active_var', ...
%                 cat(3, Sigma2, Sigma2), 'learn_sparsity_rate', ...
%                 'false', 'learn_active_mean', 'false', ...
%                 'learn_active_var', 'false', 'init_params', 'false');
            
            % Create a SupportStruct class object for MMV model
            GenieSupportObj = JointSparse();
            
            % Create an AmplitudeStruct class object for Gauss-Markov
            % process
            GenieAmplitudeObj = GaussMarkov('Alpha', Alpha*ones(1,1,D), ...
                'learn_alpha', 'false', 'dim', 'row', 'init_params', 'false');
            
            % Create Noise class object for AWGN
            GenieNoiseObj = GaussNoise('learn_prior_var', 'false', ...
                'init_params', 'false', 'prior_var', sig2e);
            
            % Place signal model objects into TurboOpt container
            GenieTurboObj = TurboOpt('Signal', GenieSignalObj, 'Noise', ...
                GenieNoiseObj, 'SupportStruct', GenieSupportObj, ...
                'AmplitudeStruct', GenieAmplitudeObj, 'RunOptions', ...
                RunOptionsObj);
            
            GAMPOptions = GampOpt();
            GAMPOptions.adaptStep = false;
            GAMPOptions.nit = inner_iters;
            GAMPOptions.tol = 1e-5;

            tic;
            X_sks = EMturboGAMP(Y, A, GenieTurboObj, GAMPOptions);
            Run_sks(b,d,n) = toc;
            
            % Time-averaged normalized MSE (TNMSE)
            TNMSE_sks(b,d,n) = sum(sum(abs(X - X_sks).^2, 1)./...
                sum(abs(X).^2, 1))/T;
            
            % NSER
            NSER_sks(b,d,n) = nser(support, find(sum(abs(X_sks),2) > 1e-2));
            
            
            % *************************************************************
            %                   Solve using naive GAMP
            % *************************************************************  
            GAMPOptions = GampOpt();
            GAMPOptions.adaptStep = false;
            GAMPOptions.nit = inner_iters;
            GAMPOptions.tol = 1e-3;
            
            % Build an appropriate TurboOpt object for the naive GAMP
            % estimator
            NaiveSignalObj = BernGauss('sparsity_rate', lambda, 'active_mean', ...
                Theta, 'active_var', Sigma2, 'learn_sparsity_rate', ...
                'scalar', 'learn_active_mean', 'scalar', 'learn_active_var', ...
                'scalar', 'init_params', 'true');
%             NaiveSignalObj = GaussMix('sparsity_rate', cat(3, lambda/D, lambda/D), ...
%                 'active_mean', cat(3, Theta, -Theta), 'active_var', ...
%                 cat(3, Sigma2, Sigma2), 'learn_sparsity_rate', ...
%                 'scalar', 'learn_active_mean', 'scalar', ...
%                 'learn_active_var', 'scalar', 'init_params', 'false');
                        
            % Create Noise class object for AWGN
            NaiveNoiseObj = GaussNoise('learn_prior_var', 'scalar', ...
                'init_params', 'false', 'prior_var', sig2e);
            
            NaiveTurboObj = TurboOpt('Signal', NaiveSignalObj, 'Noise', ...
                NaiveNoiseObj, 'SupportStruct', NoSupportStruct(), ...
                'AmplitudeStruct', NoAmplitudeStruct(), 'RunOptions', ...
                RunOptionsObj);

            tic;
            X_naive = EMturboGAMP(Y, A, NaiveTurboObj, GAMPOptions);
            Run_naive(b,d,n) = toc;
            
            % Time-averaged normalized MSE (TNMSE)
            TNMSE_naive(b,d,n) = sum(sum(abs(X - X_naive).^2, 1)./...
                sum(abs(X).^2, 1))/T;
            
            % NSER
            NSER_naive(b,d,n) = nser(support, find(sum(abs(X_naive),2) > 1e-2));
            
            
            % *************************************************************
            %       Solve using turboGAMP (no EM parameter learning)
            % *************************************************************            
            % Create the TurboOpt object for turboGAMP
            % Create a Signal class object for a BG prior
            NoEMSignalObj = BernGauss('sparsity_rate', lambda, 'active_mean', ...
                        Theta, 'active_var', Sigma2, 'learn_sparsity_rate', ...
                        'false', 'learn_active_mean', 'false', 'learn_active_var', ...
                        'false', 'init_params', 'false');
            D = 1;      % 1 mixture component
            
%             % Create a Signal class object for a BGM prior
%             D = 2;      % Two active Gaussian components
%             NoEMSignalObj = GaussMix('sparsity_rate', cat(3, lambda/D, lambda/D), ...
%                 'active_mean', cat(3, Theta, -Theta), 'active_var', ...
%                 cat(3, Sigma2, Sigma2), 'learn_sparsity_rate', ...
%                 'false', 'learn_active_mean', 'false', ...
%                 'learn_active_var', 'false', 'init_params', 'false');
            
            % Create a SupportStruct class object for MMV model
            NoEMSupportObj = JointSparse();
            
            % Create an AmplitudeStruct class object for Gauss-Markov
            % process
            NoEMAmplitudeObj = GaussMarkov('Alpha', Alpha*ones(1,1,D), ...
                'learn_alpha', 'false', 'dim', 'row', 'init_params', 'false');
            
            % Create Noise class object for AWGN
            NoEMNoiseObj = GaussNoise('learn_prior_var', 'scalar', ...
                'init_params', 'false', 'prior_var', sig2e);
            
            % Place signal model objects into TurboOpt container
            NoEMTurboObj = TurboOpt('Signal', NoEMSignalObj, 'Noise', ...
                NoEMNoiseObj, 'SupportStruct', NoEMSupportObj, ...
                'AmplitudeStruct', NoEMAmplitudeObj, 'RunOptions', ...
                RunOptionsObj);
            
            GAMPOptions = GampOpt();
            GAMPOptions.adaptStep = false;
            GAMPOptions.nit = inner_iters;
            GAMPOptions.tol = 1e-3;

            tic;
            X_noem = EMturboGAMP(Y, A, NoEMTurboObj, GAMPOptions);
            Run_noem(b,d,n) = toc;
            
            % Time-averaged normalized MSE (TNMSE)
            TNMSE_noem(b,d,n) = sum(sum(abs(X - X_noem).^2, 1)./...
                sum(abs(X).^2, 1))/T;
            
            % NSER
            NSER_noem(b,d,n) = nser(support, find(sum(abs(X_noem),2) > 1e-2));
            
            
            % *************************************************************
            %                   Solve using EMturboGAMP
            % *************************************************************  
            GAMPOptions = GampOpt();
            GAMPOptions.adaptStep = false;
            GAMPOptions.nit = inner_iters;
            GAMPOptions.tol = 1e-3;

            tic;
            X_turbo = EMturboGAMP(Y, A, TurboObj, GAMPOptions);
            Run_turbo(b,d,n) = toc;
            
            % Time-averaged normalized MSE (TNMSE)
            TNMSE_turbo(b,d,n) = sum(sum(abs(X - X_turbo).^2, 1)./...
                sum(abs(X).^2, 1))/T;
            
            % NSER
            NSER_turbo(b,d,n) = nser(support, find(sum(abs(X_turbo),2) > 1e-2));
            
            if fid ~= 1, fid = fopen([savefile(1:end-3) 'txt'], 'w'); end
            fprintf(fid, 'Progress: %g\n', (N_beta*(b-1) + d) / N_beta / N_delta);
            if fid ~= 1, fclose(fid); end
        end     % delta (M/N)
    end         % beta (E[K]/M)
    
    % Having completed one sweep of the (beta, delta) grid, estimate the
    % time remaining and save the temporary data
    est_time = (N_trials - n)*(etime(clock, time)/n)/60/60;
    disp(sprintf('(%d/%d) Estimated time remaining: %3.2f hours', ...
        n, N_trials, est_time));
    clear Y y_cell X S A A_cell X_sks X_naive X_turbo
    save(savefile);
end

disp(sprintf('Total time elapsed for test: %3.2f hours', ...
    etime(clock, time)/60/60));

% % Close the matlabpool
% matlabpool close

% Final save of data
clear Y y_cell X S A A_cell X_sks X_naive X_turbo
save(savefile);