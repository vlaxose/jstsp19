% TestEMturboGAMP
%
% Test the recovery performance of the EMturboGAMP algorithm on a single
% problem with synthetic data generated according to a desired
% distribution.
%
%                       Y = A*X + E
%
% X has dimension N-by-T
% A has dimension M-by-N
% Y has dimension M-by-T
%
% Prior distribution of X is specified through the Signal, SupportStruct,
% and AmplitudeStruct class objects (see ClassDefs folder).
%
% Prior distribution of E is specified through the Observation class object
% (see ClassDefs folder).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 10/24/12
% Change summary: 
%       - Created (01/05/12; JAZ)
%       - Changed Noise class references to Observation class references
%         (05/23/12; JAZ)
%       - Added Elastic Net and Laplacian as signal priors (10/16/12; JAZ)
%       - Added Bernoulli-Laplacian signal prior (10/24/12; JAZ)
% Version 0.2
%

% clear; clc
% % NewStream = RandStream.create('mt19937ar', 'Seed', 5489+1);
% % RandStream.setDefaultStream(NewStream);
Stream = RandStream.getGlobalStream;
SavedState = Stream.State;
Stream.State = SavedState;

%% Start by building a signal/measurement matrix realization

% Dimensions and other basic setup parameters
N = 2048;               % Row dimension of signal matrix, X
T = 1;                	% Column dimension of signal matrix, X
NoverM = 3;           	% Unknowns-to-measurements ratio (N/M)
MoverK = 5;             % Measurements-to-sparsity ratio (M/K)
SNRdB = 25;             % SNR (in dB)
data = 'real';          % Real- or complex-valued signal/noise?
Atype = 'Gauss';      	% Type of random measurement matrix (IID Gaussian:
                        % 'Gauss', Rademacher: 'Rad', Subsampled DFT:
                        % 'DFT')
gamp_ver = 'mmse';    	% Run sum-product ('mmse') or max-sum ('map') GAMP?
CommonA = true;      	% Use a common A matrix for all timesteps?
override = true;        % Give EMturboGAMP the true noise variance?
smooth_iters = 15;      % Max # of turbo iterations
min_iters = 5;        	% Min # of turbo iterations
GAMP_iters = 125;        % Max # of GAMP iterations
adaptStep = true;      % Use GAMP's adaptive step sizing rule?
verbose = true;         % Work verbosely


% ************************************************************************
%      Uncomment the following to choose different priors on X and E
% ************************************************************************


% ************************************************************************
%                       Marginal Signal Prior
% ************************************************************************
% BERNOULLI-GAUSSIAN
SigPrior = 'BG';                % Identifier (don't change)
Mean = 0;                       % Active mean
Var = 1;                        % Active variance
learn_sparsity_rate = 'false'; % ('false', 'scalar', 'row', or 'column')
learn_active_mean = 'false';   % ('false', 'scalar', 'row', or 'column')
learn_active_var = 'false';    % ('false', 'scalar', 'row', or 'column')
init_params = 'false';          % Initialize model parameters from data?

% % BERNOULLI/MIXTURE-OF-GAUSSIANS
% SigPrior = 'MoG';   % Identifier (don't change)
% Means = [-1, 1];    % Active means (one for each mixture component)
% Vars = 0.25*ones(1,2);   % Active variances (one for each mixture component)
% % Means = [0, 0];    % Active means (one for each mixture component)
% % Vars = [1, 10];   % Active variances (one for each mixture component)
% % (Note: Mixture components will share equal weights by default)
% learn_sparsity_rate = 'scalar'; % ('false', 'scalar', 'row', or 'column')
% learn_active_mean = 'scalar';   % ('false', 'scalar', 'row', or 'column')
% learn_active_var = 'scalar';    % ('false', 'scalar', 'row', or 'column')
% init_params = 'false';          % Initialize model parameters from data?

% % BERNOULLI-LAPLACIAN
% SigPrior = 'BL';                % Identifer (don't change)
% Var = 1;                        % Active variance
% learn_sparsity_rate = 'false';  % ('false', 'scalar', 'row', or 'column')
% learn_laplace_rate = 'false';  % ('false', 'scalar', 'row', or 'column')

% % LAPLACIAN
% SigPrior = 'Lap';       % Identifier (don't change)
% LapLambda = 1e1;      	% Laplacian variance = 2*LapLambda^2
% learn_lambda = true;	% Learn Laplacian rate parameter with EM?

% % ELASTIC NET
% SigPrior = 'Elastic';   % Identifier (don't change)
% Ell1Penalty = 1e0;      % Ell-1 penalty parameter
% Ell2Penalty = 1e-2;      % Ell-2 penalty parameter
% learn_lambda1 = true;   % Learn Ell-1 parameter with EM?
% learn_lambda2 = true;   % Learn ELl-2 parameter with EM?

% ************************************************************************
%                     Support Structure Prior
% ************************************************************************
% NONE
SuppPrior = 'None';     % Identifier (don't change)

% % JOINT SPARSITY (MMV w/ row sparsity)
% SuppPrior = 'JS';       % Identifier (don't change)

% % BINARY MARKOV CHAIN
% SuppPrior = 'MC1';      % Identifier (don't change)
% p01 = 0.05;             % Active-to-inactive transition probability
% learn_p01 = 'false';    % Learn p01 using EM algorithm ('true' or 'false')
% dimS = 'row';        	  % Each row forms a Markov chain ('col' for column)

% % (D+1)-ARY MARKOV CHAIN
% SuppPrior = 'MCD';          % Identifier (don't change)
% p0d = cat(3, 0.05, 0.05);	% Default active-to-inactive transition probabilities (D = 2)
% learn_p0d = 'false';       	% Learn p0d using EM alg. by default
% dimS = 'row';                % Each row forms a Markov chain

% % 2D MARKOV RANDOM FIELD
% % (Note that this class ignores any support-related properties of the
% % Signal class, e.g., a sparsity rate)
% SuppPrior = '2DMRF';          % Identifier (don't change)
% betaH = 0.9;                % Horizontal inverse temperature
% betaV = 0.9;                % Vertical inverse temperature
% MRFalpha = .05;               % Sparsity parameter
% MRFlearn_beta = 'false';     % Learn common beta using pseudo-ML?
% MRFlearn_alpha = 'false';    % Learn alpha using pseudo-ML?
% MRFmaxIter = 5;             % Max # of loopy BP iters per turbo iter

% % 3D MARKOV RANDOM FIELD (from an N-by-1 signal X, where N is a power of 3)
% % (Note that this class ignores any support-related properties of the
% % Signal class, e.g., a sparsity rate)
% SuppPrior = '3DMRF';     	% Identifier (don't change)
% betax = 5.95;                % x-axis inverse temperature
% betay = 5.95;                % y-axis inverse temperature
% betaz = 5.95;                % z-axis inverse temperature
% MRFalpha = 0.48;           	% Sparsity parameter
% MRFlearn_beta = 'true';  	% Learn common beta using pseudo-ML?
% MRFlearn_alpha = 'true';  	% Learn alpha using pseudo-ML?
% MRFmaxIter = 50;             % Max # of loopy BP iters per turbo iter

% ************************************************************************
%                  	  Amplitude Structure Prior
% ************************************************************************
% NONE
AmpPrior = 'None';      % Identifier (don't change)

% % GAUSS-MARKOV PROCESS
% AmpPrior = 'GM';        % Identifier (don't change)
% alpha = 0.10;           % Process correlation = 1 - alpha
% learn_alpha = 'true';	% Learn Alpha using EM alg. ('true' or 'false')
% dim = 'row';            % Each row forms a Gauss-Markov process ('col' for column)
% init_aparams = 'false';  % Initialize alpha from the data?

% ************************************************************************
%                       Observation Prior
% ************************************************************************
% AWGN
ObservationPrior = 'AWGN';     	% Identifier (don't change)
learn_prior_var = 'scalar';  	% ('false', 'scalar', 'row', or 'column')
init_nparams = 'false';      	% Initialize noise variance from data?

% % Binary Gaussian mixture additive noise
% ObservationPrior = 'GMN';       % Identifier (don't change)
% PI = 0.10;                      % Prob. of drawing from large variance
% NUratio = 1000;                   % Ratio of big-to-small noise variances
% learn_pi = 'false';             % ('false', 'scalar', 'row', or 'column')
% learn_nu0 = 'false';            % ('false', 'scalar', 'row', or 'column')
% learn_nuratio = 'false';        % ('false', 'scalar', 'row', or 'column')

% ************************************************************************
%               Don't change stuff below this point
% ************************************************************************


%% Use the user's specifications above to build the TurboOpt object

% Start with the signal prior
switch SigPrior
    case 'BG'
        sparsity_rate = 1 / MoverK / NoverM;
        SigObj = BernGauss('sparsity_rate', sparsity_rate, 'active_mean', ...
            Mean, 'active_var', Var, 'learn_sparsity_rate', ...
            learn_sparsity_rate, 'learn_active_mean', learn_active_mean, ...
            'learn_active_var', learn_active_var, 'init_params', ...
            init_params, 'data', data);
    case 'MoG'
        D = numel(Means);       % Number of mixture components
        sparsity_rate = NaN(1,1,D);
        active_mean = NaN(1,1,D);
        active_var = NaN(1,1,D);
        for d = 1:D
            sparsity_rate(1,1,d) = 1 / MoverK / NoverM / D;
            active_mean(1,1,d) = Means(d);
            active_var(1,1,d) = Vars(d);
        end
        SigObj = GaussMix('sparsity_rate', sparsity_rate, 'active_mean', ...
            active_mean, 'active_var', active_var, 'learn_sparsity_rate', ...
            learn_sparsity_rate, 'learn_active_mean', learn_active_mean, ...
            'learn_active_var', learn_active_var, 'init_params', ...
            init_params, 'data', data);
        if strcmp(SuppPrior, 'MC1')
            error('Cannot combine 1st-order Markov chain with Gaussian mixture prior')
        end
    case 'BL'
        sparsity_rate = 1 / MoverK / NoverM;
        laplace_rate = sqrt(2 / Var);   % Set Laplacian rate based on desired var
        SigObj = BernLaplace('sparsity_rate', sparsity_rate, ...
            'learn_sparsity_rate', learn_sparsity_rate, 'laplace_rate', ...
            laplace_rate, 'learn_laplace_rate', learn_laplace_rate, 'data', ...
            data);
    case 'Lap'
        SigObj = Laplacian('lambda', LapLambda, 'learn_lambda', ...
            learn_lambda, 'version', gamp_ver);
    case 'Elastic'
        SigObj = ElasticNet(Ell1Penalty, Ell2Penalty, learn_lambda1, ...
            learn_lambda2, gamp_ver);
    otherwise
        error('Unrecognized marginal signal prior')
end

% Now the support prior
switch SuppPrior
    case 'None'
        SuppObj = NoSupportStruct();
    case 'JS'
        SuppObj = JointSparse();
    case 'MC1'
        SuppObj = MarkovChain1('p01', p01, 'learn_p01', learn_p01, ...
            'dim', dimS);
    case 'MCD'
        SuppObj = MarkovChainD('p0d', p0d, 'learn_p0d', learn_p0d, ...
            'dim', dimS);
    case '2DMRF'
        SuppObj = MarkovField('betaH', betaH, 'betaV', betaV', 'alpha', ...
            MRFalpha, 'learn_beta', MRFlearn_beta, 'learn_alpha', ...
            MRFlearn_alpha, 'maxIter', MRFmaxIter);
    case '3DMRF'
        % Check to make sure that user has specified X to be an N-by-1
        % dimensional signal, where N is a power of 3.  If so, specify
        % integer (x,y,z) coordinates for each entry of X such that they
        % collectively define a 3D cube lattice structure
        if T ~= 1 || round(N^(1/3))^3 ~= N
            error(['Ensure that X is N-by-1 dimensional, where N is ' ...
                'a power of 3, in order to use 3D MRF'])
        else
            Ncube = round(N^(1/3));
            [coordx, coordy, coordz] = ind2sub([Ncube, Ncube, Ncube], ...
                [1:N]');
            MRFcoordinates = [coordx, coordy, coordz];
        end
        SuppObj = MarkovField3D('betax', betax, 'betay', betay, ...
            'betaz', betaz, 'alpha', MRFalpha, 'learn_beta', ...
            MRFlearn_beta, 'learn_alpha', MRFlearn_alpha, 'maxIter', ...
            MRFmaxIter, 'coordinates', MRFcoordinates);
    otherwise
        error('Unrecognized support structure prior')
end

% Now the amplitude prior
switch AmpPrior
    case 'None'
        AmpObj = NoAmplitudeStruct();
    case 'GM'
        % Check to see if signal prior is a mixture-of-Gaussians, in which
        % case alpha must be vectorized (if it isn't already)
        if isa(SigObj, 'GaussMix')
            if numel(alpha) == D
                alpha = reshape(alpha, 1, 1, D);
            elseif numel(alpha) == 1
                % Duplicate alpha
                alpha = repmat(alpha, [1, 1, D]);
            else
                error('Check number of elements in alpha')
            end
        end
        AmpObj = GaussMarkov('Alpha', alpha, 'learn_alpha', learn_alpha, ...
            'dim', dim, 'init_params', init_aparams);
    otherwise
        error('Unrecognized ampltitude structure prior')
end

% Now the observation prior
switch ObservationPrior
    case 'AWGN'
        ObservationObj = GaussNoise('learn_prior_var', learn_prior_var, ...
            'init_params', init_nparams, 'data', data, 'version', ...
            gamp_ver);
    case 'GMN'
        ObservationObj = GaussMixNoise('PI', PI, 'NUratio', NUratio, 'learn_pi', ...
            learn_pi, 'learn_nu0', learn_nu0, 'learn_nuratio', ...
            learn_nuratio, 'data', data);
    otherwise
        error('Unrecognized observation prior')
end

% Finally the turbo runtime options
RunOpt = RunOptions('smooth_iters', smooth_iters, 'min_iters', min_iters, ...
    'tol', -1, 'verbose', verbose, 'warm_start', false);

% Now build the TurboOpt object
TBobj = TurboOpt('Signal', SigObj, 'Observation', ObservationObj, ...
    'SupportStruct', SuppObj, 'AmplitudeStruct', AmpObj, 'RunOptions', ...
    RunOpt);

% Print the configuration to the command window
fprintf('\n******************************\n')
fprintf('TurboOpt object configuration:\n')
fprintf('******************************\n')
TBobj.print();

% Build the GenParams object
switch lower(Atype)
    case 'gauss'
        Atype = 1;
    case 'rad'
        Atype = 2;
    case 'dft'
        Atype = 3;
    otherwise
        error('Unrecognized random A matrix type')
end
GPobj = GenParams(N, T, ceil(N / NoverM), SNRdB, Atype, data, CommonA);

% Print the configuration to the command window
fprintf('\n*******************************\n')
fprintf('GenParams object configuration:\n')
fprintf('*******************************\n')
GPobj.print();


%% Now construct a synthetic realization of Y, A, and X

% warning('Replacing true signal prior with iid B-G prior!')
% sparsity_rate = 1 / MoverK / NoverM;
% SigObj2 = BernGauss('sparsity_rate', sparsity_rate, 'active_mean', 0, 'active_var', 1, 'data', data);
% TBobj.Signal = SigObj2;

[Y, A, X, S, Noise] = signal_gen_fxn(GPobj, TBobj);

% TBobj.Signal = SigObj;

switch override
    case true
        fprintf('Replacing Noise.prior_var (%1.4f) with the true noise variance.\n', ...
            Noise.prior_var);
        ObservationObj = Noise;
        TBobj.Observation = ObservationObj;
    case false
        fprintf('Using user-specified noise variance initialization.\n')
    otherwise
        error('Unrecognized option: override')
end

% Make a copy of the TBobj object for AMP-MMV
TBcopy = TBobj.copy();

% Plot the signal realization
switch data
    case 'real'
        if T > 1
            figure(1); clf
            imagesc(X); colorbar
            title('True Signal, X');
            xlabel('Timestep [t]'); 
            ylabel('Coeffient Index [n]')
        else
            figure(1); clf
            if ~strcmpi(SuppPrior, '3DMRF')
                stem(X, 'b');
            else
                close(1)
                Nc = round(N^(1/3));
                sliceomatic(reshape(X, Nc, Nc, Nc));
            end
        end
    case 'complex'
        if T > 1
            figure(1); clf
            imagesc(abs(X)); colorbar
            title('|X|');
            xlabel('Timestep [t]'); 
            ylabel('Coeffient Index [n]')
        else
            figure(1); clf; subplot(211)
            stem(real(X), 'b');
            subplot(212)
            stem(imag(X), 'b');
        end
end
if ~strcmpi(SuppPrior, '3DMRF'), pause(0.1);
else pause; end


%% Recover X from Y using EMturboGAMP

% Specify runtime options for GAMP
Options = GampOpt('adaptStep', adaptStep, 'nit', GAMP_iters, ...
    'varNorm', false, 'verbose', false, 'stepMax', 1e-1);

% Execute EMturboGAMP
tic
Xhat = EMturboGAMP(Y, A, TBobj, Options);
toc


%% Compute performance metrics

TNMSE = (1/T)*sum(sum(abs(Xhat - X).^2, 1) ./ sum(abs(X).^2, 1), 2);
fprintf('TNMSE: %g dB\n', 10*log10(TNMSE))

% Plot the recovered signal
switch data
    case 'real'
        if T > 1
            figure(2); clf
            imagesc(Xhat); colorbar
            title(sprintf(['Recovered Signal, X_{hat} | N = %d, N/M = %g, ' ...
                'M/K = %g, SNR = %g dB'], N, NoverM, MoverK, SNRdB));
            xlabel(sprintf('Timestep [t]  |  TNMSE = %g dB', 10*log10(TNMSE))); 
            ylabel('Coeffient Index [n]')
        else
            if ~strcmpi(SuppPrior, '3DMRF')
                figure(1);
                hold on; stem(Xhat, 'r'); hold off;
                legend('X', 'X_{hat}', 'Location', 'Best')
                xlabel(sprintf('Index [n]  |  TNMSE = %g dB', 10*log10(TNMSE))); 
                title(sprintf(['N = %d, N/M = %g, M/K = %g, SNR = %g dB'], ...
                    N, NoverM, MoverK, SNRdB));
            else
                figure(2); close(2);
                sliceomatic(reshape(Xhat, Nc, Nc, Nc));
            end
        end
    case 'complex'
        if T > 1
            figure(2); clf
            imagesc(abs(Xhat)); colorbar
            title(sprintf(['Recovered Signal, |X_{hat}| - N = %d, N/M = %g, ' ...
                'M/K = %g, SNR = %g dB'], N, NoverM, MoverK, SNRdB));
        else
            figure(1); subplot(211)
            hold on; stem(real(Xhat), 'r'); hold off;
            title(sprintf(['N = %d, N/M = %g, M/K = %g, SNR = %g dB'], ...
                N, NoverM, MoverK, SNRdB));
            legend('Real(X)', 'Real(X_{hat})', 'Location', 'Best')
            subplot(212)
            hold on; stem(imag(Xhat), 'r'); hold off
            legend('Imag(X)', 'Imag(X_{hat})', 'Location', 'Best')
            xlabel(sprintf('Index [n]  |  TNMSE = %g dB', 10*log10(TNMSE))); 
        end
end