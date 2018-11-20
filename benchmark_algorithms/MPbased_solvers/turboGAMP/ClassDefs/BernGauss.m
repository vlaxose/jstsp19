% CLASS: BernGauss
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Signal
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The BernGauss class contains the parameters needed to define a
%   Bernoulli-Gaussian marginal prior distribution for each signal
%   coefficient.  Specifically, each signal coefficient, X(n,t), has a
%   marginal prior distribution
%   pdf(X(n,t)) = (1 - LAMBDA(n,t))*delta(X(n,t)) + 
%                 LAMBDA(n,t)*Normal(X(n,t); THETA(n,t), PHI(n,t))
%   where n = 1,...,N indexes the row of the signal matrix X, 
%   t = 1,...,T indexes the column (timestep) of the signal matrix X,
%   delta(.) is the Dirac delta function, and Normal(.; theta, phi) is
%   either a real- or complex-valued normal distribution with mean theta
%   and variance phi.  Note that this class can be used in the single
%   measurement vector (SMV) problem as well, (i.e., T = 1).
%
%   To create a BernGauss object, there are two constructors to choose from
%   (see METHODS section below).  The default constructor, BernGauss(), 
%   will create a BernGauss object initialized with all default values for 
%   each parameter/property.  The alternative constructor allows the user 
%   to initialize any subset of the parameters, with the remaining 
%   parameters initialized to their default values, by using MATLAB's 
%   property/value string pairs convention, e.g.,
%   BernGauss('sparsity_rate', 0.05, 'learn_sparsity_rate', 'false') will
%   construct a BernGauss object in which LAMBDA(n,t) = 0.05 for all n,t,
%   and this sparsity rate will not be refined by the expectation-
%   maximization (EM) parameter learning procedure.  Remaining parameters 
%   will be set to their default values.  Note that the parameters LAMBDA,
%   THETA, and PHI can be initialized as either scalars, length-N row 
%   vectors, length-T column vectors, or N-by-T matrices of values, to 
%   allow for distributions that are i.i.d., temporally uniform, spatially 
%   uniform, or completely independent, respectively.  Note also that if
%   the user wishes to have EMTurboGAMP attempt to intelligently initialize
%   LAMBDA, THETA, and PHI based on the matrix of measurements, Y, and the
%   sensing matrix A, then BernGauss('init_params', 'true') is the
%   appropriate constructor.  By default, the signal is assumed to be
%   real-valued.  If it is complex, then set data = 'complex'.
%
%   Additionally, this class contains information about which
%   parameters should be learned from the data, and in what manner, (see
%   PROPERTIES below).  As an example, to prevent the parameter
%   sparsity_rate (LAMBDA) from being learned by an EM algorithm, set
%   learn_sparsity_rate = 'false'.  To learn a single, common sparsity rate
%   for all elements of the signal matrix, X, set learn_sparsity_rate =
%   'scalar'.  Likewise, to learn a unique sparsity rate for each row
%   (column) of X, set learn_sparsity_rate = 'row' ('column').
%
% PROPERTIES (State variables)
%   sparsity_rate           The prior sparsity rate(s), LAMBDA  [Default:
%                           0.05]
%   learn_sparsity_rate     Learn sparsity rate using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'scalar']
%   active_mean             The real or complex prior mean(s), THETA
%                           [Default: 0]
%   learn_active_mean       Learn prior mean using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'scalar']
%   active_var              The prior active-coefficient variance(s), PHI
%                           [Default: 1]
%   learn_active_var        Learn prior variance using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'scalar']
%   init_params             Initialize LAMBDA, THETA, and PHI automatically
%                           using Y and A ('true'), or not ('false').
%                           [Default: 'false']
%   data                    Real-valued signal ('real') or complex-valued
%                           ('complex')? [Default: 'real']
%
% METHODS (Subroutines/functions)
%   BernGauss()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   BernGauss('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class BernGauss, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'BG' when obj is a BernGauss object
%   BernGaussCopyObj = copy(obj)
%       - Create an independent copy of the BernGauss object, obj.
%   [EstimIn, S_POST] = UpdatePriors(TBobj, GAMPState, EstimInOld)
%       - Method that will take the final message state from the most
%         recent GAMP iteration and use it to generate a new object of the
%         EstimIn base class, which will be used to set the signal "prior"
%         for the next iteration of GAMP. TBobj is an object of the
%         TurboOpt class, GAMPState is an object of the GAMPState class,
%         and EstimInOld is the previous EstimIn object given to GAMP.  If 
%         TBobj.commonA is false, then this method returns a 1-by-T cell
%         array of EstimIn objects. Also returns estimated posteriors for
%         the support [Hidden method]
%   EstimIn = InitPriors(TBobj, Y, A)
%      	- Provides an initial EstimIn object for use by GAMP the first
%         time. If the user wishes to initialize parameters from the
%         data, Y, and sensing matrix A, then arguments Y and A must be
%         provided.  TBobj is a TurboOpt object.  If TBobj.commonA is 
%         false, then this method returns a 1-by-T cell array of EstimIn 
%         objects. [Hidden method]
%   [THETA_upd, PHI_upd] = obj.LearnAmplitudeParams(obj, TBobj)
%       - In the absence of any form of amplitude structure, this method
%         will perform EM parameter learning of the amplitude-related model
%         prior parameters [Hidden method]
%   [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(TBobj, GenParams)
%       - Generate a realization of the signal (X_TRUE), the underlying
%         binary support matrix (S_TRUE), and amplitude matrix,
%         (GAMMA_TRUE), using a TurboOpt object (TBobj) and a GenParams
%         object (GenParams) [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'active_mean'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Gaussian mean: '), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (10/11/11; JAZ)
%       - Added implementation of UpdatePriors, which is called by the main
%         EMturboGAMP function to produce new EstimIn object (12/13/11;
%         JAZ)
%       - Added genRand method (01/02/12; JAZ)
%       - Added support for time-varying transform matrix A(t) (02/07/12;
%         JAZ)
%       - Added ability to return support posteriors to UpdatePriors
%         method (04/12/12; JAZ)
%       - Added support for complex-valued signals (05/22/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added version property (11/01/12; JAZ)
%       - Added EMreport method implementation (01/28/13; JAZ)
% Version 0.2
%

classdef BernGauss < Signal

    properties
        % Bernoulli-Gaussian prior family properties
        sparsity_rate = 0.05;
        learn_sparsity_rate = 'scalar';
        active_mean = 0;
        learn_active_mean = 'scalar';
        active_var = 1;
        learn_active_var = 'scalar';
        init_params = 'false';
        data = 'real';
        version = 'mmse';
    end % properties
       
    properties (Constant, Hidden)
        type = 'BG';        % Bernoulli-Gaussian type identifier
        eps = 1e-6;         % Small positive constant for Taylor approx
        tau = -1;           % Set tau = -1 to use Taylor approx, or set to 
                            % a number slightly less than 1, e.g., 0.999,
                            % in order to use a threshold on incoming
                            % probabilities (to GAMP), PI_IN, to choose
                            % between informative and non-informative
                            % Gaussian messages
    end
    
    properties (Hidden)
        EMcnt = 0;          % Counter of # of EM iterations
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = BernGauss(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(lower(varargin{i}), varargin{i+1});
                end
            end
        end                    
        
        
        % *****************************************************************
        %                         SET METHODS
        % *****************************************************************
        
        % Set method for sparsity_rate (LAMBDA)
        function obj = set.sparsity_rate(obj, LAMBDA)
           if min(min(LAMBDA)) < 0 || max(max(LAMBDA)) > 1
              error('sparsity_rate must be in the interval [0,1]')
           else
              obj.sparsity_rate = LAMBDA;
           end
        end
        
        % Set method for active_mean (THETA)
        function obj = set.active_mean(obj, THETA)
              obj.active_mean = THETA;
        end
        
        % Set method for active_var (PHI)
        function obj = set.active_var(obj, PHI)
           if min(min(PHI)) < 0
              error('active_var must be non-negative')
           else
              obj.active_var = PHI;
           end
        end
        
        % Set method for learn_sparsity_rate
        function obj = set.learn_sparsity_rate(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_sparsity_rate')
            end
            obj.learn_sparsity_rate = lower(string);
        end
        
        % Set method for learn_active_mean
        function obj = set.learn_active_mean(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_active_mean')
            end
            obj.learn_active_mean = lower(string);
        end
        
        % Set method for learn_active_var
        function obj = set.learn_active_var(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_active_var')
            end
            obj.learn_active_var = lower(string);
        end
        
        % Set method for init_params
        function obj = set.init_params(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: init_params')
            end
            obj.init_params = lower(string);
        end
        
        % Set method for data
        function obj = set.data(obj, string)
            if ~any(strcmpi(string, {'real', 'complex'}))
                error('Invalid option: data')
            end
            obj.data = lower(string);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('BernGauss does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SIGNAL PRIOR: Bernoulli-Gaussian\n')
            fprintf('      sparsity_rate: %s\n', ...
                form(obj, obj.sparsity_rate))
            fprintf('        active_mean: %s\n', ...
                form(obj, obj.active_mean))
            fprintf('         active_var: %s\n', ...
                form(obj, obj.active_var))
            fprintf('learn_sparsity_rate: %s\n', obj.learn_sparsity_rate)
            fprintf('  learn_active_mean: %s\n', obj.learn_active_mean)
            fprintf('   learn_active_var: %s\n', obj.learn_active_var)
            fprintf('        init_params: %s\n', obj.init_params)
            fprintf('               data: %s\n', obj.data)
            fprintf('       GAMP version: %s\n', obj.version)
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Create an indepedent copy of a BernGauss object
        function BernGaussCopyObj = copy(obj)
            BernGaussCopyObj = BernGauss('sparsity_rate', ...
                obj.sparsity_rate, 'learn_sparsity_rate', ...
                obj.learn_sparsity_rate, 'active_mean', obj.active_mean, ...
                'learn_active_mean', obj.learn_active_mean, 'active_var', ...
                obj.active_var, 'learn_active_var', obj.learn_active_var, ...
                'init_params', obj.init_params, 'data', obj.data);
        end
        
        
        % *****************************************************************
        %                      ACCESSORY METHODs
        % *****************************************************************
        
        % This function allows one to query which type of signal family
        % this class is by returning a character string type identifier
        function type_id = get_type(obj)
            type_id = obj.type;
        end            
    end % methods
    
    methods (Hidden)
        % *****************************************************************
        %               UPDATE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        
     	% Update the EstimIn object for a Bernoulli-Gaussian prior on X
        function [EstimIn, S_POST] = UpdatePriors(obj, TBobj, ...
                GAMPState, EstimInOld)
            
            % Unpack the GAMPState object
            [XHAT, ~, RHAT, RVAR] = GAMPState.getState();
            N = size(RHAT, 1);
            T = size(RHAT, 2);
            RVAR(RVAR < 1e-14) = 1e-14;
            
            % Unpack inputs given to GAMP at the last iteration
            switch TBobj.commonA
                case true
                    PI_IN_OLD = EstimInOld.p1;
                    ETA_IN_OLD = EstimInOld.estim1.mean0;
                    KAPPA_IN_OLD = EstimInOld.estim1.var0;
                case false
                    PI_IN_OLD = NaN(N,T);  ETA_IN_OLD = NaN(N,T);
                    KAPPA_IN_OLD = NaN(N,T);
                    for t = 1:T
                        PI_IN_OLD(:,t) = EstimInOld{t}.p1;
                        ETA_IN_OLD(:,t) = EstimInOld{t}.estim1.mean0;
                        KAPPA_IN_OLD(:,t) = EstimInOld{t}.estim1.var0;
                    end
            end
            
            % Use outputs from GAMP to compute the messages moving
            % from the GAMP factor nodes to the support nodes, S
            TempVar = RVAR + KAPPA_IN_OLD;
            switch obj.data
                case 'real'
                    PI_OUT = sqrt(TempVar ./ RVAR) .* exp(-(1/2) * ...
                        ( (RHAT.^2 ./ RVAR) - ((RHAT - ETA_IN_OLD).^2 ./ TempVar)));
                case 'complex'
                    PI_OUT = TempVar ./ RVAR .* exp(-( (abs(RHAT).^2 ./ RVAR) - ...
                        (abs(RHAT - ETA_IN_OLD).^2 ./ TempVar)));
            end
            PI_OUT = 1 ./ (1 + PI_OUT);
            
                    
            % Now, let's determine the values of the incoming (to GAMP)
            % activity probabilities, PI_IN.  We need to look at the type
            % of structure present in the signal support matrix, S
            switch TBobj.SupportStruct.get_type()
                case 'None'
                    % No structured support, thus return the priors as the
                    % updated messages for GAMP
                    PI_IN = obj.sparsity_rate;
            
                    % Compute posteriors, Pr{s(n,t) | Y}
                    S_POST = (PI_IN .* PI_OUT) ./ (PI_IN .* PI_OUT + ...
                        (1 - PI_IN) .* (1 - PI_OUT));
                    
                    % If the user has requested EM refinement of LAMBDA, do
                    % that now.
                    switch TBobj.Signal.learn_sparsity_rate
                        case 'scalar'
                            % Update a single scalar
                            lambda_upd = sum(sum(S_POST)) / N / T;
                        case 'row'
                            % Update different lambda for each row
                            lambda_upd = sum(S_POST, 2) / T;
                        case 'column'
                            % Update different lambda for each column
                            lambda_upd = sum(S_POST, 1) / N;
                        case 'false'
                            % Do not update the prior
                            lambda_upd = obj.sparsity_rate;
                    end
                    obj.sparsity_rate = TBobj.resize(lambda_upd, N, T);
                    
                otherwise
                    % Call the UpdateSupport method for the appropriate
                    % model of support structure.  EM learning of sparsity
                    % rate will be handled by the UpdateSupport method
                    [PI_IN, S_POST] = ...
                        TBobj.SupportStruct.UpdateSupport(TBobj, PI_OUT);
            end
            
            % Now let's determine the values of the incoming active means
            % and variances.  We need to look at the type of structure
            % present in the signal amplitude matrix, GAMMA.  EM updates of
            % the Bernoulli-Gaussian model parameters occurs here as well.
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'               
                    % Compute posterior means and variances
                    POST_VAR = (1./RVAR + 1./obj.active_var).^(-1);
                    POST_MEAN = (RHAT ./ RVAR + ...
                        obj.active_mean ./ obj.active_var);
                    POST_MEAN = POST_VAR .* POST_MEAN;
                    
                    % Call the EM update method built into this class file
                    obj.LearnAmplitudeParams(TBobj, S_POST, POST_MEAN, ...
                        POST_VAR);
                    
                    % No amplitude structure, thus return priors as
                    % GAMP-bound messages
                    ETA_IN = obj.active_mean;
                    KAPPA_IN = obj.active_var;
                otherwise
                    % Use the outputs from GAMP to compute the messages 
                    % moving from the GAMP factor nodes to the amplitude 
                    % nodes, GAMMA.  These messages are Gaussian, with mean 
                    % ETA_OUT and variance KAPPA_OUT.  To compute these, we 
                    % require a Taylor series approximation which relies on 
                    % outgoing GAMP messages and incoming (to GAMP) support 
                    % probability messages.
                    [ETA_OUT, KAPPA_OUT] = taylor_approx(obj, PI_IN_OLD, ...
                        RHAT, RVAR);
                    
                    % Call the UpdateAmplitude method for the appropriate
                    % model of amplitude structure
                    [ETA_IN, KAPPA_IN, POST_MEAN, POST_VAR] = ...
                        TBobj.AmplitudeStruct.UpdateAmplitude(TBobj, ...
                        ETA_OUT, KAPPA_OUT);
            end
            
            % Okay, we've now taken into account all forms of signal
            % structure, and performed any required message passing.  We
            % now have everything we need to build the new EstimIn object
            % for the next iteration of GAMP
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    switch obj.data
                        case 'real'
                            EstimIn = AwgnEstimIn(ETA_IN, KAPPA_IN);
                        case 'complex'
                            EstimIn = CAwgnEstimIn(ETA_IN, KAPPA_IN);
                    end
                    EstimIn = SparseScaEstim(EstimIn, PI_IN);
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        switch obj.data
                            case 'real'
                                EstimIn{t} = AwgnEstimIn(ETA_IN(:,t), KAPPA_IN(:,t));
                            case 'complex'
                                EstimIn{t} = CAwgnEstimIn(ETA_IN(:,t), KAPPA_IN(:,t));
                        end
                        EstimIn{t} = SparseScaEstim(EstimIn{t}, PI_IN(:,t));
                    end
            end
        end
        
        
        % *****************************************************************
        %         	   INITIALIZE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        
        % Initialize EstimIn object for a Bernoulli-Gaussian signal prior
        function EstimIn = InitPriors(obj, TBobj, Y, A)
            
            % First get the problem sizes
            [M, T] = size(Y);
            switch TBobj.commonA
                case true, 
                    [M, N] = A.size();
                case false, 
                    [M, N] = A{1}.size();
%                     A = A{1};   % Use A(1) only for initialization
            end
            
            % Initialize differently based on user preference
            switch TBobj.Signal.init_params
                case 'false'
                    % Use user-specified parameter initializations, and
                    % resize them if needed to make them N-by-T in
                    % dimension
                    obj.active_mean = TBobj.resize(obj.active_mean, N, T);
                    obj.active_var = TBobj.resize(obj.active_var, N, T);
                    obj.sparsity_rate = TBobj.resize(obj.sparsity_rate, N, T);
                case 'true'
                    % Use simple heuristics involving Y and A to initialize
                    % the parameters
                    %Define cdf/pdf for Gaussian
                    normal_cdf = @(x) 1/2*(1 + erf(x/sqrt(2)));
                    normal_pdf = @(x) 1/sqrt(2*pi)*exp(-x.^2/2);
                    
                    theta = 0;  % Guess a zero active mean
                    
                    del = M/N;  % Undersampling rate
                    temp = linspace(0,10,1024);
                    rho_SE = (1 - (2/del)*((1+temp.^2).*normal_cdf(-temp)-...
                        temp.*normal_pdf(temp)))./(1 + temp.^2 - ...
                        2*((1+temp.^2).*normal_cdf(-temp)-temp.*normal_pdf(temp)));
                    rho_SE = max(rho_SE);
                    lambda = del*rho_SE;    % State evolution est. of lambda
%                     if isa(A, 'double')
%                         % Initialization of active var
%                         phi = (norm(Y, 'fro')^2 * (1 - 1/1001)) / ...
%                             sum((A.^2)*ones(M,1)) / T / lambda;
%                     elseif isa(A, 'LinTrans')
%                         phi = (norm(Y, 'fro')^2 * (1 - 1/1001)) / ...
%                             sum(A.multSqTr(ones(M,1))) / T / lambda;
%                     end
                    
                    if any(strcmpi('prior_var', properties(TBobj.Observation)))
                        noisevar = TBobj.Observation.prior_var;
                        if numel(noisevar) == M*T
                            noisevar = mean(noisevar, 1);
                        elseif numel(noisevar) == M
                            noisevar = mean(noisevar)*ones(1,T);
                        elseif numel(noisevar) == T
                            noisevar = reshape(noisevar, 1, T);
                        elseif numel(noisevar) == 1
                            noisevar = repmat(noisevar, 1, T);
                        else
                            noisevar = repmat(mean(noisevar(:)), 1, T);
                        end
                    else
                        noisevar = 1e-3*ones(1,T);    % Just a guess
                    end

                    if TBobj.commonA
                        phi = (sum(abs(Y).^2, 1) - M*noisevar) / ...
                            sum(A.multSqTr(ones(M,1)));
                        phi = TBobj.resize(phi, N, T) ./ ...
                            TBobj.resize(lambda, N, T);
                    else
                        phi = NaN(1,T);
                        for t = 1:T
                            phi(t) = (norm(Y(:,t))^2 - M*noisevar(t)) / ...
                                sum(A{t}.multSqTr(ones(M,1)));
                        end
                        phi = TBobj.resize(phi, N, T) ./ ...
                            TBobj.resize(lambda, N, T);
                    end
                    
                    % Place initializations into Signal structure
                    obj.sparsity_rate = TBobj.resize(lambda, N, T);
                    obj.active_mean = TBobj.resize(theta, N, T);
                    obj.active_var = TBobj.resize(phi, N, T);
            end
            
            % Check for compatibility of initializations with the amplitude
            % structure
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % Nothing to worry about in structure-less case
                case 'GM'
                    % Gauss-Markov random processes must have static means
                    % over either time or space
                    if numel(unique(TBobj.Signal.active_mean(:))) > N && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'row')
                        error(['Cannot specify more than N unique means' ...
                            ' for a Gauss-Markov process across columns of X'])
                    elseif numel(unique(TBobj.Signal.active_mean(:))) > T && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'col')
                        error(['Cannot specify more than T unique means' ...
                            ' for a Gauss-Markov process across rows of X'])
                    elseif numel(unique(TBobj.Signal.active_var(:))) > N && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'row')
                        error(['Cannot specify more than N unique variances' ...
                            ' for a Gauss-Markov process across columns of X'])
                    elseif numel(unique(TBobj.Signal.active_var(:))) > T && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'col')
                        error(['Cannot specify more than N unique variances' ...
                            ' for a Gauss-Markov process across rows of X'])
                    end
                otherwise
                    warning(['Unable to check validity of Bernoulli-' ...
                        'Gaussian inputs for this type of amplitude structure'])
            end
            
            % Form initial EstimIn object
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    switch obj.data
                        case 'real'
                            EstimIn = AwgnEstimIn(obj.active_mean, obj.active_var);
                        case 'complex'
                            EstimIn = CAwgnEstimIn(obj.active_mean, obj.active_var);
                    end
                    EstimIn = SparseScaEstim(EstimIn, obj.sparsity_rate);
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        switch obj.data
                            case 'real'
                                EstimIn{t} = AwgnEstimIn(obj.active_mean(:,t), ...
                                    obj.active_var(:,t));
                            case 'complex'
                                EstimIn{t} = CAwgnEstimIn(obj.active_mean(:,t), ...
                                    obj.active_var(:,t));
                        end
                        EstimIn{t} = SparseScaEstim(EstimIn{t}, ...
                            obj.sparsity_rate(:,t));
                    end
            end
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the
        % Bernoulli-Gaussian prior on X
        %
        % INPUTS:
        % obj       	An object of the BernGauss class
        % TBobj         An object of the TurboOpt class
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % X_TRUE        A realization of the signal matrix, X
        % S_TRUE        A realization of the support matrix, S
        % GAMMA_TRUE    A realization of the amplitude matrix, GAMMA
        function [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(obj, TBobj, GenParams)
            % Extract signal dimensions
            N = GenParams.N;
            T = GenParams.T;
            
            % Start by producing a realization of S
            switch TBobj.SupportStruct.get_type()
                case 'None'
                    % No support structure, so draw iid
                    S_TRUE = rand(N,T) < TBobj.resize(TBobj.Signal.sparsity_rate, N, T);
                otherwise
                    % Call the genRand method of the particular form of
                    % support structure to produce S_TRUE
                    SuppStruct = TBobj.SupportStruct;
                    S_TRUE = SuppStruct.genRand(TBobj, GenParams);
            end
            
            % Now produce a realization of GAMMA
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % No amplitude structure, so draw iid
                    MEAN = TBobj.resize(TBobj.Signal.active_mean, N, T);
                    VAR = TBobj.resize(TBobj.Signal.active_var, N, T);
                    switch GenParams.type
                        case 'real'
                            % Real-valued amplitudes
                            if strcmp(obj.data, 'complex')
                                warning(['Changing BernGauss.data to ' ...
                                    '''real'' to match GenParams.type'])
                                obj.data = 'real';
                            end
                            GAMMA_TRUE = MEAN + sqrt(VAR).*randn(N,T);
                        case 'complex'
                            % Complex-valued amplitudes
                            if strcmp(obj.data, 'real')
                                warning(['Changing BernGauss.data to ' ...
                                    '''complex'' to match GenParams.type'])
                                obj.data = 'complex';
                            end
                            GAMMA_TRUE = MEAN + sqrt(VAR/2).*randn(N,T) + ...
                                1j*sqrt(VAR/2).*randn(N,T);
                    end
                otherwise
                    % Call the genRand method of the particular form of
                    % amplitude structure to produce GAMMA_TRUE
                    AmpStruct = TBobj.AmplitudeStruct;
                    GAMMA_TRUE = AmpStruct.genRand(TBobj, GenParams);
            end
            
            % Combine S_TRUE and GAMMA_TRUE to yield X_TRUE
            X_TRUE = S_TRUE .* GAMMA_TRUE;
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = ~strcmpi(obj.learn_sparsity_rate, 'false') + ...
                ~strcmpi(obj.learn_active_mean, 'false') + ...
                ~strcmpi(obj.learn_active_var, 'false');
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'sparsity_rate',    'Sparsity rate',	'learn_sparsity_rate'; 
                        'active_mean',      'Gaussian mean',   	'learn_active_mean';
                        'active_var',       'Gaussian variance','learn_active_var'};
            % Iterate through each parameter, adding to Report array as
            % needed
			j = 0;
            for i = 1:size(Params, 1)
                switch obj.(Params{i,3})
                    case 'scalar'
						j = j + 1;
                        Report{j,1} = Params{i,1};
                        Report{j,2} = Params{i,2};
                        Report{j,3} = obj.(Params{i,1})(1,1);
                    case 'row'
						j = j + 1;
                        Report{j,1} = Params{i,1};
                        Report{j,2} = Params{i,2};
                        Report{j,3} = obj.(Params{i,1})(:,1);
                    case 'column'
						j = j + 1;
                        Report{j,1} = Params{i,1};
                        Report{j,2} = Params{i,2};
                        Report{j,3} = obj.(Params{i,1})(1,:);
                    otherwise
                        % Don't add this parameter to the report
                end
            end
        end
    end
    
    methods (Access = private)
        
        % *****************************************************************
        %         EM LEARNING OF AMPLITUDE MODEL PARAMETERS METHOD
        % *****************************************************************
        
        function LearnAmplitudeParams(obj, TBobj, S_POST, POST_MEAN, POST_VAR)
            % Learn amplitude parameters in the case where amplitude
            % variables are independent of one another

            [N, T] = size(obj.active_mean);
            
            % Initialize updates to old values, in case no EM learning is
            % to take place
            THETA_upd = obj.active_mean;
            PHI_upd = obj.active_var;
            
            % Normalize the posterior probabilities, Pr{s(n,t) = 1 | Y}, by
            % the sparsity rate (in accordance with the Vila/Schniter
            % EM-BG-AMP algorithm EM updates)
            S_POST = S_POST ./ obj.sparsity_rate;
            
            % Increment the EM iteration counter
            obj.EMcnt = obj.EMcnt + 1;
            
            if mod(obj.EMcnt, 2) == 0 && obj.EMcnt > 1
                switch obj.learn_active_mean
                    case 'scalar'
                        THETA_upd = sum(sum(S_POST .* POST_MEAN)) / N / T;
                    case 'row'
                        THETA_upd = sum(S_POST .* POST_MEAN, 2) / T;
                    case 'column'
                        THETA_upd = sum(S_POST .* POST_MEAN, 1) / N;
                end
            elseif mod(obj.EMcnt, 2) == 1 && obj.EMcnt > 1
                switch obj.learn_active_var
                    case 'scalar'
                        PHI_upd = (1/N/T) * sum(sum(S_POST .* (POST_VAR + ...
                            abs(POST_MEAN - obj.active_mean).^2)));
                    case 'row'
                        PHI_upd = (1/T) * sum(S_POST .* (POST_VAR + ...
                            abs(POST_MEAN - obj.active_mean).^2), 2);
                    case 'column'
                        PHI_upd = (1/N) * sum(S_POST .* (POST_VAR + ...
                            abs(POST_MEAN - obj.active_mean).^2), 1);
                end
            end
            obj.active_mean = TBobj.resize(THETA_upd, N, T);
            obj.active_var = TBobj.resize(PHI_upd, N, T);
            
%             dummy2 = sum(sum(S_POST));
%             gamma_n = (RHAT.*TBobj.Signal.active_var + RVAR.*TBobj.Signal.active_mean)./TempVar;
%             nu_n = RVAR.*TBobj.Signal.active_var./TempVar;
%             if strcmpi('scalar', TBobj.Signal.learn_active_mean)
%                 theta_upd = real(sum(sum(S_POST.*gamma_n))/dummy2);
%                 TBobj.Signal.active_mean = repmat(theta_upd, size(TBobj.Signal.active_mean));
%             elseif sum(strcmpi({'row', 'column'}, TBobj.Signal.learn_active_mean)) ~= 0
%                 error('Learning rows or columns not yet supported')
%             end
%             if strcmpi('scalar', TBobj.Signal.learn_active_var)
%                 phi_upd = sum(sum(S_POST.*(nu_n+(gamma_n-TBobj.Signal.active_mean).^2)))/dummy2;
%                 TBobj.Signal.active_var = repmat(phi_upd, size(TBobj.Signal.active_var));
%             elseif sum(strcmpi({'row', 'column'}, TBobj.Signal.learn_active_var)) ~= 0
%                 error('Learning rows or columns not yet supported')
%             end
        end
        
        
        % *****************************************************************
        %         TAYLOR APPROX OF OUTGOING GAMP MESSAGES METHOD
        % *****************************************************************
        
        % TAYLOR_APPROX         Method for computing a Taylor series
        % approximation of a single Gaussian message from a GAMP factor
        % node to an amplitude node, GAMMA(n,t)
        %
        % INPUTS
        %  PI_IN        Incoming activity probabilities (to GAMP)
        %  RHAT         GAMP variable
        %  RVAR         GAMP variable
        %
        % OUTPUTS
        %  ETA_OUT      Outbound message of mean of theta
        %  KAPPA_OUT    Outbound message of variance of theta
        %
        function [ETA_OUT, KAPPA_OUT] = taylor_approx(obj, PI_IN, RHAT, RVAR)
                    
            % Taylor series approximation about RHAT is used to derive means
            % and variances
            THETA_0 = RHAT;      % Taylor series expansion point

            % Compute the various factors
            switch obj.data
                case 'real'
                    Omega = (obj.eps*PI_IN)./((1 - PI_IN) + obj.eps*PI_IN);
                    Omega(Omega < sqrt(realmin)) = sqrt(realmin);
                    alpha = obj.eps * (1 - Omega);
                    alpha_bar = Omega;
                    beta = obj.eps^2/2./RVAR .* abs(THETA_0 - ...
                        RHAT/obj.eps).^2;
                    delta = -obj.eps^2./RVAR .* (real(THETA_0) - ...
                        real(RHAT)/obj.eps);
            
                case 'complex'
                    Omega = (obj.eps^2*PI_IN)./((1 - PI_IN) + obj.eps^2*PI_IN);
                    Omega(Omega < sqrt(realmin)) = sqrt(realmin);
                    alpha = obj.eps^2 * (1 - Omega);
                    alpha_bar = Omega;
                    beta = (obj.eps^2./RVAR) .* abs(THETA_0 - ...
                        RHAT/obj.eps).^2;
                    delta = -(obj.eps^2./RHAT) .* (2*real(THETA_0) - ...
                        2*real(RHAT)/obj.eps);
                    gamma = -obj.eps^2./RHAT .* (2*imag(THETA_0) - ...
                        2*imag(RHAT)/obj.eps);
            end

            % Use factors to compute outgoing variance.  Perform computation
            % slightly differently for coefficients with large abs(beta)'s for
            % numerical precision reasons
            switch obj.data
                case 'real'
                    LB = find(abs(beta) > 10);      % Indices with large betas
                    SB = find(abs(beta) <= 10);     % Indices with small betas
                    numer = NaN(size(RHAT));
                    denom = NaN(size(RHAT));
                    numer(SB) = alpha(SB).^2 .* exp(-(beta(SB))) + ...
                        alpha(SB) .* alpha_bar(SB) + alpha_bar(SB).^2 .* exp(beta(SB));
                    denom(SB) = (obj.eps^2./RVAR(SB)).*alpha(SB).^2 .* exp(-(beta(SB))) + ...
                        ((obj.eps^2 + 1)./RVAR(SB) - (1/2)*delta(SB).^2).*alpha(SB) .* ...
                        alpha_bar(SB) + (1./RVAR(SB)).*alpha_bar(SB).^2 .* exp(beta(SB));
                    numer(LB) = alpha(LB).^2 .* exp(-2*beta(LB)) + ...
                        alpha(LB) .* alpha_bar(LB) .* ...
                        exp(-beta(LB)) + alpha_bar(LB).^2;
                    denom(LB) = (obj.eps^2./RVAR(LB)).*alpha(LB).^2 .* exp(-2*beta(LB)) + ...
                        ((obj.eps^2 + 1)./RVAR(LB) - (1/2)*delta(LB).^2) .* alpha(LB) .* ...
                        alpha_bar(LB) .* exp(-beta(LB)) + (1./RVAR(LB)).*alpha_bar(LB).^2;
                    
                    KAPPA_OUT = numer ./ denom; 
                    % Use factors to compute compute outgoing mean
                    ETA_OUT = RHAT - (1/2)*KAPPA_OUT.*(-alpha.*delta.*...
                        exp(-beta))./(alpha.*exp(-beta) + alpha_bar);
            
                case 'complex'
                    LB = find(abs(beta) > 10);      % Indices with large betas
                    SB = find(abs(beta) <= 10);     % Indices with small betas
                    numer = NaN*ones(size(RHAT));
                    denom = NaN*ones(size(RHAT));
                    numer(SB) = alpha(SB).^2 .* exp(-(beta(SB))) + ...
                        alpha(SB) .* alpha_bar(SB) + alpha_bar(SB).^2 .* exp(beta(SB));
                    denom(SB) = (obj.eps^2./RVAR(SB)).*alpha(SB).^2 .* exp(-(beta(SB))) + ...
                        ((obj.eps^2 + 1)./RVAR(SB) - (1/2)*delta(SB).^2).*alpha(SB) .* ...
                        alpha_bar(SB) + (1./RVAR(SB)).*alpha_bar(SB).^2 .* exp(beta(SB));
                    numer(LB) = alpha(LB).^2 .* exp(-2*beta(LB)) + ...
                        alpha(LB) .* alpha_bar(LB) .* ...
                        exp(-beta(LB)) + alpha_bar(LB).^2;
                    denom(LB) = (obj.eps^2./RVAR(LB)).*alpha(LB).^2 .* exp(-2*beta(LB)) + ...
                        ((obj.eps^2 + 1)./RVAR(LB) - (1/2)*delta(LB).^2) .* alpha(LB) .* ...
                        alpha_bar(LB) .* exp(-beta(LB)) + (1./RVAR(LB)).*alpha_bar(LB).^2;
                    
                    KAPPA_OUT = numer ./ denom;
                    % Use factors to compute compute outgoing mean
                    ETA_REAL = real(RHAT) - (1/2)*KAPPA_OUT.*(-alpha.*delta.*...
                        exp(-beta))./(alpha.*exp(-beta) + alpha_bar);
                    ETA_IMAG = imag(RHAT) - (1/2)*KAPPA_OUT.*(-alpha.*gamma.*...
                        exp(-beta))./(alpha.*exp(-beta) + alpha_bar);
                    ETA_OUT = ETA_REAL + 1j*ETA_IMAG;
                    
                    % For numerical reasons, KAPPA_OUT might appear to be complex.
                    % Force to be real, but run a sanity check first.
                    if norm(imag(KAPPA_OUT(:))) > 1e-1
                        warning('Non-negligible imaginary variances encountered')
                    end
                    KAPPA_OUT = real(KAPPA_OUT);
            end

            if 0 < obj.tau && obj.tau < 1
                % Use thresholding instead of Taylor approx
                ETA_OUT = RHAT/obj.eps;
                KAPPA_OUT = RVAR/obj.eps^2;
                ETA_OUT(PI_IN > obj.tau) = RHAT(PI_IN > obj.tau);
                KAPPA_OUT(PI_IN > obj.tau) = RVAR(PI_IN > obj.tau);
            end
        end
        
        
        % *****************************************************************
        %                          HELPER METHODS
        % *****************************************************************
        
        % This method makes sure that the inputs to learn_sparsity_rate,
        % etc. are valid options
        function flag = check_input(obj, string)
            flag = 1;
            if ~isa(string, 'char')
                flag = 0;
            elseif sum(strcmpi(string, {'scalar', 'row', 'column', ...
                    'false'})) == 0
                flag = 0;
            end
        end
        
        % This method is called by the print method in order to format how
        % properties are printed.  Properties that are scalars will have
        % their values printed to the command window, while arrays will
        % have their dimensions, minimal, and maximal values printed to the
        % screen
        function string = form(obj, prop)
            if numel(prop) == 1
                string = num2str(prop);
            else
                string = sprintf('%d-by-%d array (Min: %g, Max: %g)', ...
                    size(prop, 1), size(prop, 2), min(min(prop)), ...
                    max(max(prop)));
            end
        end
    end % Private methods
   
end % classdef