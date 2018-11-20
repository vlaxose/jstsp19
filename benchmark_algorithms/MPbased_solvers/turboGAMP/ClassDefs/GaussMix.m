% CLASS: GaussMix
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Signal
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The GaussMix class contains the parameters needed to define a
%   Bernoulli/Mixture-of-Gaussians marginal prior distribution for each
%   signal coefficient.  It is probably easiest to first describe this
%   distribution in words, prior to describing it mathematically.  Briefly,
%   each signal coefficient, X(n,t), equals zero with some prior
%   probability.  If X(n,t) is non-zero, then it is assumed to be drawn
%   from one of D different "mixture components", each of which is a
%   Gaussian distribution of (possibly) distinct means and variances.  The
%   relative weight assigned to each mixture component corresponds to the
%   prior probability that X(n,t) is drawn from that particular Gaussian
%   distribution.
%
%   Now we will define this prior distribution mathematically.  In what
%   follows, we will use MATLAB's tensor convention to index variables
%   corresponding to distinct mixture components by using the third
%   dimension of the tensor as an index into the particular active mixture
%   component.  This prior distribution is uniquely defined by three
%   tensor variables, LAMBDA, THETA, and PHI.  Their interpretation is as
%   follows: LAMBDA(n,t,d) equals the prior probability that X(n,t) is
%   drawn from the d-th Gaussian mixture component, (d = 1, ..., D).
%   THETA(n,t,d) is the mean of the d-th Gaussian mixture component at
%   location (n,t).  Similarly, PHI(n,t,d) denotes the variance of the d-th
%   Gaussian mixture component.  Note that no parameters are required to
%   describe the inactive component of the prior; the prior probability
%   that X(n,t) equals zero is 1 - ( LAMBDA(n,t,1) + ... + LAMBDA(n,t,D) ).
%   Given these three variables, each signal coefficient, X(n,t), has a 
%   marginal prior distribution given by
%   pdf(X(n,t)) = (1 - sum(LAMBDA(n,t,:))*delta(X(n,t)) + 
%                 LAMBDA(n,t,1)*Normal(X(n,t); THETA(n,t,1), PHI(n,t,1)) + 
%                 LAMBDA(n,t,2)*Normal(X(n,t); THETA(n,t,2), PHI(n,t,2)) + 
%                 ... +
%                 LAMBDA(n,t,D)*Normal(X(n,t); THETA(n,t,D), PHI(n,t,D)),
%   where n = 1,...,N indexes the row of the signal matrix X, 
%   t = 1,...,T indexes the column (timestep) of the signal matrix X,
%   delta(.) is the Dirac delta function, and Normal(.; theta, phi) is
%   either a real- or complex-valued normal distribution with mean theta
%   and variance phi.  Note that this class can be used in the single
%   measurement vector (SMV) problem as well, (i.e., T = 1).  By default,
%   it is assumed that the signal is real-valued.  If complex-valued, set
%   data = 'complex'.
%
%   Note that the parameters LAMBDA, THETA, and PHI can be initialized in 
%   a number of different ways.  In all cases, the third dimension of the 
%   tensor, which we will refer to as a plane, is used to index a specific
%   active mixture component.  If each plane contains a scalar, then the
%   same value at each plane is applied to all N*T elements.  For example,
%   consider setting the property sparsity_rate (LAMBDA) as follows:
%       sparsity_rate = ones(1,1,3);
%       sparsity_rate(1,1,:) = [0.05, 0.10, 0.15];
%   This setting is interpreted as being equivalent to setting
%   LAMBDA(n,t,1) = 0.05 for all n,t, and LAMBDA(n,t,2) = 0.10 for all n,t,
%   and LAMBDA(n,t,3) = 0.15 for all n,t.  Similarly, if each plane
%   contains a row (column) vector, than the row (column) vector in each 
%   plane will be replicated over T columns (N rows).  Finally, parameters
%   can be initialized as full N-by-T-by-D tensors if one wishes to assign
%   a unique value to each (n,t) element of each of the D active
%   components.  Note also that if the user wishes to have EMTurboGAMP 
%   attempt to intelligently initialize LAMBDA, THETA, and PHI based on the 
%   matrix of measurements, Y, and the sensing matrix A, then 
%   GaussMix('init_params', 'true') is the appropriate constructor.
%
%   To create a GaussMix object, there are two constructors to choose from
%   (see METHODS section below).  The default constructor, GaussMix(), 
%   will create a GaussMix object initialized with all default values for 
%   each parameter/property.  The alternative constructor allows the user 
%   to initialize any subset of the parameters, with the remaining 
%   parameters initialized to their default values, by using MATLAB's 
%   property/value string pairs convention, e.g.,
%   GaussMix('active_mean', MEAN, 'learn_active_mean', 'false') will
%   construct a GaussMix object in which THETA is initialized by a tensor
%   (assumed declared already) named MEAN.  Furthermore, this THETA will 
%   not be refined by the expectation-maximization (EM) parameter learning 
%   procedure.  Remaining parameters will be set to their default values.  
%
%   Additionally, this class contains information about which
%   parameters should be learned from the data, and in what manner, (see
%   PROPERTIES below).  As an example, to prevent the parameter
%   sparsity_rate (LAMBDA) from being learned by an EM algorithm, set
%   learn_sparsity_rate = 'false'.  To learn a scalar sparsity rate
%   for each mixture component (i.e., LAMBDA(n,t,d) = sparsity_rate(1,1,d) 
%   for all n and t, set learn_sparsity_rate = 'scalar'.  Likewise, to 
%   learn a unique sparsity rate for each row (column) of X, for each 
%   mixture component, set learn_sparsity_rate = 'row' ('column').
%
% PROPERTIES (State variables)
%   sparsity_rate           The prior sparsity rate(s), LAMBDA  [Default:
%                           0.05]
%   learn_sparsity_rate     Learn sparsity rates using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'scalar']
%   active_mean             The real or complex active Gaussian component
%                           prior mean(s), THETA  [Default: 0]
%   learn_active_mean       Learn prior means using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'scalar']
%   active_var              The active Gaussian component prior variances,
%                           PHI  [Default: 1]
%   learn_active_var        Learn prior variances using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'scalar']
%   init_params             Initialize LAMBDA, THETA, and PHI automatically
%                           using Y and A ('true'), or not ('false').
%                           [Default: 'false']
%   data                    Real-valued signal ('real') or complex-valued
%                           ('complex')? [Default: 'real']
%
% METHODS (Subroutines/functions)
%   GaussMix()
%       - Default constructor.  Assigns all properties to their default 
%         values, which corresponds to a Bernoulli-Gaussian prior on each
%         element of X (i.e., D = 1)
%   GaussMix('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class GaussMix, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'MoG' when obj is a GaussMix object
%   GaussMixCopyObj = copy(obj)
%       - Creates an independent copy of the GaussMix object, obj
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
%         (e.g., 'sparsity_rate'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Sparsity rate'), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (12/20/11; JAZ)
%       - Added support for time-varying transform matrix A(t) (02/07/12;
%         JAZ)
%       - Added ability to return support posteriors to UpdatePriors
%         method (04/12/12; JAZ)
%       - Added support for complex-valued signals (05/22/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/28/13; JAZ)
% Version 0.2
%

classdef GaussMix < Signal

    properties
        % Mixture-of-Gaussian default prior family properties (amounts to a
        % Bernoulli-Gaussian prior)
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
        type = 'MoG';     	% Mixture-of-Gaussians type identifier
        eps = 1e-6;         % Small positive constant for Taylor approx
    end
    
    properties (Hidden)
        FirstIter = true;   % Flag for first turbo iteration to ignore EM
                            % updates of parameters
    end
    
    properties (Dependent = true, SetAccess = private)
        D;                  % Number of active Gaussian mixture components
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = GaussMix(varargin)
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
           if any(LAMBDA(:) < 0) || any(LAMBDA(:) > 1)
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
           if any(PHI(:) < 0)
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
            if sum(strcmpi(string, {'real', 'complex'})) == 0
                error('Invalid option: data')
            end
            obj.data = lower(string);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('GaussMix does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        
        % *****************************************************************
        %                        GET METHOD
        % *****************************************************************
        
        % Get the number of active Gaussian mixture components (D)
        function D = get.D(obj)
            if size(obj.sparsity_rate, 3) ~= size(obj.active_mean, 3) || ...
                    size(obj.sparsity_rate, 3) ~= size(obj.active_var, 3)
                error(['Inconsistent third dimension (D) between ' ...
                    'LAMBDA, THETA, and PHI'])
            else
                D = size(obj.sparsity_rate, 3);
            end
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SIGNAL PRIOR: Bernoulli/Mixture-of-Gaussians\n')
            fprintf('# of active Gaussian mixture components: %d\n', obj.D)
            fprintf('         sparsity_rate: %s\n', ...
                form(obj, obj.sparsity_rate))
            fprintf('           active_mean: %s\n', ...
                form(obj, obj.active_mean))
            fprintf('            active_var: %s\n', ...
                form(obj, obj.active_var))
            fprintf('   learn_sparsity_rate: %s\n', obj.learn_sparsity_rate)
            fprintf('     learn_active_mean: %s\n', obj.learn_active_mean)
            fprintf('      learn_active_var: %s\n', obj.learn_active_var)
            fprintf('           init_params: %s\n', obj.init_params)
            fprintf('                  data: %s\n', obj.data)
        end
        
        
        % *****************************************************************
        %                           COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of the GaussMix object
        function GaussMixCopyObj = copy(obj)
            GaussMixCopyObj = GaussMix('sparsity_rate', ...
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
            [~, ~, RHAT, RVAR] = GAMPState.getState();
            N = size(RHAT, 1);
            T = size(RHAT, 2);
            D = obj.D;
            % Set a minimum allowable value for RVAR
            RVAR(RVAR < 1e-6) = 1e-6;
            
            % Unpack inputs given to GAMP at the last iteration
            switch TBobj.commonA
                case true
                    PI_IN_OLD = EstimInOld.OMEGA .* ...
                        repmat(EstimInOld.LAMBDA, [1, 1, D]);
                    ETA_IN_OLD = EstimInOld.THETA;
                    KAPPA_IN_OLD = EstimInOld.PHI;
                case false
                    PI_IN_OLD = NaN(N,T,D);  ETA_IN_OLD = NaN(N,T,D);
                    KAPPA_IN_OLD = NaN(N,T,D);
                    for t = 1:T
                        PI_IN_OLD(:,t,:) = EstimInOld{t}.OMEGA .* ...
                            repmat(EstimInOld{t}.LAMBDA, [1, 1, D]);
                        ETA_IN_OLD(:,t,:) = EstimInOld{t}.THETA;
                        KAPPA_IN_OLD(:,t,:) = EstimInOld{t}.PHI;
                    end
            end
            
            % Use outputs from GAMP to compute the messages moving
            % from the GAMP factor nodes to the support nodes, S
            PI_OUT = support_outputs(obj, ETA_IN_OLD, KAPPA_IN_OLD, ...
                RHAT, RVAR);
                    
            % Now, let's determine the values of the incoming (to GAMP)
            % activity probabilities, PI_IN.  We need to look at the type
            % of structure present in the signal support matrix, S
            switch TBobj.SupportStruct.get_type()
                case 'None'
                    % No structured support, thus return the priors as the
                    % updated messages for GAMP
                    PI_IN = obj.sparsity_rate;
            
                    % Compute posteriors, Pr{S(n,t) = d | Y}, d = 1, ..., D
                    S_POST = NaN(size(PI_IN));
                    for d = 1:D
                        S_POST(:,:,d) = (obj.sparsity_rate(:,:,d) .* ...
                            PI_OUT(:,:,d));
                    end
                    NORM = sum(S_POST, 3) + (1 - sum(obj.sparsity_rate, 3)) .* ...
                        (1 - sum(PI_OUT, 3));
                    S_POST = S_POST ./ repmat(NORM, [1, 1, D]);
                    
                    % If the user has requested EM refinement of LAMBDA, do that
                    % now.  Note that certain update types may not make sense,
                    % e.g., learning a different LAMBDA value for each column of S
                    % would not make sense for the MMV problem, in which all
                    % columns share the same prior sparsity rate
                    switch TBobj.Signal.learn_sparsity_rate
                        case 'scalar'
                            % Update a single scalar
                            lambda_upd = sum(sum(S_POST, 1), 2) / N / T;
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
                    obj.sparsity_rate = TBobj.resize(lambda_upd, N, T, D);
                    
                otherwise
                    % Call the UpdateSupport method for the appropriate
                    % model of support structure
                    [PI_IN, S_POST] = ...
                        TBobj.SupportStruct.UpdateSupport(TBobj, PI_OUT);
            end
            
            % Use the outputs from GAMP to compute the messages moving from
            % the GAMP factor nodes to the amplitude nodes, GAMMA.  These
            % messages are Gaussian, with mean ETA_OUT and variance
            % KAPPA_OUT.  To compute these, we require a Taylor series
            % approximation which relies on outgoing GAMP messages and
            % incoming (to GAMP) support probability messages.
            [ETA_OUT, KAPPA_OUT] = taylor_approx(obj, PI_IN_OLD, ...
                ETA_IN_OLD, KAPPA_IN_OLD, RHAT, RVAR);
            
            % Now let's determine the values of the incoming active means
            % and variances.  We need to look at the type of structure
            % present in the signal amplitude matrix, GAMMA.  EM parameter
            % learning of means and variances occur here as well
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % No amplitude structure, thus return priors as
                    % GAMP-bound messages
                    ETA_IN = obj.active_mean;
                    KAPPA_IN = obj.active_var;
                    
                    % Compute posterior means and variances
                    POST_VAR = (1./KAPPA_OUT + 1./obj.active_var).^(-1);
                    POST_MEAN = (ETA_OUT ./ KAPPA_OUT + ...
                        obj.active_mean ./ obj.active_var);
                    POST_MEAN = POST_VAR .* POST_MEAN;
                    
                    % Call the EM update method built into this class file
                    [THETA_upd, PHI_upd] = LearnAmplitudeParams(obj, ...
                        POST_MEAN, POST_VAR);
                    TBobj.Signal.active_mean = TBobj.resize(THETA_upd, ...
                        N, T, D);
                    TBobj.Signal.active_var = TBobj.resize(PHI_upd, ...
                        N, T, D);
                otherwise
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
            OMEGA = PI_IN ./ repmat(sum(PI_IN, 3), [1, 1, D]);
            OMEGA(isnan(OMEGA)) = 0;    % Needed for support-aware genie
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    switch obj.data
                        case 'real'
                            EstimIn = GM2EstimIn(sum(PI_IN, 3), OMEGA, ...
                                ETA_IN, KAPPA_IN);
                        case 'complex'
                            EstimIn = CGM2EstimIn(sum(PI_IN, 3), OMEGA, ...
                                ETA_IN, KAPPA_IN);
                    end
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    switch obj.data
                        case 'real'
                            for t = 1:T
                                EstimIn{t} = GM2EstimIn(sum(PI_IN(:,t,:), 3), ...
                                    OMEGA(:,t,:), ETA_IN(:,t,:), KAPPA_IN(:,t,:));
                            end
                        case 'complex'
                            for t = 1:T
                                EstimIn{t} = CGM2EstimIn(sum(PI_IN(:,t,:), 3), ...
                                    OMEGA(:,t,:), ETA_IN(:,t,:), KAPPA_IN(:,t,:));
                            end
                    end
            end
        end
        
        
        % *****************************************************************
        %         	   INITIALIZE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        
        % Initialize EstimIn object for a Bernoulli/MoG signal prior
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
            D = obj.D;  % Get the # of active mixture components
            
            % Initialize differently based on user preference
            switch TBobj.Signal.init_params
                case 'false'
                    % Use user-specified parameter initializations, and
                    % resize them if needed to make them N-by-T-by-D in
                    % dimension
                    obj.active_mean = TBobj.resize(obj.active_mean, N, T, D);
                    obj.active_var = TBobj.resize(obj.active_var, N, T, D);
                    obj.sparsity_rate = TBobj.resize(obj.sparsity_rate, N, T, D);
                case 'true'
                    % Use simple heuristics involving Y and A to initialize
                    % the parameters
                    %Define cdf/pdf for Gaussian
                    normal_cdf = @(x) 1/2*(1 + erf(x/sqrt(2)));
                    normal_pdf = @(x) 1/sqrt(2*pi)*exp(-x.^2/2);
                    
                    % Guess initial active mean for each component that is
                    % spread across the interval [-1, 1] and distinct for 
                    % each component
                    theta = linspace(0, 0, D);   % Scalar for each component
                    theta = reshape(theta, [1, 1, D]);
                    
                    del = M/N;  % Undersampling rate
                    temp = linspace(0,10,1024);
                    rho_SE = (1 - (2/del)*((1+temp.^2).*normal_cdf(-temp)-...
                        temp.*normal_pdf(temp)))./(1 + temp.^2 - ...
                        2*((1+temp.^2).*normal_cdf(-temp)-temp.*normal_pdf(temp)));
                    rho_SE = max(rho_SE);
                    lambda1 = del*rho_SE;    % State evolution est. of lambda
                    
                    % Now break activity rate lambda into D equal weights,
                    % implying each mixture component is equally likely
                    lambda = (lambda1/D) * ones(1,1,D);
                    
                    if isa(A, 'double')
                        % Initialization of active var
                        phi = (norm(Y, 'fro')^2 * (1 - 1/1001)) / ...
                            sum((A.^2).'*ones(M,1)) / T / lambda1;
                    else
%                         if TBobj.commonA
% %                             phi = (norm(Y, 'fro')^2 * (1 - 1/1001)) / ...
% %                                 sum(A.multSqTr(ones(M,1))) / T / lambda1;
%                             backproj = A.multTr(Y);
%                             phi = var(backproj(find(abs(backproj) > 1e-3)));
%                         else
% %                             phi = (norm(Y, 'fro')^2 * (1 - 1/1001)) / ...
% %                                 sum(A{1}.multSqTr(ones(M,1))) / T / lambda1;
%                             backproj = NaN(N,T);
%                             for t = 1:T
%                                 backproj(:,t) = A{t}.multTr(Y(:,t));
%                             end
%                             phi = var(backproj(find(abs(backproj) > 1e-3)));
%                         end
                        
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
                            noisevar = 1e-3;    % Just a guess
                        end
                        
                        
                        if TBobj.commonA
                            phi = (sum(abs(Y).^2, 1) - M*noisevar) / ...
                                sum(A.multSqTr(ones(M,1)));
                            phi = TBobj.resize(phi, N, T, D)/sqrt(D) ./ ...
                                TBobj.resize(lambda, N, T, D);
                            % Right now all components would be
                            % initialized with the same means and
                            % variances.  Multiply the variances by a
                            % logarithmic factor to incorporate some
                            % diversity in our estimates.
                            mult = logspace(-2, 2, D);
                            for d = 1:D
                                phi(:,:,d) = mult(d)*phi(:,:,d);
                            end
                        else
                            phi = NaN(1,T);
                            for t = 1:T
                                phi(t) = (norm(Y(:,t))^2 - M*noisevar(t)) / ...
                                    sum(A{t}.multSqTr(ones(M,1)));
                            end
                            phi = TBobj.resize(phi, N, T, D)/sqrt(D) ./ ...
                                TBobj.resize(lambda, N, T, D);
                            mult = logspace(-2, 2, D);
                            for d = 1:D
                                phi(:,:,d) = mult(d)*phi(:,:,d);
                            end
                        end
                    end
%                     phi = phi / D;
%                     phi = phi*ones(1,1,D);      % Duplicate

                    % Place initializations into Signal structure
                    TBobj.Signal.sparsity_rate = TBobj.resize(lambda, N, T, D);
                    TBobj.Signal.active_mean = TBobj.resize(theta, N, T, D);
                    TBobj.Signal.active_var = TBobj.resize(phi, N, T, D);
            end
            
            % Check for compatibility of initializations with the amplitude
            % structure (to prevent user from assigning unique means across
            % amplitudes that form a Gauss-Markov process, e.g.)
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % Nothing to worry about in structure-less case
                case 'GM'
                    % Gauss-Markov random processes must have static means
                    % over either time or space
                    if numel(unique(TBobj.Signal.active_mean(:))) > N*D && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'row')
                        error(['Cannot specify more than N*D unique means' ...
                            ' for a Gauss-Markov process across columns of X'])
                    elseif numel(unique(TBobj.Signal.active_mean(:))) > T*D && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'col')
                        error(['Cannot specify more than T*D unique means' ...
                            ' for a Gauss-Markov process across rows of X'])
                    elseif numel(unique(TBobj.Signal.active_var(:))) > N*D && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'row')
                        error(['Cannot specify more than N*D unique variances' ...
                            ' for a Gauss-Markov process across columns of X'])
                    elseif numel(unique(TBobj.Signal.active_var(:))) > T*D && ...
                            strcmpi(TBobj.AmplitudeStruct.dim, 'col')
                        error(['Cannot specify more than N*D unique variances' ...
                            ' for a Gauss-Markov process across rows of X'])
                    end
                otherwise
                    warning(['Unable to check validity of Bernoulli/' ...
                        'Mixture-of-Gaussian inputs for this type of '...
                        'amplitude structure'])
            end
            
            % Build the initial EstimIn object for GAMP's first iteration,
            % using the newly initialized parameters
            OMEGA = obj.sparsity_rate ./ ...
                repmat(sum(obj.sparsity_rate, 3), [1, 1, obj.D]);
            OMEGA(isnan(OMEGA)) = 0;        % For genie version
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    switch obj.data
                        case 'real'
                            EstimIn = GM2EstimIn(sum(obj.sparsity_rate, 3), ...
                                OMEGA, obj.active_mean, obj.active_var);
                        case 'complex'
                            EstimIn = CGM2EstimIn(sum(obj.sparsity_rate, 3), ...
                                OMEGA, obj.active_mean, obj.active_var);
                    end
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    switch obj.data
                        case 'real'
                            for t = 1:T
                                EstimIn{t} = ...
                                    GM2EstimIn(sum(obj.sparsity_rate(:,t,:), 3), ...
                                    OMEGA(:,t,:), obj.active_mean(:,t,:), ...
                                    obj.active_var(:,t,:));
                            end
                        case 'complex'
                            for t = 1:T
                                EstimIn{t} = ...
                                    CGM2EstimIn(sum(obj.sparsity_rate(:,t,:), 3), ...
                                    OMEGA(:,t,:), obj.active_mean(:,t,:), ...
                                    obj.active_var(:,t,:));
                            end
                    end
            end
        end
        
        
        % *****************************************************************
        %         	   GENERATE BERNOULLI/GAUSS-MIX REALIZATION
        % *****************************************************************
        
        function [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(obj, TBobj, GenParams)
            % Extract signal dimensions
            N = GenParams.N;
            T = GenParams.T;
            D = obj.D;
            
            % Resize key model parameters
            LAMBDA = TBobj.resize(obj.sparsity_rate, N, T, D);
            THETA = TBobj.resize(obj.active_mean, N, T, D);
            PHI = TBobj.resize(obj.active_var, N, T, D);
            
            % Start by producing a realization of S
            switch TBobj.SupportStruct.get_type()
                case 'None'
                    % No support structure, so draw iid
                    LAMBDASUM = cat(3, zeros(N,T), cumsum(LAMBDA, 3));
                    RANDMTX = rand(N,T);
                    S_TRUE = zeros(N,T);
                    
                    for d = 1:D
                        inds = (RANDMTX > LAMBDASUM(:,:,d) & RANDMTX < ...
                            LAMBDASUM(:,:,d+1));    % Logical indexing
                        S_TRUE(inds) = d;
                    end
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
                    switch GenParams.type
                        case 'real'
                            % Real-valued amplitudes
                            if strcmp(obj.data, 'complex')
                                warning(['Changing GaussMix.data to ' ...
                                    '''real'' to match GenParams.type'])
                                obj.data = 'real';
                            end
                            GAMMA_TRUE = THETA + sqrt(PHI).*randn(N,T,D);
                        case 'complex'
                            % Complex-valued amplitudes
                            if strcmp(obj.data, 'real')
                                warning(['Changing GaussMix.data to ' ...
                                    '''complex'' to match GenParams.type'])
                                obj.data = 'complex';
                            end
                            GAMMA_TRUE = THETA + sqrt(PHI/2).*randn(N,T,D) + ...
                                1j*sqrt(PHI/2).*randn(N,T,D);
                    end
                otherwise
                    % Call the genRand method of the particular form of
                    % amplitude structure to produce GAMMA_TRUE
                    AmpStruct = TBobj.AmplitudeStruct;
                    GAMMA_TRUE = AmpStruct.genRand(TBobj, GenParams);
            end
            
            % Use S_TRUE and GAMMA_TRUE to produce X_TRUE
            X_TRUE = zeros(N,T);
            for d = 1:D
                % Find S_TRUE locations equal to d
                inds = (S_TRUE == d);
                TEMP = GAMMA_TRUE(:,:,d);
                X_TRUE(inds) = TEMP(inds);
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = ~strcmpi(obj.learn_sparsity_rate, 'false') + ...
                ~strcmpi(obj.learn_active_mean, 'false') + ...
                ~strcmpi(obj.learn_active_var, 'false');
            Report = cell(obj.D*Nparam, 3);   % Declare Report array
            Params = {  'sparsity_rate',    'Sparsity rate',	'learn_sparsity_rate'; 
                        'active_mean',      'Gaussian mean',   	'learn_active_mean';
                        'active_var',       'Gaussian variance','learn_active_var'};
            % Iterate through each parameter, adding to Report array as
            % needed
            j = 0;
            for d = 1:obj.D
                for i = 1:size(Params, 1)
                    switch obj.(Params{i,3})
                        case 'scalar'
                            j = j + 1;
                            Report{j,1} = Params{i,1};
                            Report{j,2} = [Params{i,2}, ' #', num2str(d)];
                            Report{j,3} = obj.(Params{i,1})(1,1,d);
                        case 'row'
                            j = j + 1;
                            Report{j,1} = Params{i,1};
                            Report{j,2} = [Params{i,2}, ' #', num2str(d)];
                            Report{j,3} = obj.(Params{i,1})(:,1,d);
                        case 'column'
                            j = j + 1;
                            Report{j,1} = Params{i,1};
                            Report{j,2} = [Params{i,2}, ' #', num2str(d)];
                            Report{j,3} = obj.(Params{i,1})(1,:,d);
                        otherwise
                            % Don't add this parameter to the report
                    end
                end
            end
        end
    end
    
    methods (Access = private)
        
        % *****************************************************************
        %         EM LEARNING OF AMPLITUDE MODEL PARAMETERS METHOD
        % *****************************************************************
        
        function [THETA_upd, PHI_upd] = LearnAmplitudeParams(obj, ...
                POST_MEAN, POST_VAR)
            % Learn amplitude parameters in the case where amplitude
            % variables are independent of one another
            N = size(POST_MEAN, 1);
            T = size(POST_MEAN, 2);
            
            % Initialize updates to old values, in case no EM learning is
            % to take place
            THETA_upd = obj.active_mean;
            PHI_upd = obj.active_var;
            
            % Make sure this isn't the first turbo iteration.  If it is,
            % return using the previous estimates
            if obj.FirstIter
                obj.FirstIter = false;
                return
            end
            
            switch obj.learn_active_mean
                case 'scalar'
                    THETA_upd = sum(sum(POST_MEAN)) / N / T;
                case 'row'
                    THETA_upd = sum(POST_MEAN, 2) / T;
                case 'column'
                    THETA_upd = sum(POST_MEAN, 1) / N;
            end

            switch obj.learn_active_var
                case 'scalar'
                    PHI_upd = (1/N/T) * sum(sum(POST_VAR + ...
                        abs(POST_MEAN).^2 - 2*real(conj(obj.active_mean) .* ...
                        POST_MEAN) + abs(obj.active_mean).^2, 1), 2);
                case 'row'
                    PHI_upd = (1/T) * sum(POST_VAR + ...
                        abs(POST_MEAN).^2 - 2*real(conj(obj.active_mean) .* ...
                        POST_MEAN) + abs(obj.active_mean).^2, 2);
                case 'column'
                    PHI_upd = (1/N) * sum(POST_VAR + ...
                        abs(POST_MEAN).^2 - 2*real(conj(obj.active_mean) .* ...
                        POST_MEAN) + abs(obj.active_mean).^2, 1);
            end
        end
        
        
        % *****************************************************************
        %         COMPUTE GAMP-TO-SUPPORT-VARIABLE MESSAGES
        % *****************************************************************
        
        % SUPPORT_OUTPUTS       Compute the messages moving from the GAMP
        % factor nodes to the S variable nodes
        %
        % INPUTS
        %  ETA_IN_OLD   Means of Gaussian messages from mixture component
        %               variables to GAMP factor nodes from last iteration
        %               of GAMP
        %  KAPPA_IN_OLD Variances of Gaussian messages from mixture 
        %               component variables to GAMP factor nodes from last 
        %               iteration of GAMP
        %  RHAT         GAMP variable
        %  RVAR         GAMP variable
        %
        % OUTPUTS
        %  PI_OUT       N-by-T-by-D tensor of outbound messages (from GAMP)
        %               containing active mixture component probabilities
        %
        function PI_OUT = support_outputs(obj, ETA_IN_OLD, KAPPA_IN_OLD, ...
                RHAT, RVAR)
            
            % Extract prior activity probs, means and variances, and add 
            % dummy means and variances (zeros) for indexing simplicity
            [N, T, D] = size(obj.sparsity_rate);
            ETA = cat(3, zeros(N,T), ETA_IN_OLD);
            KAPPA = cat(3, zeros(N,T), KAPPA_IN_OLD);
            RHAT = repmat(RHAT, [1, 1, D]);
            RVAR = repmat(RVAR, [1, 1, D]);
            PI_OUT = NaN(N,T,D);
            
            % Work through each of the D mixture components
            for d = 2:D+1,  % Index is advanced by 1 due to phantom component
                inds = [1:d-1, d+1:D+1];    % Included indices
                
                switch obj.data
                    case 'real'
                        VAR = sqrt((RVAR + repmat(KAPPA(:,:,d), [1, 1, D])) ./ ...
                            (RVAR + KAPPA(:,:,inds)));
                        EXPARG = (abs(RHAT - ETA(:,:,inds)).^2 ./ ...
                            (RVAR + KAPPA(:,:,inds))) - ...
                            (abs(RHAT - repmat(ETA(:,:,d), [1, 1, D])).^2 ./ ...
                            (RVAR + repmat(KAPPA(:,:,d), [1, 1, D])));
                        PI_OUT(:,:,d-1) = sum( VAR .* exp(-(1/2)*EXPARG), 3 );
                    case 'complex'
                        VAR = (RVAR + repmat(KAPPA(:,:,d), [1, 1, D])) ./ ...
                            (RVAR + KAPPA(:,:,inds));
                        EXPARG = (abs(RHAT - ETA(:,:,inds)).^2 ./ ...
                            (RVAR + KAPPA(:,:,inds))) - ...
                            (abs(RHAT - repmat(ETA(:,:,d), [1, 1, D])).^2 ./ ...
                            (RVAR + repmat(KAPPA(:,:,d), [1, 1, D])));
                        PI_OUT(:,:,d-1) = sum( VAR .* exp(-EXPARG), 3 );
                end
                PI_OUT(:,:,d-1) = (1 + PI_OUT(:,:,d-1)).^(-1);
            end
        end
        
        
        % *****************************************************************
        %         TAYLOR APPROX OF OUTGOING GAMP MESSAGES METHOD
        % *****************************************************************
        
        % TAYLOR_APPROX         Method for computing a Taylor series
        % approximation of a single Gaussian message from a GAMP factor
        % node to a particular Gauss mixture amplitude variable node
        %
        % INPUTS
        %  PI_IN        N-by-T-by-D tensor of incoming (to GAMP) mixture
        %               component weights (probabilities) from the last
        %               GAMP iteration
        %  ETA_IN_OLD   Means of Gaussian messages from mixture component
        %               variables to GAMP factor nodes from last iteration
        %               of GAMP
        %  KAPPA_IN_OLD Variances of Gaussian messages from mixture 
        %               component variables to GAMP factor nodes from last 
        %               iteration of GAMP
        %  RHAT         GAMP variable
        %  RVAR         GAMP variable
        %
        % OUTPUTS
        %  ETA_OUT      N-by-T-by-D tensor of outbound messages (from GAMP)
        %               containing means of Gaussian messages
        %  KAPPA_OUT    N-by-T-by-D tensor of outbound messages (from GAMP)
        %               containing variances of Gaussian messages
        %
        function [ETA_OUT, KAPPA_OUT] = taylor_approx(obj, PI_IN, ...
                ETA_IN_OLD, KAPPA_IN_OLD, RHAT, RVAR)
        
            % We will perform D different Taylor series approximations, one
            % for each active Gaussian mixture component
            
            % Extract prior activity probs, means and variances, and add 
            % dummy means and variances (zeros) for indexing simplicity
            [N, T] = size(RHAT);
            D = obj.D;      % # of mixture components
            occurA = 0;   	% Warning flag
            occurB = 0;    	% Warning flag
            PI_IN = cat(3, 1 - sum(PI_IN, 3), PI_IN);
            ETA = cat(3, zeros(N,T), ETA_IN_OLD);
            KAPPA = cat(3, zeros(N,T), KAPPA_IN_OLD);
            KAPPA_OUT = NaN(N,T,D);
            ETA_OUT = NaN(N,T,D);
            RVAR(RVAR == 0) = 1e-5;
            
            for d = 2:D+1,  % Index is advanced by 1 due to phantom component
                % Compute constants for this mixture component
                inds = [1:d-1, d+1:D+1];    % Included indices
                switch obj.data
                    case 'real'
                        A = PI_IN ./ repmat((1 - PI_IN(:,:,d)) + ...
                            obj.eps * PI_IN(:,:,d), [1, 1, D+1]);
                        A(:,:,d) = obj.eps*PI_IN(:,:,d) ./ ((1 - PI_IN(:,:,d)) + ...
                            obj.eps * PI_IN(:,:,d));
                        B = obj.eps^2 / (repmat(RVAR, [1, 1, D+1]) + KAPPA);
                        B(B > realmax^(2/3)) = realmax^(2/3);
                        C = (1/obj.eps) * ((obj.eps - 1) * repmat(RHAT, ...
                            [1, 1, D+1]) + ETA);
                    case 'complex'
                        A = PI_IN ./ repmat((1 - PI_IN(:,:,d)) + ...
                            obj.eps^2 * PI_IN(:,:,d), [1, 1, D+1]);
                        A(:,:,d) = obj.eps^2*PI_IN(:,:,d) ./ ((1 - PI_IN(:,:,d)) + ...
                            obj.eps^2 * PI_IN(:,:,d));
                        B = obj.eps^2 / (repmat(RVAR, [1, 1, D+1]) + KAPPA);
                        B(B > realmax^(2/3)) = realmax^(2/3);
                        C = (1/obj.eps) * ((obj.eps - 1) * repmat(RHAT, ...
                            [1, 1, D+1]) + ETA);
                end
                
                % Build MoG message function, and its derivatives...
                switch obj.data
                    case 'real'
                        G = sum( A(:,:,inds) .* B(:,:,inds).^(1/2) .* ...
                            exp(-(1/2) * B(:,:,inds) .* C(:,:,inds).^2), 3 ) + ...
                            ( A(:,:,d) .* RVAR.^(-1/2) );

                        dG = -sum( A(:,:,inds) .* B(:,:,inds).^(3/2) .* ...
                            C(:,:,inds) .* exp(-(1/2)*B(:,:,inds) .* ...
                            C(:,:,inds).^2), 3 );

                        d2G = -sum( A(:,:,inds) .* B(:,:,inds).^(3/2) .* ...
                            exp(-(1/2) * B(:,:,inds) .* C(:,:,inds).^2) .* (1 - ...
                            B(:,:,inds) .* C(:,:,inds).^2), 3 ) - ...
                            ( A(:,:,d) .* RVAR.^(-3/2) );
                    case 'complex'
                        G = sum( A(:,:,inds) .* B(:,:,inds) .* ...
                            exp(-B(:,:,inds) .* abs(C(:,:,inds)).^2), 3 ) + ...
                            ( A(:,:,d) .* RVAR.^(-1) );

                        dG = -sum( A(:,:,inds) .* B(:,:,inds).^(2) .* ...
                            C(:,:,inds) .* exp(-B(:,:,inds) .* ...
                            abs(C(:,:,inds)).^2), 3 );

                        d2G = -sum( A(:,:,inds) .* B(:,:,inds).^(2) .* ...
                            exp(-B(:,:,inds) .* abs(C(:,:,inds)).^2) .* (1 - ...
                            B(:,:,inds) .* abs(C(:,:,inds)).^2), 3 ) - ...
                            ( A(:,:,d) .* RVAR.^(-2) );
                end
                
                if any(G(:) < sqrt(realmin)) || any(G(:) > sqrt(realmax)) || ...
                        any(abs(dG(:)) > sqrt(realmax)) || ...
                        any(abs(d2G(:)) > sqrt(realmax))
                    if ~occurA      % Suppress repeat warnings
                        fprintf('Taylor approx numerical precision probs: G, dG, d2G\n');
                    end
                    G(G > sqrt(realmax)) = sqrt(realmax);
                    G(G < sqrt(realmin)) = sqrt(realmin);
                    dG(abs(dG) > sqrt(realmax)) = ...
                        sign(dG(abs(dG) > sqrt(realmax))) * sqrt(realmax);
                    d2G(abs(d2G) > sqrt(realmax)) = ...
                        sign(d2G(abs(d2G) > sqrt(realmax))) * sqrt(realmax);
                end
                
                % Use G and its derivatives to compute the derivatives of
                % F = -log(G) 
                dF = -dG ./ G;
                
                d2F = (dG.^2 - d2G.*G) ./ G.^2;
                
                if any(isnan(dF(:))) || any(isnan(d2F(:))) || ...
                        any(d2F(:) <= 0)
                    if ~occurB      % Suppress repeat warnings
                        fprintf('Taylor approx numerical precision probs: F\n');
                    end
                    d2F(d2F <= 0) = realmin;
                end
                
                % Compute outgoing (from GAMP) message variables...
                switch obj.data
                    case 'real'
                        KAPPA_OUT(:,:,d-1) = d2F.^(-1);
                        ETA_OUT(:,:,d-1) = RHAT - KAPPA_OUT(:,:,d-1).*dF;
                    case 'complex'
                        KAPPA_OUT(:,:,d-1) = 2*(d2F.^(-1));
                        % For numerical reasons, KAPPA_OUT might appear to 
                        % be complex. Force to be real, but run a sanity 
                        % check first.
                        if norm(imag(KAPPA_OUT(:))) > 1e-1
                            warning('Non-negligible imaginary variances encountered')
                        end
                        KAPPA_OUT = real(KAPPA_OUT);
                        ETA_OUT(:,:,d-1) = RHAT - 1/2*KAPPA_OUT(:,:,d-1).*dF;
                end
            end
            
            if any(KAPPA_OUT(:) < 0)
                fprintf('Taylor approx: negative variance\n')
            elseif any(isnan(KAPPA_OUT(:)))
                fprintf('Taylor approx: NaNs in KAPPA_OUT\n')
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
                string = sprintf('%d-by-%d-by-%d array (Min: %g, Max: %g)', ...
                    size(prop, 1), size(prop, 2), size(prop, 3), min(min(min(prop))), ...
                    max(max(max(prop))));
            end
        end
    end % Private methods
   
end % classdef