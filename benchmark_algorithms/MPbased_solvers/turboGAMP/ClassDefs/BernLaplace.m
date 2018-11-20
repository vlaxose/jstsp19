% CLASS: BernLaplace
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Signal
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The BernLaplace class contains the parameters needed to define a
%   Bernoulli-Laplacian marginal prior distribution for each signal
%   coefficient.  Specifically, each signal coefficient, X(n,t), has a
%   marginal prior distribution
%   pdf(X(n,t)) = (1 - PI(n,t))*delta(X(n,t)) + 
%                 PI(n,t)*Laplace(X(n,t); LAMBDA(n,t))
%   where n = 1,...,N indexes the row of the signal matrix X, 
%   t = 1,...,T indexes the column (timestep) of the signal matrix X,
%   delta(.) is the Dirac delta function, and Laplace(.; lambda) is a
%   real-valued zero-mean Laplacian distribution with rate lambda (lambda
%   is defined as the parameter that yields a variance of 2/lambda^2).  
%   Note that this class can be used in the single measurement vector (SMV)
%   problem as well, (i.e., T = 1).
%
%   The Bernoulli-Laplacian prior can be viewed as the marginal
%   distribution of X(n,t) when X(n,t) can be represented as the product of
%   two hidden variables: a Bernoulli-distributed binary (0,1) varianble, 
%   S(n,t), with Pr{S(n,t) = 1} = PI(n,t), and a Laplacian random variable, 
%   GAMMA(n,t), with rate LAMBDA(n,t).  Together these hidden variables 
%   define X(n,t): X(n,t) = S(n,t) * GAMMA(n,t).
%
%   To create a BernLaplace object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   BernLaplace(), will create a BernLaplace object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters, 
%   with the remaining parameters initialized to their default values, by 
%   using MATLAB's property/value string pairs convention, e.g.,
%   BernLaplace('sparsity_rate', 0.05, 'laplace_rate', 3) will
%   construct a BernGauss object in which PI(n,t) = 0.05 for all n,t,
%   and LAMBDA(n,t) = 3 for all n,t.  Remaining parameters will be set to 
%   their default values.  Note that the parameters PI and LAMBDA can be 
%   initialized as either scalars, length-N row vectors, length-T column 
%   vectors, or N-by-T matrices of values, to allow for distributions that 
%   are i.i.d., temporally uniform, spatially uniform, or completely 
%   independent, respectively.
%
%   Additionally, this class contains information about which
%   parameters should be learned from the data, and in what manner, (see
%   PROPERTIES below).  As an example, to prevent the parameter
%   sparsity_rate (PI) from being learned by an EM algorithm, set
%   learn_sparsity_rate = 'false'.  To learn a single, common sparsity rate
%   for all elements of the signal matrix, X, set learn_sparsity_rate =
%   'scalar'.  Likewise, to learn a unique sparsity rate for each row
%   (column) of X, set learn_sparsity_rate = 'row' ('column').
%
%   *** BernLaplace only supports sum-product GAMP, and not max-sum GAMP, 
%   and thus should only be paired with other turboGAMP objects that 
%   support sum-product message passing. ***
%
% PROPERTIES (State variables)
%   sparsity_rate           The prior sparsity rate(s), PI [Default: 0.05]
%   learn_sparsity_rate     Learn sparsity rate using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'false']
%   laplace_rate            The Laplacian distribution rate parameter,
%                           LAMBDA [Default: sqrt(2)]
%   learn_laplace_rate      Learn Laplacian rate using EM algorithm?  (See
%                           DESCRIPTION for options)  [Default: 'false']
%
% METHODS (Subroutines/functions)
%   BernLaplace()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   BernLaplace('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class BernLaplace, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'BL' when obj is a BernLaplace 
%         object
%   BernLaplaceCopyObj = copy(obj)
%       - Create an independent copy of the BernLaplace object, obj.  Note
%         that this method is important if one wishes to create an
%         independent copy of the BernLaplace object, obj, since simply
%         using "CopyObj = obj;" will *not* create an independent copy, but
%         will point to the same underlying object.
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
%   EstimIn = InitPriors(TBobj)
%      	- Provides an initial EstimIn object for use by GAMP the first
%         time. TBobj is a TurboOpt object.  If TBobj.commonA is 
%         false, then this method returns a 1-by-T cell array of EstimIn 
%         objects. [Hidden method]
%   LAMBDA_upd = obj.LearnLambda(obj, TBobj)
%       - In the absence of any form of amplitude structure, this method
%         will perform EM parameter learning of the Laplacian rate 
%         parameter, LAMBDA. [Hidden method]
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
%       - Created (10/23/12; JAZ)
%       - Added EMreport method implementation (01/28/13; JAZ)
% Version 0.2
%

classdef BernLaplace < Signal

    properties
        % Bernoulli-Gaussian prior family properties
        sparsity_rate = 0.05;
        learn_sparsity_rate = 'false';
        laplace_rate = sqrt(2);
        learn_laplace_rate = 'false';
        data = 'real';
        version = 'mmse';
    end % properties
       
    properties (Constant, Hidden)
        type = 'BL';        % Bernoulli-Gaussian type identifier
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
        function obj = BernLaplace(varargin)
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
        % Set method for sparsity_rate (PI)
        function obj = set.sparsity_rate(obj, PI)
           if any(PI(:)) < 0 || any(PI(:)) > 1
              error('sparsity_rate must be in the interval [0,1]')
           else
              obj.sparsity_rate = PI;
           end
        end
        
        % Set method for active_var (PHI)
        function obj = set.laplace_rate(obj, LAMBDA)
           if any(LAMBDA(:)) < 0
              error('active_var must be non-negative')
           else
              obj.laplace_rate = LAMBDA;
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
        function obj = set.learn_laplace_rate(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_active_mean')
            end
            obj.learn_laplace_rate = lower(string);
        end
        
        % Set method for data
        function obj = set.data(obj, string)
            if ~strcmpi(string, {'real', 'complex'})
                error('Invalid option: data')
            elseif strcmpi(string, 'complex')
                error('BernLaplace does not currently support complex data')
            end
            obj.data = lower(string);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('BernLaplace does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SIGNAL PRIOR: Bernoulli-Laplacian\n')
            fprintf('      sparsity_rate: %s\n', ...
                form(obj, obj.sparsity_rate))
            fprintf('       laplace_rate: %s\n', ...
                form(obj, obj.laplace_rate))
            fprintf('learn_sparsity_rate: %s\n', obj.learn_sparsity_rate)
            fprintf(' learn_laplace_rate: %s\n', obj.learn_laplace_rate)
            fprintf('               data: %s\n', obj.data)
            fprintf('       GAMP version: %s\n', obj.version)
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        % Create an indepedent copy of a BernGauss object
        function BernLaplaceCopyObj = copy(obj)
            BernLaplaceCopyObj = BernLaplace('sparsity_rate', ...
                obj.sparsity_rate, 'learn_sparsity_rate', ...
                obj.learn_sparsity_rate, 'laplace_rate', obj.laplace_rate, ...
                'learn_laplace_rate', obj.learn_laplace_rate, 'data', obj.data);
        end
        
        
        % *****************************************************************
        %                      ACCESSORY METHODS
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
            [N, T] = size(RHAT);
            RVAR = max(1e-14, RVAR);
            
            % Unpack inputs given to GAMP at the last iteration
            switch TBobj.commonA
                case true
                    PI_IN_OLD = EstimInOld.p1;
                    LAMBDA_IN_OLD = EstimInOld.estim1.lambda;
                case false
                    PI_IN_OLD = NaN(N,T);  LAMBDA_IN_OLD = NaN(N,T);
                    for t = 1:T
                        PI_IN_OLD(:,t) = EstimInOld{t}.p1;
                        LAMBDA_IN_OLD(:,t) = EstimInOld{t}.estim1.lambda;
                    end
            end
            
            % Use outputs from GAMP to compute the messages moving
            % from the GAMP factor nodes to the support nodes, S
            sig = sqrt(RVAR);                           % Gaussian std dev
            muL = RHAT + LAMBDA_IN_OLD.*RVAR;          	% Lower integral mean
            muU = RHAT - LAMBDA_IN_OLD.*RVAR;          	% Upper integral mean
            muL_over_sig = muL ./ sig;
            muU_over_sig = muU ./ sig;
            cdfL = normcdf(-muL_over_sig);              % Lower cdf
            cdfU = normcdf(muU_over_sig);               % Upper cdf
%             NormConL = LAMBDA_IN_OLD/2 .* ...           % Mass of lower integral
%                 exp( (muL.^2 - RHAT.^2) ./ (2*RVAR) ) .* cdfL;
%             NormConU = LAMBDA_IN_OLD/2 .* ...           % Mass of upper integral
%                 exp( (muU.^2 - RHAT.^2) ./ (2*RVAR) ) .* cdfU;
            
            TMP = LAMBDA_IN_OLD/2 .* sqrt(2*pi*RVAR) .* ...
                (min(realmax, exp((muL.^2)./(2*RVAR))) .* cdfL + ...
                min(realmax, exp((muU.^2)./(2*RVAR))) .* cdfU);
            TMP = min(realmax, TMP);    % Clip extreme values
            PI_OUT = TMP ./ (1 + TMP);
            
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
                    
                    % If the user has requested EM refinement of PI, do
                    % that now.
                    if obj.EMcnt >= 1 && mod(obj.EMcnt, 2) == 0   % Delay learning for 1 turbo iter
                        switch TBobj.Signal.learn_sparsity_rate
                            case 'scalar'
                                % Update a single scalar
                                pi_upd = sum(sum(S_POST)) / N / T;
                            case 'row'
                                % Update different lambda for each row
                                pi_upd = sum(S_POST, 2) / T;
                            case 'column'
                                % Update different lambda for each column
                                pi_upd = sum(S_POST, 1) / N;
                            case 'false'
                                % Do not update the prior
                                pi_upd = obj.sparsity_rate;
                        end
                        obj.sparsity_rate = TBobj.resize(pi_upd, N, T);
                    end
                    
                otherwise
                    % Call the UpdateSupport method for the appropriate
                    % model of support structure.  EM learning of sparsity
                    % rate will be handled by the UpdateSupport method
                    [PI_IN, S_POST] = ...
                        TBobj.SupportStruct.UpdateSupport(TBobj, PI_OUT);
            end
            
            % Use the outputs from GAMP to compute the messages moving from
            % the GAMP factor nodes to the amplitude nodes, GAMMA.  These
            % messages are Gaussian, with mean ETA_OUT and variance
            % KAPPA_OUT.  To compute these, we require a Taylor series
            % approximation which relies on outgoing GAMP messages and
            % incoming (to GAMP) support probability messages.
            [ETA_OUT, KAPPA_OUT] = taylor_approx(obj, PI_IN_OLD, RHAT, RVAR);
            
            % Now let's determine the values of the Laplacian rate 
            % parameters, LAMBDA.  We need to look at the type of structure
            % present in the signal amplitude matrix, GAMMA.  EM updates of
            % the Bernoulli-Laplacian model parameters occurs here as well.
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'                    
                    % No amplitude structure, thus return priors as
                    % GAMP-bound messages
                    LAMBDA_IN = obj.laplace_rate;
                    
                    % To avoid numerical problems (0/0) when evaluating
                    % ratios of Gaussian CDFs, impose a firm cap on the
                    % maximum value of entries of KAPPA_OUT
                    KAPPA_OUT = min(KAPPA_OUT, 700);
                    
                    % *********************************************************
                    % Begin by computing various constants on which the
                    % posterior mean and variance depend
                    sig = sqrt(KAPPA_OUT);                   	% Gaussian prod std dev
                    muL = ETA_OUT + LAMBDA_IN_OLD.*KAPPA_OUT; 	% Lower integral mean
                    muU = ETA_OUT - LAMBDA_IN_OLD.*KAPPA_OUT; 	% Upper integral mean
                    muL_over_sig = muL ./ sig;
                    muU_over_sig = muU ./ sig;
                    cdfL = normcdf(-muL_over_sig);              % Lower cdf
                    cdfU = normcdf(muU_over_sig);               % Upper cdf
                    cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
                    SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*KAPPA_OUT) ) .* ...
                        cdfRatio;
                    NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants
                    
                    % For the "trouble" constants (those with muL's and muU's
                    % that are too large to give accurate numerical answers),
                    % we will effectively peg the special constant to be Inf or
                    % 0 based on whether muL dominates muU or vice-versa
                    SpecialConstant(NaN_Idx & (-muL > muU)) = Inf;
                    SpecialConstant(NaN_Idx & (-muL < muU)) = 0;
                    
                    % Compute the ratio normpdf(a)/normcdf(a) for
                    % appropriate upper- and lower-integral constants, a
                    RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
                    RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));
                    
                    % Now compute the first posterior moment...
                    POST_MEAN = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                        (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                        (muU + sig.*RatioU);
                    
                    % ...and second central posterior moment
                    varL = KAPPA_OUT .* (1 - RatioL.*(RatioL - muL_over_sig));
                    varU = KAPPA_OUT .* (1 - RatioU.*(RatioU + muU_over_sig));
                    meanL = muL - sig.*RatioL;
                    meanU = muU + sig.*RatioU;
                    SecondMoment = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                        (varL + meanL.^2) + (1 ./ (1 + SpecialConstant)) .* ...
                        (varU + meanU.^2);
                    POST_VAR = SecondMoment - POST_MEAN.^2;
                    % *********************************************************
                
                    
                    % Call the EM update method built into this class file
                    obj.LearnLambda(TBobj, POST_MEAN, POST_VAR);
                    
                otherwise
                    % Amplitude structure not presently supported
                    error(['BernLaplace Signal class does not currently' ...
                        ' support structured amplitude classes'])
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
                    EstimIn = SoftThreshEstimIn(LAMBDA_IN, ...
                        strcmp(obj.version, 'map'));
                    EstimIn = SparseScaEstim(EstimIn, PI_IN);
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        EstimIn{t} = SoftThreshEstimIn(LAMBDA_IN(:,t), ...
                            strcmp(obj.version, 'map'));
                        EstimIn{t} = SparseScaEstim(EstimIn{t}, PI_IN(:,t));
                    end
            end
            
            % Increment the EM iteration counter
            obj.EMcnt = obj.EMcnt + 1;
        end
        
        
        % *****************************************************************
        %         	   INITIALIZE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        % Initialize EstimIn object for a Bernoulli-Laplacian signal prior
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
            
            % Initialize parameters
            obj.sparsity_rate = TBobj.resize(obj.sparsity_rate, N, T);
            obj.laplace_rate = TBobj.resize(obj.laplace_rate, N, T);
            
            % Form initial EstimIn object
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    EstimIn = SoftThreshEstimIn(obj.laplace_rate, ...
                        strcmp(obj.version, 'map'));
                    EstimIn = SparseScaEstim(EstimIn, obj.sparsity_rate);
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        EstimIn{t} = SoftThreshEstimIn(obj.laplace_rate(:,t), ...
                            strcmp(obj.version, 'map'));
                        EstimIn{t} = SparseScaEstim(EstimIn{t}, ...
                            obj.sparsity_rate(:,t));
                    end
            end
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the
        % Bernoulli-Laplacian prior on X
        %
        % INPUTS:
        % obj       	An object of the BernLaplace class
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
                    S_TRUE = rand(N,T) < TBobj.resize(obj.sparsity_rate, N, T);
                otherwise
                    % Call the genRand method of the particular form of
                    % support structure to produce S_TRUE
                    S_TRUE = genRand(TBobj.SupportStruct, TBobj, GenParams);
            end
            
            % Now produce a realization of GAMMA
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % No amplitude structure, so draw iid
                    LAMBDA = TBobj.resize(obj.laplace_rate, N, T);
                    E1 = exprnd(1./LAMBDA);
                    E2 = exprnd(1./LAMBDA);
                    GAMMA_TRUE = E2 - E1;
                otherwise
                    % Call the genRand method of the particular form of
                    % amplitude structure to produce GAMMA_TRUE
                    GAMMA_TRUE = genRand(TBobj.AmplitudeStruct, TBobj, ...
                        GenParams);
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
                ~strcmpi(obj.learn_laplace_rate, 'false');
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'sparsity_rate',    'Sparsity rate',	'learn_sparsity_rate'; 
                        'laplace_rate',     'Laplacian rate',   'learn_laplace_rate'};
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
        
        function LearnLambda(obj, TBobj, POST_MEAN, POST_VAR)
            % Learn amplitude parameters in the case where amplitude
            % variables are independent of one another

            [N, T] = size(obj.laplace_rate);
            
            % Initialize updates to old values, in case no EM learning is
            % to take place
            LAMBDA_upd = obj.laplace_rate;
            
            % Compute mean of |gamma_n|
            sig = sqrt(POST_VAR);
            MU = 2*sig .* normpdf(POST_MEAN ./ sig) + POST_MEAN .* (1 - ...
                2*normcdf(-POST_MEAN ./ sig));
            
            if obj.EMcnt > 1 && mod(obj.EMcnt, 2) == 1
                switch obj.learn_laplace_rate
                    case 'scalar'
                        LAMBDA_upd = N*T / sum(MU(:));
                    case 'row'
                        LAMBDA_upd = T ./ sum(MU, 2);
                    case 'column'
                        LAMBDA_upd = N ./ sum(MU, 1);
                end
            end
            obj.laplace_rate = TBobj.resize(LAMBDA_upd, N, T);
            
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