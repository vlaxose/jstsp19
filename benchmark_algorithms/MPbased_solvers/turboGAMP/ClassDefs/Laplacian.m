% CLASS: Laplacian
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Signal
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The Laplacian class contains the parameters needed to define a
%   Laplacian marginal prior distribution for each signal coefficient.  
%   Specifically, each signal coefficient, X(n,t), has a marginal prior 
%   distribution:
%       pdf(X(n,t)) = (lambda(n,t)/2) exp(-lambda(n,t) * |X(n,t)|)
%   where n = 1,...,N indexes the row of the signal matrix X, 
%   t = 1,...,T indexes the column (timestep) of the signal matrix X.  The
%   Laplacian distribution has mean 0 and variance 2/lambda^2.  Note that 
%   this class can be used in the single measurement vector (SMV) problem 
%   as well, (i.e., T = 1).
%
%   To create a Laplacian object, there are two constructors to choose from
%   (see METHODS section below).  The default constructor, Laplacian(), 
%   will create a Laplacian object initialized with all default values for 
%   each parameter/property.  The alternative constructor allows the user 
%   to initialize any subset of the parameters, with the remaining 
%   parameters initialized to their default values, by using MATLAB's 
%   property/value string pairs convention, e.g., Laplacian('lambda', 2) 
%   will construct a Laplacian object in which lambda(n,t) = 0.05 for all 
%   n,t.
%
%   Laplacian supports both sum-product and max-sum GAMP, to allow for
%   MMSE and MAP estimation, respectively.  The property "version" dictates
%   which version of message updates to use.
%
%   If desired, an expectation-maximization (EM) learning algorithm can be
%   used to automatically learn the value of lambda from the data by 
%   setting learn_lambda appropriately (if running sum-product GAMP.
%   Otherwise, an approximate EM-for-MAP scheme is used in the max-sum
%   case).
%
% PROPERTIES (State variables)
%   lambda          Either a scalar, or an N-by-T matrix, with positive 
%                   entries [Default: sqrt(2)]
%   learn_lambda    Learn value of lambda through EM learning procedure
%                   (true) or not (false)? [Default: false]
%   version         Apply sum-product ('mmse') or max-sum ('map') GAMP 
%                   message passing? [Default: 'map']
%
% METHODS (Subroutines/functions)
%   Laplacian()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   Laplacian('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class Laplacian, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'Lap' when obj is a Laplacian object
%   LaplaceCopyObj = copy(obj)
%       - Create an independent copy of the Laplacian object, obj.
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
%         (e.g., 'lambda'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Laplacian rate: '), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (09/18/12; JAZ)
%       - Added max-sum GAMP support and EM learning (10/15/12; JAZ)
%       - Added EMreport method implementation (01/28/13; JAZ)
%       - Modified EM learning update based on more accurate closed-form
%         GAMP posterior (06/11/13; JAZ)
% Version 0.2
%

classdef Laplacian < Signal

    properties
        % Laplacian prior family properties
        lambda = sqrt(2);       % Ell-1 penalty
        learn_lambda = false;   % Don't learn lambda using EM
        version = 'map';        % Max-sum GAMP updates
        data = 'real';          % Works only for real-valued signals
    end % properties
       
    properties (Constant, Hidden)
        type = 'Lap';        % Laplacian type identifier
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = Laplacian(varargin)
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
        
        % Set method for variance (lambda)
        function obj = set.lambda(obj, lambda)
            if any(lambda(:)) <= 0
                error('lambda must be non-negative')
            else
                obj.lambda = lambda;
            end
        end
        
        % Set method for lambda1 EM learning (learn_lambda)
        function obj = set.learn_lambda(obj, val)
            obj.learn_lambda = logical(val);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if any(strcmpi(version, {'mmse', 'map'}))
                obj.version = lower(version);
            else
                error('Invalid option: version')
            end
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SIGNAL PRIOR: Laplacian\n')
            fprintf('            lambda: %s\n', form(obj, obj.lambda))
            if obj.learn_lambda
                fprintf('lambda EM learning: true\n')
            else
                fprintf('lambda EM learning: false\n')
            end
            fprintf('      GAMP version: %s\n', obj.version)
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Create an indepedent copy of a BernGauss object
        function LaplacianCopyObj = copy(obj)
            LaplacianCopyObj = Laplacian('lambda', obj.lambda, ...
                'learn_lambda', obj.learn_lambda, 'version', ...
                obj.version);
        end
        
        
        % *****************************************************************
        %                       ACCESSORY METHOD
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
        
     	% Update the EstimIn object for a Laplacian prior on X
        function [EstimIn, S_POST] = UpdatePriors(obj, TBobj, ...
                GAMPState, ~)
            
            [Xhat, Xvar, Rhat, Rvar] = GAMPState.getState();
            [N, T] = size(Xhat);
            
            % Perform (scalar) parameter learning if desired
            if obj.learn_lambda && strcmpi(obj.version, 'mmse')
                % Compute the EM update of lambda
                
                % To avoid numerical problems (0/0) when evaluating
                % ratios of Gaussian CDFs, impose a firm cap on the
                % maximum value of entries of rvar
                Rvar = min(Rvar, 700);
                
                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean, E[|x_n| | y] depends
                sig = sqrt(Rvar);                       	% Gaussian prod std dev
                muL = Rhat + obj.lambda.*Rvar;          	% Lower integral mean
                muU = Rhat - obj.lambda.*Rvar;          	% Upper integral mean
                muL_over_sig = muL ./ sig;
                muU_over_sig = muU ./ sig;
                cdfL = normcdf(-muL_over_sig);              % Lower cdf
                cdfU = normcdf(muU_over_sig);               % Upper cdf
                cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
                SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*Rvar) ) .* ...
                    cdfRatio;
                NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants
                
                % For the "trouble" constants (those with muL's and muU's
                % that are too large to give accurate numerical answers),
                % we will effectively peg the special constant to be Inf or
                % 0 based on whether muL dominates muU or vice-versa
                SpecialConstant(NaN_Idx & (-muL >= muU)) = Inf;
                SpecialConstant(NaN_Idx & (-muL < muU)) = 0;
                
                % Compute the ratio normpdf(a)/normcdf(a) for
                % appropriate upper- and lower-integral constants, a
                RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
                RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));
                
                % Now compute E[|x_n| | y]...
                mu = (1 ./ (1 + SpecialConstant)) .* (muU + sig.*RatioU) ...
                    - (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                    (muL - sig.*RatioL);
                
                % Compute lambda EM update
                lam_upd = N*T / sum(mu(:));
                
                if lam_upd <= 0 || isnan(lam_upd)
                    warning('EM-for-MAP update of lambda was negative or NaN...ignoring')
                    lam_upd = mean(obj.lambda(:));
                end
                obj.lambda = TBobj.resize(lam_upd, N, T);
            elseif obj.learn_lambda && strcmpi(obj.version, 'map')
                % Compute the EM-for-MAP approximate update of lambda
                
                % Pass final MAP-GAMP messages to an MMSE-GAMP configured
                % EstimIn class, in order to obtain approximate posterior
                % marginal means and variances
                EstimIn = SoftThreshEstimIn(obj.lambda, false);
                [Xhat2, Xvar2] = EstimIn.estim(Rhat, Rvar);
                
                % Using the (approximate) posterior means and variance,
                % compute the traditional EM estimate of lambda
                sig = sqrt(Xvar2);
                mu = 2*sig .* normpdf(Xhat2 ./ sig) + Xhat2 .* (1 - ...
                    2*normcdf(-Xhat2 ./ sig));   % E[|x| | y]
                lam_upd = N*T / sum(mu(:));
                
                if lam_upd <= 0 || isnan(lam_upd)
                    warning('EM-for-MAP update of lambda was negative or NaN...ignoring')
                    lam_upd = mean(obj.lambda(:));
                end
                obj.lambda = TBobj.resize(lam_upd, N, T);
            end
            
            % Return a binary hard estimate of S_POST
            S_POST = double(Xhat ~= 0);
            
            % Prepare next round of EstimIn objects for GAMP
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    EstimIn = SoftThreshEstimIn(obj.lambda, ...
                        strcmpi('map', obj.version));
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        EstimIn{t} = SoftThreshEstimIn(obj.lambda, ...
                            strcmpi('map', obj.version));
                    end
            end
        end
        
        
        % *****************************************************************
        %         	   INITIALIZE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        
        % Initialize EstimIn object for a Laplacian signal prior
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
            
            obj.lambda = TBobj.resize(obj.lambda, N, T);
                        
            % Prepare initial round of EstimIn objects for GAMP
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    EstimIn = SoftThreshEstimIn(obj.lambda, ...
                        strcmpi('map', obj.version));
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        EstimIn{t} = SoftThreshEstimIn(obj.lambda, ...
                            strcmpi('map', obj.version));
                    end
            end
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the
        % Laplacian prior on X
        %
        % INPUTS:
        % obj       	An object of the Laplacian class
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
                    % No support structure, so generate non-sparse X
                    S_TRUE = ones(N,T);
                otherwise
                    % Call the genRand method of the particular form of
                    % support structure to produce S_TRUE
                    SuppStruct = TBobj.SupportStruct;
                    S_TRUE = SuppStruct.genRand(TBobj, GenParams);
                    warning('SupportStruct may violate Laplacian stats')
            end
            
            % Now produce a realization of GAMMA
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % No amplitude structure, so draw iid
                    lam = TBobj.resize(obj.lambda, N, T);
                    E1 = exprnd(1./lam);
                    E2 = exprnd(1./lam);
                    GAMMA_TRUE = E2 - E1;
                otherwise
                    % Call the genRand method of the particular form of
                    % amplitude structure to produce GAMMA_TRUE
                    AmpStruct = TBobj.AmplitudeStruct;
                    GAMMA_TRUE = AmpStruct.genRand(TBobj, GenParams);
                    warning('AmplitudeStruct may violate Laplacian stats')
            end
            
            % Combine S_TRUE and GAMMA_TRUE to yield X_TRUE
            X_TRUE = S_TRUE .* GAMMA_TRUE;
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = double(obj.learn_lambda);
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {'lambda',	'Laplacian rate',	'learn_lambda'};
            % Iterate through each parameter, adding to Report array as
            % needed
			j = 0;
            for i = 1:size(Params, 1)
                switch obj.(Params{i,3})
                    case true
                        j = j + 1;
                        Report{j,1} = Params{i,1};
                        Report{j,2} = Params{i,2};
                        Report{j,3} = obj.(Params{i,1})(1,1);
                    otherwise
                        % Don't add this parameter to the report
                end
            end
        end
    end
    
    methods (Access = private)        
        
        % *****************************************************************
        %                          HELPER METHODS
        % *****************************************************************
        
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