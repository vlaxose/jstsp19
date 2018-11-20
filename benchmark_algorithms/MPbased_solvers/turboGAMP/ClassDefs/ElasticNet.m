% CLASS: ElasticNet
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Signal
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The ElasticNet class contains the parameters needed to define an
%   elastic net marginal prior distribution for each signal coefficient.  
%   Specifically, each signal coefficient, X(n,t), has a marginal prior 
%   distribution:
%    p(X(n,t)) \propto N(X(n,t); 0, (2*lambda2)^-1)*Lap(X(n,t); lambda1),
%   where N(X(n,t); 0, a) is a normal distribution with variance a, and
%   Lap(X(n,t); b) is a Laplacian distribution with variance 2/b^2.  For
%   the purposes of MAP estimation, the elastic net prior distribution
%   corresponds to a penalty function on the regressors of the form
%           f(X) = lambda1*norm(X(:), 1) + lambda2*norm(X(:), 2).
%
%   n = 1,...,N indexes the row of the signal matrix X, t = 1,...,T indexes 
%   the column (timestep) of the signal matrix X.  Note that this class can 
%   be used in the single measurement vector (SMV) problem as well, 
%   (i.e., T = 1).
%
%   If desired, an expectation-maximization (EM) learning algorithm can be
%   used to automatically learn the values of lambda1 and lambda2 from the
%   data by setting learn_lambda1 and learn_lambda2 appropriately.
%
%   ElasticNet supports both sum-product and max-sum GAMP, to allow for
%   MMSE and MAP estimation, respectively.  The property "version" dictates
%   which version of message updates to use.
%
%   To create an ElasticNet object, there are several constructors to 
%   choose from (see METHODS section below).  The default constructor, 
%   ElasticNet(), will create an ElasticNet object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructors allow the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values.
%
% PROPERTIES (State variables)
%   lambda1         Either a scalar, or an N-by-T matrix, with positive 
%                   entries [Default: sqrt(2)]
%   lambda2         Either a scalar, or an N-by-T matrix, with non-negative
%                   entries [Default: 1/2]
%   learn_lambda1   Learn value of lambda1 through EM learning procedure
%                   (true) or not (false)? [Default: false]
%   learn_lambda2   Learn value of lambda2 through EM learning procedure
%                   (true) or not (false)? [Default: false]
%   version         Apply sum-product ('mmse') or max-sum ('map') GAMP 
%                   message passing? [Default: 'map']
%
% METHODS (Subroutines/functions)
%   ElasticNet()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   ElasticNet(lambda1, lambda2, learn_lambda1, learn_lambda2, version)
%       - Full constructor.  Pass empty ([]) argument to any subset of
%         parameters to leave them set at default values.
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class ElasticNet, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'Elastic' when obj is an ElasticNet
%         object
%   ElasticNetCopyObj = copy(obj)
%       - Create an independent copy of the ElasticNet object, obj.
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
%   genRand(TBobj, GenParams)
%       - Not implemented [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'lambda1'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Ell-1 penalty'), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (10/11/12; JAZ)
%       - Added EM learning (10/15/12; JAZ)
%       - Added EMreport method implementation (01/28/13; JAZ)
% Version 0.2
%

classdef ElasticNet < Signal

    properties
        % Elastic net prior properties
        lambda1 = sqrt(2);  % ell-1 norm penalty
        lambda2 = 1/2;      % ell-2 norm penalty
        learn_lambda1 = false;  % Don't learn lambda1 using EM
        learn_lambda2 = false;  % Don't learn lambda2 using EM
        version = 'map';    % Max-sum GAMP updates
        data = 'real';      % Works only for real-valued signals
    end % properties
       
    properties (Constant, Hidden)
        type = 'Elastic';        % ElasticNet type identifier
    end
    
    properties (Hidden, Access = private)
        EMcnt = 0;  % Counter for EM algorithm purposes
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = ElasticNet(lambda1, lambda2, learn_lambda1, ...
                learn_lambda2, version)
            if nargin >= 1 && ~isempty(lambda1)
                obj.lambda1 = lambda1;
            end
            if nargin >= 2 && ~isempty(lambda2)
                obj.lambda2 = lambda2;
            end
            if nargin >= 3 && ~isempty(learn_lambda1)
                obj.learn_lambda1 = learn_lambda1;
            end
            if nargin >= 4 && ~isempty(learn_lambda2)
                obj.learn_lambda2 = learn_lambda2;
            end
            if nargin >= 5 && ~isempty(version)
                obj.version = version;
            end
        end                    
        
        
        % *****************************************************************
        %                         SET METHODS
        % *****************************************************************
        
        % Set method for ell-1 penalty (lambda1)
        function obj = set.lambda1(obj, lambda1)
           if any(lambda1(:)) <= 0
              error('lambda1 must be positive')
           else
              obj.lambda1 = lambda1;
           end
        end
        
        % Set method for ell-2 penalty (lambda1)
        function obj = set.lambda2(obj, lambda2)
           if any(lambda2(:)) < 0
              error('lambda2 must be positive')
           else
              obj.lambda2 = lambda2;
           end
        end
        
        % Set method for lambda1 EM learning (learn_lambda1)
        function obj = set.learn_lambda1(obj, val)
            obj.learn_lambda1 = logical(val);
        end
        
        % Set method for lambda2 EM learning (learn_lambda2)
        function obj = set.learn_lambda2(obj, val)
            obj.learn_lambda2 = logical(val);
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
            fprintf('SIGNAL PRIOR: ElasticNet\n')
            fprintf('            lambda1: %s\n', form(obj, obj.lambda1))
            fprintf('            lambda2: %s\n', form(obj, obj.lambda2))
            if obj.learn_lambda1
                fprintf('lambda1 EM learning: true\n')
            else
                fprintf('lambda1 EM learning: false\n')
            end
            if obj.learn_lambda2
                fprintf('lambda2 EM learning: true\n')
            else
                fprintf('lambda2 EM learning: false\n')
            end
            fprintf('       GAMP version: %s\n', obj.version)
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Create an indepedent copy of a BernGauss object
        function ElasticNetCopyObj = copy(obj)
            ElasticNetCopyObj = ElasticNet(obj.lambda1, obj.lambda2, ...
                obj.learn_lambda1, obj.learn_lambda2, obj.version);
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
        
     	% Update the EstimIn object for an ElasticNet prior on X
        function [EstimIn, S_POST] = UpdatePriors(obj, TBobj, ...
                GAMPState, ~)
            
            [Xhat, Xvar] = GAMPState.getState();
            [N, T] = size(Xhat);
            
            % Perform (scalar) learning of model parameters if desired
            if (obj.learn_lambda1 && mod(obj.EMcnt, 2) == 1) || ...
                    (obj.learn_lambda1 && ~obj.learn_lambda2)
                
                % Distinguish between EM learning and joint MAP-ML learning
                % depending on version of GAMP being executed
                switch obj.version
                    case 'mmse'
                        % Compute mean of |x_n|
                        sig = sqrt(Xvar);
                        mu = 2*sig .* normpdf(Xhat ./ sig) + Xhat .* ...
                            (1 - 2*normcdf(-Xhat ./ sig));

                        % Compute lambda1 update
                        lam_upd = N*T / sum(mu(:));
                        
                    case 'map'
                        % We must solve a transcendental equation
                        % numerically to identify the choice of lambda1
                        % that equates the ML cost function derivative to
                        % zero
                        opts = optimset('Jacobian', 'off', 'MaxIter', ...
                            20, 'Display', 'off');
                        F = @(lam1) zero_deriv1(obj, lam1, Xhat);
                        lam_upd = fsolve(F, mean(obj.lambda1(:)), opts);
                end
                
                if lam_upd <= 0
                    warning('EM update of lambda1 was negative...ignoring')
                    lam_upd = mean(obj.lambda1(:));
                end
                obj.lambda1 = TBobj.resize(lam_upd, N, T);
                obj.EMcnt = obj.EMcnt + 1;
            end
            
            if (obj.learn_lambda2  && mod(obj.EMcnt, 2) == 0) || ...
                    (obj.learn_lambda2 && ~obj.learn_lambda1)
                % Compute lambda2 update
                
                % Distinguish between EM learning and joint MAP-ML learning
                % depending on version of GAMP being executed
                switch obj.version
                    case 'mmse'
                        lam_upd = N / 4 / sum(Xvar(:) + Xhat(:).^2);
                        
                    case 'map'
                        % We must solve a transcendental equation
                        % numerically to identify the choice of lambda1
                        % that equates the ML cost function derivative to
                        % zero
                        opts = optimset('Jacobian', 'off', 'MaxIter', ...
                            20, 'Display', 'off');
                        F = @(lam2) zero_deriv2(obj, lam2, Xhat);
                        lam_upd = fsolve(F, mean(obj.lambda2(:)), opts);
                end
                
                if lam_upd <= 0
                    warning('EM update of lambda2 was negative...ignoring')
                    lam_upd = mean(obj.lambda2(:));
                end
                obj.lambda2 = TBobj.resize(lam_upd, N, T);
                obj.EMcnt = obj.EMcnt + 1;
            end
            
            % Return a binary hard estimate of S_POST
            S_POST = double(Xhat ~= 0);
            
            % Prepare next round of EstimIn objects for GAMP
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    EstimIn = ElasticNetEstimIn(obj.lambda1, obj.lambda2, ...
                        strcmpi('map', obj.version));
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        EstimIn{t} = ElasticNetEstimIn(obj.lambda1, ...
                            obj.lambda2, strcmpi('map', obj.version));
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
            
            obj.lambda1 = TBobj.resize(obj.lambda1, N, T);
            obj.lambda2 = TBobj.resize(obj.lambda2, N, T);
                        
            % Prepare initial round of EstimIn objects for GAMP
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimIn object and run
                    % matrix GAMP
                    EstimIn = ElasticNetEstimIn(obj.lambda1, obj.lambda2, ...
                        strcmpi('map', obj.version));
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimIn objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimIn = cell(1,T);
                    for t = 1:T
                        EstimIn{t} = ElasticNetEstimIn(obj.lambda1, ...
                            obj.lambda2, strcmpi('map', obj.version));
                    end
            end
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the
        % ElasticNet prior on X
        %
        % INPUTS:
        % obj       	An object of the Laplacian class
        % TBobj         An object of the TurboOpt class
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % X_TRUE        A realization of the signal array, X
        % S_TRUE        A realization of the support array, S
        % GAMMA_TRUE    A realization of the amplitude array, GAMMA
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
                    warning('SupportStruct may violate ElasticNet stats')
            end
            
            % Now produce a realization of GAMMA
            switch TBobj.AmplitudeStruct.get_type()
                case 'None'
                    % No amplitude structure, so draw iid
                    TempObj = ElasticNetEstimIn(obj.lambda1, obj.lambda2);
                    GAMMA_TRUE = TempObj.genRand([N, T]);
                    TempObj.delete();
                otherwise
                    % Call the genRand method of the particular form of
                    % amplitude structure to produce GAMMA_TRUE
                    AmpStruct = TBobj.AmplitudeStruct;
                    GAMMA_TRUE = AmpStruct.genRand(TBobj, GenParams);
                    warning('AmplitudeStruct may violate ElasticNet stats')
            end
            
            % Combine S_TRUE and GAMMA_TRUE to yield X_TRUE
            X_TRUE = S_TRUE .* GAMMA_TRUE;
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = double(obj.learn_lambda1) + double(obj.learn_lambda2);
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'lambda1',  'Ell-1 penalty',	'learn_lambda1'; 
                        'lambda2', 	'Ell-2 penalty',    'learn_lambda2'};
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
        
        
        % *****************************************************************
        %                         FSOLVE METHODS
        % *****************************************************************
        % This method is used by MATLAB's fsolve function in the max-sum
        % case to locate the value of lambda1 that sets the ML cost
        % function derivative equal to zero
        function [Fval] = zero_deriv1(obj, lam1, Xhat)
            [N, T] = size(Xhat);
            lam2 = mean(obj.lambda2(:));
            
            % Make sure numerical solver doesn't attempt non-positive 
            % values of lambda1
            if lam1 <= 0
                Fval = -realmax;
                return
            end
            
            C = -lam1 / sqrt(2*lam2);
            % Now compute the ratio normpdf(C)/normcdf(C)
            ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
            
            % Evaluate ML derivative at lambda1 = lam1
            Fval = N*T*(lam1/2/lam2 - (1/sqrt(2*lam2))*ratio) + ...
                sum(abs(Xhat(:)));
        end
        
        % This method is used by MATLAB's fsolve function in the max-sum
        % case to locate the value of lambda2 that sets the ML cost
        % function derivative equal to zero
        function [Fval] = zero_deriv2(obj, lam2, Xhat)
            [N, T] = size(Xhat);
            lam1 = mean(obj.lambda1(:));
            
            % Make sure numerical solver doesn't attempt non-positive 
            % values of lambda2
            if lam2 <= 0
                Fval = -realmax;
                return
            end
            
            C = -lam1 / sqrt(2*lam2);
            % Now compute the ratio normpdf(C)/normcdf(C)
            ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
            
            % Evaluate ML derivative at lambda2 = lam2
            Fval = N*T*(lam1^2/4/lam2^2 - lam1/2/sqrt(2)/lam2^(3/2)*ratio) + ...
                N*T/2/lam2 - sum(abs(Xhat(:)).^2);
        end
    end % Private methods
end % classdef