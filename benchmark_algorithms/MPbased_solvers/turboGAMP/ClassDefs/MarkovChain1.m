% CLASS: MarkovChain1
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: SupportStruct
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class, which is a subclass of the SupportStruct class, can be used 
%   to define the parameters of a first-order Markov chain that describes 
%   the statistics of the support of the signal matrix, X, along either 
%   columns or rows.  Specifically, assume that X is an N-by-T signal 
%   matrix, and that S is an N-by-T binary matrix in which S(n,t) = 0 if 
%   X(n,t) = 0, and S(n,t) = 1 otherwise.  Next, suppose that the 1's in 
%   the columns or rows of S tend to cluster together in a way that is 
%   well-described by a first-order Markov chain.  Then, an object of this 
%   class can be created and used anywhere a SupportStruct object is used 
%   in order to specify the parameters of this Markov chain.
%
%   In order to define a steady-state Markov chain, two parameters are
%   used.  The first parameter, which is specified by the signal prior on X
%   (see Signal.m), is the prior probability that an element of X is
%   non-zero, i.e., Pr{S(n,t) = 1}.  The second parameter, p01, is the
%   probability that a support variable transitions to the value 0, given
%   that its neighbor had the value 1.  The definition of neighbor depends
%   on whether the Markov chain is describing the column support structure,
%   or the row support structure of X.  If describing the structure of the
%   columns of X, then p01 = Pr{S(n,t) = 0 | S(n-1,t) = 1}.  Likewise, if
%   the Markov chain is for the rows of X, then p01 = Pr{S(n,t) = 0 |
%   S(n,t-1) = 1}.  Note that the average run-length of a sequence of 1's
%   is 1/p01.  Together, Pr{S(n,t) = 1} and p01 are sufficient to fully
%   characterize a first-order Markov chain.  In this class, the property
%   "p01" holds the value of p01.  (A subclass of the Signal class will
%   hold the property Pr{S(n,t) = 1}).  The property "dim", which specifies
%   whether each row or each column of S forms a Markov chain, holds the
%   character string 'row' or 'col' respectively.
%
%   This class can also be used, more generally, in the event that there
%   are D distinct "active states", i.e., S(n,t) \in {0,1,...,D}, provided
%   that transitions between the active states are equiprobable, and each
%   active state is equally likely to occur in inactive-to-active
%   transitions.  To enable this behavior, this class should be paired with
%   a Signal class that allows for D-ary active states (e.g., GaussMix.m).
%
%   If the user would like an expectation-maximization (EM) algorithm to
%   attempt to learn the value p01 from the data, then the property
%   "learn_p01" can be set to 'true' (if not, set to 'false').
%
%   To create a MarkovChain1 object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   MarkovChain1(), will create a MarkovChain1 object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values, by 
%   using MATLAB's property/value string pairs convention, e.g.,
%   MarkovChain1('p01', 0.10, 'learn_p01', 'false') will
%   construct a MarkovChain1 object in which p01(n,t) = 0.10 for all n,t,
%   and this transition probability will not be refined by the expectation-
%   maximization (EM) parameter learning procedure.  Any parameters not 
%   explicitly set in the constructor will be set to their default values.
%
%   ** Note that this class assumes that there exists a property in the 
%   Signal class object called sparsity_rate, which contains the prior 
%   probabilities  Pr{S(n,t) = d}, i.e., Pr{S(n,t) = d} = 
%   Signal.sparsity_rate(n,t), d = 1,...,D. **
%
% PROPERTIES (State variables)
%   p01                     The prior active-to-inactive transition
%                           probability [Default: 0.05]
%   learn_p01               Learn p01 using EM algorithm?  (See DESCRIPTION
%                           for options)  [Default: 'true']
%   dim                     Is each row of S a Markov chain ('row'), or 
%                           each column ('col')? [Default: 'row']
%
% METHODS (Subroutines/functions)
%	MarkovChain1()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   MarkovChain1('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class MarkovChain1, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'MC1' when obj is a MarkovChain1
%         object
%   MarkovChain1CopyObj = copy(obj)
%       - Creates an independent copy of the MarkovChain1 object, obj
%   [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
%       - As required by the SupportStruct base class, this method accepts
%         a TurboOpt object, TBobj, as an input, along with outgoing
%         messages from GAMP to the N-by-T-by-1 binary support matrix S, in
%         PI_OUT.  Using these, it generates incoming messages to GAMP from
%         the S variables, along with marginal posteriors, Pr{S(n,t) = 1},
%         in S_POST. If EM parameter learning of the prior sparsity rate is
%         enabled, this method will also perform EM parameter learning of 
%         the Signal class property "sparsity_rate" [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'p01'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Active-to-inactive 
%         transition prob.'), and value is a numeric scalar containing the 
%         most recent EM update. [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created (10/21/11; JAZ)
%       - Replaced inheritance of MarkovChain1 from ColSuppStruct and
%         RowSuppStruct to the more general SupportStruct, and added a new
%         property, dim, which specifies whether Markov chain is over rows
%         or columns of support matrix S (12/09/11; JAZ)
%       - Added UpdateSupport method and associated helper method
%         (12/14/11; JAZ)
%       - Added EM learning of sparsity rate to UpdateSupport (01/02/12;
%         JAZ)
%       - Added genRand method (01/02/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef MarkovChain1 < SupportStruct

    properties
        % First-order Markov chain properties
        p01 = 0.05;     % Default active-to-inactive transition probability
        learn_p01 = 'true';      % Learn p01 using EM alg. by default
        dim = 'row';    % Each row forms a Markov chain
    end % properties
    
    properties (Hidden)
        version = 'mmse';
    end
       
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'MC1';        % 1st-order Markov chain type identifier
    end
    
    methods
        % *****************************************************************
        %                       CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = MarkovChain1(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(lower(varargin{i}), varargin{i+1});
                end
            end
        end
        
        
        % *****************************************************************
        %                        SET METHODS
        % *****************************************************************
        
        % Set method for active-to-inactive transition prob (p01)
        function obj = set.p01(obj, p01)
            if numel(p01) ~= 1
                error('p01 must be a scalar')
            end
            if p01 < 0 || p01 > 1
                error('p01 must be in the interval [0,1]')
            else
                obj.p01 = p01;
            end
        end
        
        % Set method for learn_p01
        function obj = set.learn_p01(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: learn_p01')
            end
            obj.learn_p01 = lower(string);
        end
        
        % Set method for dim
        function obj = set.dim(obj, string)
            if sum(strcmpi(string, {'row', 'col'})) == 0
                error('Invalid option: dim')
            end
            obj.dim = lower(string);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('MarkovChain1 does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
            
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SUPPORT STRUCTURE: Binary Markov Chain\n')
            switch obj.dim
                case 'row'
                    fprintf('   Structure of each row: 1st-order Markov chain\n')
                    fprintf('Structure of each column: No structure\n')
                case 'col'
                    fprintf('Structure of each column: 1st-order Markov chain\n')
                    fprintf('   Structure of each row: No structure\n')
            end
            fprintf('                     p01: %s\n', ...
                form(obj, obj.p01))
            fprintf('               learn_p01: %s\n', obj.learn_p01)
        end
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        
        % Creates an independent copy of a MarkovChain1 object
        function MarkovChain1CopyObj = copy(obj)
            MarkovChain1CopyObj = MarkovChain1('p01', obj.p01, ...
                'learn_p01', obj.learn_p01, 'dim', obj.dim);
        end
        
        
        % *****************************************************************
        %                      ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of signal family
        % this class is by returning a character string type identifier
        function type_id = get_type(obj)
            type_id = obj.type;
        end            
    end % methods
    
    methods (Hidden)
        % *****************************************************************
        %                 SUPPORT MESSAGES UPDATE METHOD
        % *****************************************************************
        
        function [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
            
            % Check for missing property
            if ~any(strcmp(properties(TBobj.Signal), 'sparsity_rate'))
                error('Signal object is missing sparsity_rate property')
            end
            
            N = size(PI_OUT, 1);
            T = size(PI_OUT, 2);
            D = size(PI_OUT, 3);
            PI_OUT = sum(PI_OUT, 3);
            
            % First-order Markov chain, thus perform a single
            % forward-backward pass
            LAMBDA = TBobj.Signal.sparsity_rate;    % Pr{S(n,t) = 1}
            LAMBDA = sum(LAMBDA, 3);

            % Take the outgoing PI_OUT messages, and use them to
            % perform a forward/backward pass between the S(n,t)
            % nodes that are adjacent along either columns or rows, 
            % depending on the user's specification.
            [PI_IN, S_POST, p01_upd] = binarymarkov_msgs(obj, PI_OUT, ...
                LAMBDA);

            % If user has specified EM parameter learning for p01,
            % update its value now
            if strcmpi('true', TBobj.SupportStruct.learn_p01)
                TBobj.SupportStruct.p01 = p01_upd;
            end
            
            % If user has specified EM parameter learning for sparsity
            % rate, update it now
            switch TBobj.Signal.learn_sparsity_rate
                case 'scalar'
                    % Update a single scalar
                    lambda_upd = sum(sum(S_POST, 1), 2) / N / T;
                case 'row'
                    % Update different lambda for each row
                    switch obj.dim
                        case 'row'
                            lambda_upd = sum(S_POST, 2) / T;
                        case 'col'
                            error(['Cannot learn a distinct sparsity ' ...
                                'rate for each column when each row ' ...
                                'of S forms a Markov chain'])
                    end
                case 'column'
                    % Update different lambda for each column
                    switch obj.dim
                        case 'row'
                            error(['Cannot learn a distinct sparsity ' ...
                                'rate for each row when each column ' ...
                                'of S forms a Markov chain'])
                        case 'col'
                            lambda_upd = sum(S_POST, 1) / N;
                    end
                case 'false'
                    % Do not update the prior
                    lambda_upd = TBobj.Signal.sparsity_rate;
            end
            lambda_upd = min(max(0, lambda_upd), 1);
            TBobj.Signal.sparsity_rate = TBobj.resize(lambda_upd, N, T, D);
            
            PI_IN = repmat((1/D)*PI_IN, [1, 1, D]);
            S_POST = repmat((1/D)*S_POST, [1, 1, D]);
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the signal support
        % matrix, S
        %
        % INPUTS:
        % obj       	An object of the MarkovChain1 class
        % TBobj         An object of the TurboOpt class
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % S_TRUE        An N-by-T realization of the support matrix, S
        
        function S_TRUE = genRand(obj, TBobj, GenParams)
            
            N = GenParams.N;
            T = GenParams.T;
            S_TRUE = NaN(N,T);
            LAMBDA = TBobj.resize(TBobj.Signal.sparsity_rate, N, T);
            P01 = TBobj.resize(obj.p01, N, T);
            P10 = LAMBDA .* P01 ./ (1 - LAMBDA);
            
            % Initial column (row) of S
            switch obj.dim
                case 'row'  % Each row forms a Markov chain
                    S_TRUE(:,1) = rand(N,1) < LAMBDA(:,1);
                case 'col'  % Each column forms a Markov chain
                    S_TRUE(1,:) = rand(1,T) < LAMBDA(1,:);
            end
            
            % Iterate through successive columns (rows) of the chain
            switch obj.dim
                case 'row'  % Each row forms a Markov chain
                    for t = 2:T
                        Inacts = find(S_TRUE(:,t-1) == 0);
                        S_TRUE(Inacts,t) = rand(numel(Inacts),1) < ...
                            P10(Inacts,t);
                        Acts = find(S_TRUE(:,t-1) == 1);
                        S_TRUE(Acts,t) = 1 - (rand(numel(Acts),1) < ...
                            P01(Acts,t));
                    end
                case 'col'  % Each column forms a Markov chain
                    for n = 2:N
                        Inacts = find(S_TRUE(n-1,:) == 0);
                        S_TRUE(n,Inacts) = rand(1,numel(Inacts)) < ...
                            P10(n,Inacts);
                        Acts = find(S_TRUE(n-1,:) == 1);
                        S_TRUE(n,Acts) = 1 - (rand(1,numel(Acts)) < ...
                            P01(n,Acts));
                    end
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = strcmpi(obj.learn_p01, 'true');
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'p01',    'Active-to-inactive Markov transition prob',	'learn_p01'};
            % Iterate through each parameter, adding to Report array as
            % needed
			j = 0;
            for i = 1:size(Params, 1)
                switch obj.(Params{i,3})
                    case 'true'
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
        % BINARYMARKOV_MSGS     This function will perform a single
        % forward/backward pass of sum-product message passing for 
        % binary-valued support indicator variables, S(n,t), that form 
        % support indicators.
        %
        % In addition, this function will perform expectation-maximization
        % (EM) updates of the active-to-inactive transition probability, 
        % p01.
        %
        % SYNTAX:
        % [PI_IN, S_POST, p01_upd] = binarymarkov_msgs(obj, PI_OUT, LAMBDA)
        %
        % INPUTS:
        % obj           A MarkovChain1 object
        % PI_OUT		An N-by-T matrix of incident messages to the S(n,t)
        %				variable nodes, each element being a probability in 
        %               [0,1]
        % LAMBDA        The priors on S, i.e., LAMBDA(n,t) = Pr{S(n,t) = 1}
        %
        % OUTPUTS:
        % PI_IN 		An N-by-T matrix of outgoing turbo messages from 
        %               the S(n,t) variable nodes, each element being a 
        %               probability in [0,1]
        % S_POST        An N-by-T matrix of posterior probabilities, 
        %               Pr{S(n,t) = 1 | Y}
        % p01_upd		An updated estimate of p01 obtained from an EM
        %               algorithm
        %
        function [PI_IN, S_POST, p01_upd] = binarymarkov_msgs(obj, PI_OUT, LAMBDA)

            % Fill in missing variables and do basic error-checking

            [N, T] = size(PI_OUT);
            
            % Make sure the PI_OUT given to us by the Signal class
            % UpdatePriors method makes sense
            if sum(sum((PI_OUT < 0) | (PI_OUT > 1))) ~= 0
                error('Elements of PI_OUT must be within [0,1]');
            end

            % Get p01, and make it N-by-T dimensional
            P01 = repmat(obj.p01, N, T);
            
            % Compute p10 based on p01 and LAMBDA to ensure steady-state
            % sparsity level
            P10 = obj.p01 * (LAMBDA ./ (1 - LAMBDA));
            
            % Initialize temporary message variables
            LAMBDA_FWD = NaN*ones(N,T);
            LAMBDA_BWD = NaN*ones(N,T);
            switch obj.dim
                case 'col'      % Markov chain is across rows of S
                    LAMBDA_FWD(1,:) = LAMBDA(1,:);
                    LAMBDA_BWD(N,:) = 1/2;
                case 'row'      % Markov chain is across columns of S
                    LAMBDA_FWD(:,1) = LAMBDA(:,1);
                    LAMBDA_BWD(:,T) = 1/2;
            end
            
            % Execute the forward and backward message passes along the chain
            
            switch obj.dim
                case 'col'  % Markov chain is across rows of S
                    % First the forward pass
                    for n = 2 : 1 : N
                        LAMBDA_FWD(n,:) = (P10(n,:) .* (1 - PI_OUT(n-1,:)) .* (1 - LAMBDA_FWD(n-1,:)) + ...
                            (1 - P01(n,:)) .* PI_OUT(n-1,:) .* LAMBDA_FWD(n-1,:)) ./ ...
                            ((1 - PI_OUT(n-1,:)) .* (1 - LAMBDA_FWD(n-1,:)) + ...
                            PI_OUT(n-1,:) .* LAMBDA_FWD(n-1,:));
                    end

                    % Now the backward pass
                    for n = N-1 : -1 : 1
                        LAMBDA_BWD(n,:) = (P01(n,:) .* (1 - PI_OUT(n+1,:)) .* (1 - LAMBDA_BWD(n+1,:)) + ...
                            (1 - P01(n,:)) .* PI_OUT(n+1,:) .* LAMBDA_BWD(n+1,:)) ./ ...
                            ((1 - P10(n,:) + P01(n,:)) .* (1 - PI_OUT(n+1,:)) .* (1 - LAMBDA_BWD(n+1,:)) + ...
                            (1 - P01(n,:) + P10(n,:)) .* PI_OUT(n+1,:) .* LAMBDA_BWD(n+1,:));
                    end

                case 'row'  % Markov chain is across columns of S
                    % First the forward pass
                    for t = 2 : 1 : T
                        LAMBDA_FWD(:,t) = (P10(:,t) .* (1 - PI_OUT(:,t-1)) .* (1 - LAMBDA_FWD(:,t-1)) + ...
                            (1 - P01(:,t)) .* PI_OUT(:,t-1) .* LAMBDA_FWD(:,t-1)) ./ ...
                            ((1 - PI_OUT(:,t-1)) .* (1 - LAMBDA_FWD(:,t-1)) + ...
                            PI_OUT(:,t-1) .* LAMBDA_FWD(:,t-1));
                    end

                    % Now the backward pass
                    for t = T-1 : -1 : 1
                        LAMBDA_BWD(:,t) = (P01(:,t) .* (1 - PI_OUT(:,t+1)) .* (1 - LAMBDA_BWD(:,t+1)) + ...
                            (1 - P01(:,t)) .* PI_OUT(:,t+1) .* LAMBDA_BWD(:,t+1)) ./ ...
                            ((1 - P10(:,t) + P01(:,t)) .* (1 - PI_OUT(:,t+1)) .* (1 - LAMBDA_BWD(:,t+1)) + ...
                            (1 - P01(:,t) + P10(:,t)) .* PI_OUT(:,t+1) .* LAMBDA_BWD(:,t+1));
                    end
            end

            % Now use the resulting messages to compute the outgoing turbo messages

            % Compute the messagescol that will be transmitted along the outgoing arc of
            % each s(n,t) to the calling function
            PI_IN = (LAMBDA_FWD .* LAMBDA_BWD) ./ ((1 - LAMBDA_FWD) .* ...
                (1 - LAMBDA_BWD) + LAMBDA_FWD .* LAMBDA_BWD);


            % Now for EM parameter updates of p01

            % First compute posterior means
            MU_S = (PI_OUT .* LAMBDA_FWD .* LAMBDA_BWD) ./ ((1 - PI_OUT) .* ...
                (1 - LAMBDA_FWD) .* (1 - LAMBDA_BWD) + PI_OUT .* LAMBDA_FWD .* ...
                LAMBDA_BWD);

            % Next compute E[S(n,t)*S(n-1,t) | Y] or E[S(n,t)*S(n,t-1) | Y]
            switch obj.dim
                case 'col'  % Markov chain is over the rows of S
                    PS0S0 = (1 - P10(1:N-1,:)) .* ((1 - LAMBDA_FWD(1:N-1,:)) .* ...
                        (1 - PI_OUT(1:N-1,:))) .* ((1 - LAMBDA_BWD(2:N,:)) .* ...
                        (1 - PI_OUT(2:N,:)));
                    PS1S0 = P10(1:N-1,:) .* ((1 - LAMBDA_FWD(1:N-1,:)) .* ...
                        (1 - PI_OUT(1:N-1,:))) .* ((LAMBDA_BWD(2:N,:)) .* ...
                        (PI_OUT(2:N,:)));
                    PS0S1 = P01(1:N-1,:) .* ((LAMBDA_FWD(1:N-1,:)) .* ...
                        (PI_OUT(1:N-1,:))) .* ((1 - LAMBDA_BWD(2:N,:)) .* ...
                        (1 - PI_OUT(2:N,:)));
                    PS1S1 = (1 - P01(1:N-1,:)) .* ((LAMBDA_FWD(1:N-1,:)) .* ...
                        (PI_OUT(1:N-1,:))) .* ((LAMBDA_BWD(2:N,:)) .* ...
                        (PI_OUT(2:N,:)));
                case 'row'  % Markov chain is over the columns of S
                    PS0S0 = (1 - P10(:,1:T-1)) .* ((1 - LAMBDA_FWD(:,1:T-1)) .* ...
                        (1 - PI_OUT(:,1:T-1))) .* ((1 - LAMBDA_BWD(:,2:T)) .* ...
                        (1 - PI_OUT(:,2:T)));
                    PS1S0 = P10(:,1:T-1) .* ((1 - LAMBDA_FWD(:,1:T-1)) .* ...
                        (1 - PI_OUT(:,1:T-1))) .* ((LAMBDA_BWD(:,2:T)) .* ...
                        (PI_OUT(:,2:T)));
                    PS0S1 = P01(:,1:T-1) .* ((LAMBDA_FWD(:,1:T-1)) .* ...
                        (PI_OUT(:,1:T-1))) .* ((1 - LAMBDA_BWD(:,2:T)) .* ...
                        (1 - PI_OUT(:,2:T)));
                    PS1S1 = (1 - P01(:,1:T-1)) .* ((LAMBDA_FWD(:,1:T-1)) .* ...
                        (PI_OUT(:,1:T-1))) .* ((LAMBDA_BWD(:,2:T)) .* ...
                        (PI_OUT(:,2:T)));
            end
            S_CORR = PS1S1 ./ (PS0S0 + PS0S1 + PS1S0 + PS1S1);

            % EM updates of p01
            switch obj.dim
                case 'col'
                    % Now update the active-to-inactive transition probability via EM
                    p01_upd = sum(sum(MU_S(1:N-1,:) - S_CORR)) / ...
                        sum(sum(MU_S(1:N-1,:)));
                    p01_upd = max(min(p01_upd, 1), 0);
                case 'row'
                    % Now update the active-to-inactive transition probability via EM
                    p01_upd = sum(sum(MU_S(:,1:T-1) - S_CORR)) / ...
                        sum(sum(MU_S(:,1:T-1)));
                    p01_upd = max(min(p01_upd, 1), 0);
            end

            % Finally return Pr{S(n,t) = 1 | Y} = E[S(n,t) | Y] = MU_S
            S_POST = MU_S;
        
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