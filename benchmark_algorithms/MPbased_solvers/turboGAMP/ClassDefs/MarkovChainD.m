% CLASS: MarkovChainD
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
%   matrix, and that S is an N-by-T (D+1)-ary matrix in which S(n,t) = 0 if 
%   X(n,t) = 0, and S(n,t) = d otherwise, d = 1, ..., D.  The purpose of 
%   the non-zero entries of S is typically to index a particular mixture 
%   component that defines the distribution of the amplitude of X(n,t), 
%   conditioned on the fact that S(n,t) = d.  This class can be used to
%   model slow changes in the entries of S across either rows or columns
%   by defining a Markov chain between the various states (0, 1, ..., D).
%   It is assumed that the only transitions in states that occur across
%   rows or columns of S are either from 0 to d, or from d to 0 (d = 1,
%   ..., D).
%
%   In order to define the steady-state Markov chain, we rely on two
%   parameters.  The first parameter, which is specified by the signal 
%   prior on X (see Signal.m), is the prior probability that an element of 
%   X is drawn from mixture component d, i.e., Pr{S(n,t) = d}.  The second 
%   parameter, p0d, is the probability that a support variable transitions 
%   to state 0, given that its neighbor was in state d.  The definition of 
%   neighbor depends on whether the Markov chain is describing the column 
%   support structure, or the row support structure of X.  If describing 
%   the structure of the columns of X, then p0d = Pr{S(n,t) = 0 | 
%   S(n-1,t) = d}.  Likewise, if the Markov chain is for the rows of X, 
%   then p0d = Pr{S(n,t) = 0 | S(n,t-1) = d}.  Note that the average 
%   run-length of a sequence of d's is 1/p0d.  Together, Pr{S(n,t) = d} and
%   p0d (d = 1, ..., D) are sufficient to fully characterize a first-order 
%   (D+1)-ary Markov chain.  In this class, the property "p0d" holds the 
%   values of p0d for d = 1, ..., D.  (A subclass of the Signal class must 
%   hold the parameter Pr{S(n,t) = d} in a property named sparsity_rate.)  
%   The property "dim", which specifies whether each row or each column of 
%   S forms a Markov chain, holds the character string 'row' or 'col' 
%   respectively.
%
%   If the user would like an expectation-maximization (EM) algorithm to
%   attempt to learn the values of p0d for d = 1, ..., D from the data, 
%   then the property "learn_p0d" can be set to 'true' (if not, set to 
%   'false').
%
%   To create a MarkovChainD object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   MarkovChainD(), will create a MarkovChainD object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values, by 
%   using MATLAB's property/value string pairs convention, e.g.,
%   MarkovChainD('p0d', [0.10, 0.05], 'learn_p0d', 'false') will
%   construct a MarkovChainD object in which p01(n,t) = 0.10 and 
%   p02(n,t) = 0.05 for all n,t, and these transition probabilities will 
%   not be refined by the expectation-maximization (EM) parameter learning 
%   procedure.  Any parameters not explicitly set in the constructor will 
%   be set to their default values.
%
%   ** Note that this class assumes that there exists a property in the 
%   Signal class object called sparsity_rate, which contains the prior 
%   probabilities  Pr{S(n,t) = d}, i.e., Pr{S(n,t) = d} = 
%   Signal.sparsity_rate(n,t,d), (d = 1, ..., D). **
%
% PROPERTIES (State variables)
%   p0d                     The prior active-to-inactive transition
%                           probabilities.  Note that the value of D
%                           (# of active mixture components) is determined
%                           automatically from the input to p0d, thus
%                           inputs should have only D elements. 
%                           [Default: [0.05, 0.05] (D = 2)]
%   learn_p0d               Learn p0d using EM algorithm?  (See DESCRIPTION
%                           for options)  [Default: 'true']
%   dim                     Is each row of S a Markov chain ('row'), or 
%                           each column ('col')? [Default: 'row']
%
% METHODS (Subroutines/functions)
%	MarkovChainD()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   MarkovChainD('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class MarkovChainD, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'MCD' when obj is a MarkovChainD
%         object
%   MarkovChainDCopyObj = copy(obj)
%       - Creates an independent copy of a MarkovChainD object, obj
%   [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
%       - As required by the SupportStruct base class, this method accepts
%         a TurboOpt object, TBobj, as an input, along with outgoing
%         messages from GAMP to the N-by-T-by-D binary support matrix S, in
%         PI_OUT.  Using these, it generates N-by-T-by-D incoming messages 
%         to GAMP from the S variables, along with marginal posteriors, 
%         Pr{S(n,t) = d}, in S_POST. If EM parameter learning of the prior 
%         sparsity rate is enabled, this method will also perform EM 
%         parameter learning of the Signal class property "sparsity_rate" 
%         [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'p0d'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Active-to-inactive 
%         transition prob.'), and value is a numeric scalar containing the 
%         most recent EM update. [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created (03/01/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef MarkovChainD < SupportStruct

    properties
        % First-order Markov chain properties for D active mixture
        % components
        p0d = cat(3, 0.05, 0.05);	% Default active-to-inactive transition probabilities (D = 2)
        learn_p0d = 'true';       	% Learn p0d using EM alg. by default
        dim = 'row';                % Each row forms a Markov chain
    end % properties
    
    properties (Access = private, Dependent)
        D;      % The number of Gaussian mixture components
    end
    
    properties (Hidden)
        version = 'mmse';
    end
    
    properties (Access = private, Hidden)
        InitIter = true;        % Flag to prevent EM learning on 1st 
                                % smoothing iteration
    end
       
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'MCD';        % (D+1)-state Markov chain type identifier
    end
    
    methods
        % *****************************************************************
        %                       CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = MarkovChainD(varargin)
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
        function obj = set.p0d(obj, p0d)
            % Make sure that user doesn't try to provide an array of
            % transition probabilities.  We want D inputs only.
            [x, y, z] = size(p0d);
            if numel(find([x; y; z] == 1)) < 2
                error('p0d should have only one non-singleton dimension')
            end
            if any(p0d(:) < 0) || any(p0d(:) > 1)
                error('Elements of p0d must be in the interval [0,1]')
            else
                obj.p0d = reshape(p0d, [1, 1, numel(p0d)]);	% Use third dim. index
            end
        end
        
        % Set method for learn_p01
        function obj = set.learn_p0d(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: learn_p0d')
            end
            obj.learn_p0d = lower(string);
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
                error('MarkovChainD does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        % *****************************************************************
        %                           GET METHOD
        % *****************************************************************
        function D = get.D(obj)
            D = size(obj.p0d, 3);       % # of active mixture components
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SUPPORT STRUCTURE: (D+1)-ary Markov Chain\n')
            switch obj.dim
                case 'row'
                    fprintf('                     Structure of each row: 1st-order Markov chain\n')
                    fprintf('                  Structure of each column: No structure\n')
                case 'col'
                    fprintf('                  Structure of each column: 1st-order Markov chain\n')
                    fprintf('                     Structure of each row: No structure\n')
            end
            fprintf('        # of active mixture components (D): %d\n', obj.D);
            fprintf('Active-to-Inactive Transition Probs. (p0d): %s\n', ...
                form(obj, obj.p0d))
            fprintf('                                 learn_p0d: %s\n', obj.learn_p0d)
        end
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of a MarkovChainD object
        function MarkovChainDCopyObj = copy(obj)
            MarkovChainDCopyObj = MarkovChainD('p0d', obj.p0d, ...
                'learn_p0d', obj.learn_p0d, 'dim', obj.dim);
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
            elseif size(TBobj.Signal.sparsity_rate, 3) ~= obj.D
                error(['Signal class object appears to use a different ' ...
                    '# of mixture components (D) than MarkovChainD'])
            elseif size(PI_OUT, 3) ~= obj.D
                error(['PI_OUT third dimension appears to use a different ' ...
                    '# of mixture components (D) than MarkovChainD'])
            end
            
            N = size(PI_OUT, 1);
            T = size(PI_OUT, 2);
            D = obj.D;
            
            % First-order Markov chain, thus perform a single
            % forward-backward pass
            LAMBDA = TBobj.Signal.sparsity_rate;    % Pr{S(n,t) = d}

            % Take the outgoing PI_OUT messages, and use them to
            % perform a forward/backward pass between the S(n,t)
            % nodes that are adjacent along either columns or rows, 
            % depending on the user's specification.
            [PI_IN, S_POST, p0d_upd] = markov_msgs(obj, PI_OUT, LAMBDA);

            % If user has specified EM parameter learning for p01,
            % update its value now
            if strcmpi('true', TBobj.SupportStruct.learn_p0d)
                TBobj.SupportStruct.p0d = p0d_upd;
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
            TBobj.Signal.sparsity_rate = TBobj.resize(lambda_upd, N, T, D);
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the signal support
        % matrix, S
        %
        % INPUTS:
        % obj       	An object of the MarkovChainD class
        % TBobj         An object of the TurboOpt class
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % S_TRUE        An N-by-T realization of the support matrix, S
        
        function S_TRUE = genRand(obj, TBobj, GenParams)
            
            N = GenParams.N;
            T = GenParams.T;
            D = obj.D;
            S_TRUE = zeros(N,T);
            if size(TBobj.Signal.sparsity_rate, 3) ~= D
                error(['Model mismatch btwn. Signal class and SupportStruct' ...
                    ' class (# of mixture components, D)'])
            end
            LAMBDA = TBobj.resize(TBobj.Signal.sparsity_rate, N, T, D);
            P0D = TBobj.resize(obj.p0d, N, T, D);
            PD0 = LAMBDA .* P0D ./ repmat(1 - sum(LAMBDA, 3), [1, 1, D]);
            
            LAMBDASUM = cat(3, zeros(N,T), cumsum(LAMBDA, 3));
            PD0SUM = cat(3, zeros(N,T), cumsum(PD0, 3));
            
            % Initial column (row) of S
            switch obj.dim
                case 'row'  % Each row forms a Markov chain
                    randvec = rand(N,1);
                    for d = 1:D
                        inds = (randvec > LAMBDASUM(:,1,d) & randvec < ...
                            LAMBDASUM(:,1,d+1));
                        S_TRUE(inds,1) = d;
                    end
                case 'col'  % Each column forms a Markov chain
                    randvec = rand(1,T);
                    for d = 1:D
                        inds = (randvec > LAMBDASUM(1,:,d) & randvec < ...
                            LAMBDASUM(1,:,d+1));
                        S_TRUE(1,inds) = d;
                    end
            end
            
            % Iterate through successive columns (rows) of the chain
            switch obj.dim
                case 'row'  % Each row forms a Markov chain
                    for t = 2:T
                        Inacts = find(S_TRUE(:,t-1) == 0);
                        randvec = rand(numel(Inacts),1);
                        for d = 1:D
                            inds = randvec > PD0SUM(Inacts,t,d) & ...
                                randvec < PD0SUM(Inacts,t,d+1);
                            S_TRUE(Inacts(inds),t) = d;
                            Acts = find(S_TRUE(:,t-1) == d);
                            S_TRUE(Acts,t) = d * (1 - (rand(numel(Acts),1) < ...
                                P0D(Acts,t,d)));
                        end
                    end
                case 'col'  % Each column forms a Markov chain
                    for n = 2:N
                        Inacts = find(S_TRUE(n-1,:) == 0);
                        randvec = rand(1,numel(Inacts));
                        for d = 1:D
                            inds = randvec > PD0SUM(n,Inacts,d) & ...
                                randvec < PD0SUM(n,Inacts,d+1);
                            S_TRUE(n,Inacts(inds)) = d;
                            Acts = find(S_TRUE(n-1,:) == d);
                            S_TRUE(n,Acts) = d * (1 - (rand(1,numel(Acts)) < ...
                                P0D(n,Acts,d)));
                        end
                    end
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = strcmpi(obj.learn_p0d, 'true');
            Report = cell(obj.D*Nparam, 3);   % Declare Report array
            Params = {  'p0d',    'Active-to-inactive Markov transition prob.',	'learn_p0d'};
            % Iterate through each parameter, adding to Report array as
            % needed
            j = 0;
            for d = 1:obj.D
                for i = 1:size(Params, 1)
                    switch obj.(Params{i,3})
                        case 'true'
						j = j + 1;
                            Report{i + (d-1)*Nparam,1} = Params{i,1};
                            Report{i + (d-1)*Nparam,2} = [Params{i,2}, ' #', num2str(d)];
                            Report{i + (d-1)*Nparam,3} = obj.(Params{i,1})(1,1,d);
                        otherwise
                            % Don't add this parameter to the report
                    end
                end
            end
        end
    end
    
    methods (Access = private)
        % MARKOV_MSGS     This function will perform a single
        % forward/backward pass of sum-product message passing for 
        % (D+1)-ary-valued support indicator variables, S(n,t), that form 
        % support indicators.
        %
        % In addition, this function will perform expectation-maximization
        % (EM) updates of the active-to-inactive transition probabilities, 
        % p0d, d = 1, ..., D.
        %
        % SYNTAX:
        % [PI_IN, S_POST, p01_upd] = binarymarkov_msgs(obj, PI_OUT, LAMBDA)
        %
        % INPUTS:
        % obj           A MarkovChainD object
        % PI_OUT		An N-by-T-by-D matrix of incident messages to the 
        %				S(n,t) variable nodes, each element being a 
        %               probability in [0,1]
        % LAMBDA        An N-by-T-by-D matrix of priors on S, i.e., 
        %               LAMBDA(n,t,d) = Pr{S(n,t) = d}, d = 1, ..., D
        %
        % OUTPUTS:
        % PI_IN 		An N-by-T-by-D matrix of outgoing turbo messages 
        %               from the S(n,t) variable nodes, each element being 
        %               a probability in [0,1]
        % S_POST        An N-by-T-by-D matrix of posterior probabilities, 
        %               i.e., S_POST(n,t,d) = Pr{S(n,t) = d | Y}
        % p0d_upd		A 1-by-1-by-D vector of updated estimates of p0d 
        %               obtained from an EM algorithm
        %
        function [PI_IN, S_POST, p0d_upd] = markov_msgs(obj, PI_OUT, LAMBDA)

            % Fill in missing variables and do basic error-checking

            [N, T, D] = size(PI_OUT);
            
            if D ~= obj.D
                error(['The number of mixture components (D) declared in ' ...
                    'the Signal sub-class does not agree with the ' ...
                    'number declared in the MarkovChainD class'])
            end
            
            % Make sure the PI_OUT given to us by the Signal class
            % UpdatePriors method makes sense
            if any(PI_OUT(:) < 0) || any(PI_OUT(:) > 1)
                error('Elements of PI_OUT must be within [0,1]');
            end

            % Get p0d, and make it N-by-T-by-D dimensional
            P0D = repmat(obj.p0d, [N, T, 1]);
            
            % Compute pd0 based on p0d and LAMBDA to ensure steady-state
            % sparsity level
            PD0 = LAMBDA .* P0D ./ repmat(1 - sum(LAMBDA, 3), [1, 1, D]);
            
            % Initialize temporary message variables
            LAMBDA_FWD = NaN*ones(N,T,D);
            LAMBDA_BWD = NaN*ones(N,T,D);
            switch obj.dim
                case 'col'      % Markov chain is across rows of S
                    LAMBDA_FWD(1,:,:) = LAMBDA(1,:,:);
                    LAMBDA_BWD(N,:,:) = 1/(D + 1);
                case 'row'  % Markov chain is across columns of S
                    LAMBDA_FWD(:,1,:) = LAMBDA(:,1,:);
                    LAMBDA_BWD(:,T,:) = 1/(D + 1);
            end
            
            % Execute the forward and backward message passes along the chain
            
            switch obj.dim
                case 'col'  % Markov chain is across rows of S
                    % First the forward pass
                    B_FWD = NaN(N,T,D); B0_FWD = NaN(N,T,D);
                    for n = 2 : 1 : N
                        B_FWD(n-1,:,:) = LAMBDA_FWD(n-1,:,:) .* PI_OUT(n-1,:,:);
                        B0_FWD(n-1,:,:) = repmat((1 - sum(LAMBDA_FWD(n-1,:,:), 3)) .* ...
                            (1 - sum(PI_OUT(n-1,:,:), 3)), [1, 1, D]);
                        LAMBDA_FWD(n,:,:) = (PD0(n,:,:) .* B0_FWD(n-1,:,:) + ...
                            (1 - PD0(n,:,:)) .* B_FWD(n-1,:,:)) ./ ...
                            repmat(B0_FWD(n-1,:,1) + ...
                            sum(B_FWD(n-1,:,:), 3), [1, 1, D]);
                    end

                    % Now the backward pass
                    B_BWD = NaN(N,T,D); B0_BWD = NaN(N,T,D);
                    for n = N-1 : -1 : 1
                        B_BWD(n+1,:,:) = LAMBDA_BWD(n+1,:,:) .* PI_OUT(n+1,:,:);
                        B0_BWD(n+1,:,:) = repmat((1 - sum(LAMBDA_BWD(n+1,:,:), 3)) .* ...
                            (1 - sum(PI_OUT(n+1,:,:), 3)), [1, 1, D]);
                        P00 = 1 - sum(PD0(n,:,:), 3);
                        LAMBDA_BWD(n,:,:) = (P0D(n,:,:) .* B0_BWD(n+1,:,:) + ...
                            (1 - P0D(n,:,:)) .* B_BWD(n+1,:,:)) ./  ...
                            repmat(B0_BWD(n+1,:,1) .* (P00 + sum(P0D(n,:,:), 3)) + ...
                            sum((1 - P0D(n,:,:) + PD0(n,:,:)) .* B_BWD(n+1,:,:), 3), ...
                            [1, 1, D]);
                    end

                case 'row'  % Markov chain is across columns of S
                    % First the forward pass
                    B_FWD = NaN(N,T,D); B0_FWD = NaN(N,T,D);
                    for t = 2 : 1 : T
                        B_FWD(:,t-1,:) = LAMBDA_FWD(:,t-1,:) .* PI_OUT(:,t-1,:);
                        B0_FWD(:,t-1,:) = repmat((1 - sum(LAMBDA_FWD(:,t-1,:), 3)) .* ...
                            (1 - sum(PI_OUT(:,t-1,:), 3)), [1, 1, D]);
                        LAMBDA_FWD(:,t,:) = (PD0(:,t,:) .* B0_FWD(:,t-1,:) + ...
                            (1 - PD0(:,t,:)) .* B_FWD(:,t-1,:)) ./ ...
                            repmat(B0_FWD(:,t-1,1) + ...
                            sum(B_FWD(:,t-1,:), 3), [1, 1, D]);
                    end

                    % Now the backward pass
                    B_BWD = NaN(N,T,D); B0_BWD = NaN(N,T,D);
                    for t = T-1 : -1 : 1
                        B_BWD(:,t+1,:) = LAMBDA_BWD(:,t+1,:) .* PI_OUT(:,t+1,:);
                        B0_BWD(:,t+1,:) = repmat((1 - sum(LAMBDA_BWD(:,t+1,:), 3)) .* ...
                            (1 - sum(PI_OUT(:,t+1,:), 3)), [1, 1, D]);
                        P00 = 1 - sum(PD0(:,t,:), 3);
                        LAMBDA_BWD(:,t,:) = (P0D(:,t,:) .* B0_BWD(:,t+1,:) + ...
                            (1 - P0D(:,t,:)) .* B_BWD(:,t+1,:)) ./  ...
                            repmat(B0_BWD(:,t+1,1) .* (P00 + sum(P0D(:,t,:), 3)) + ...
                            sum((1 - P0D(:,t,:) + PD0(:,t,:)) .* B_BWD(:,t+1,:), 3), ...
                            [1, 1, D]);
                    end
            end

            % Now use the resulting messages to compute the outgoing turbo messages

            % Compute the messages that will be transmitted along the outgoing arc of
            % each s(n,t) to the calling function
            B0 = (1 - sum(LAMBDA_FWD, 3)) .* (1 - sum(LAMBDA_BWD, 3));
            PI_IN = (LAMBDA_FWD .* LAMBDA_BWD) ./ repmat(B0 + ...
                sum(LAMBDA_FWD .* LAMBDA_BWD, 3), [1, 1, D]);
            if any(PI_IN(:) < -1e-4) || any(PI_IN(:) > 1 + 1e-4)
                warning('Incoming probs PI_IN outside interval [0,1]')
            end
            PI_IN = max(0, min(PI_IN, 1));


            % Now for EM parameter updates of p01

            % First compute posteriors, Pr{S(n,t) = d | y}
            S_POST = (LAMBDA_FWD .* LAMBDA_BWD .* PI_OUT) ./ ...
                repmat(B0 .* (1 - sum(PI_OUT, 3)) + ...
                sum(LAMBDA_FWD .* LAMBDA_BWD, 3), [1, 1, D]);
            if any(S_POST(:) < -1e-4) || any(S_POST(:) > 1 + 1e-4)
                warning('Pr{S(n,t) = d | Y} outside interval [0,1]')
            end
            S_POST = max(0, min(S_POST, 1));

            % Next compute Pr{S(n,t) = 0 | S(n,t-1) = d | Y} and 
            % Pr{S(n,t) = d | S(n,t-1) = d | Y} (or similar for 'col' case)
            switch obj.dim
                case 'col'  % Markov chain is over the rows of S
                    PS0S0 = B0_FWD(1:N-1,:,1) .* (1 - sum(PD0(1:N-1,:,:), 3)) .* ...
                        B0_BWD(2:N,:,1);    % N-by-T-by-1 (unlike the rest)
                    PSDS0 = B0_FWD(1:N-1,:,:) .* PD0(2:N,:,:) .* B_BWD(2:N,:,:);
                    PS0SD = B_FWD(1:N-1,:,:) .* P0D(2:N,:,:) .* B0_BWD(2:N,:,:);
                    PSDSD = B_FWD(1:N-1,:,:) .* (1 - P0D(2:N,:,:)) .* B_BWD(2:N,:,:);
                case 'row'  % Markov chain is over the columns of S
                    PS0S0 = B0_FWD(:,1:T-1,1) .* (1 - sum(PD0(:,1:T-1,:), 3)) .* ...
                        B0_BWD(:,2:T,1);    % N-by-T-by-1 (unlike the rest)
                    PSDS0 = B0_FWD(:,1:T-1,:) .* PD0(:,2:T,:) .* B_BWD(:,2:T,:);
                    PS0SD = B_FWD(:,1:T-1,:) .* P0D(:,2:T,:) .* B0_BWD(:,2:T,:);
                    PSDSD = B_FWD(:,1:T-1,:) .* (1 - P0D(:,2:T,:)) .* B_BWD(:,2:T,:);
            end
            MU0D = PS0SD ./ repmat(PS0S0 + sum(PS0SD + PSDS0 + PSDSD, 3), ...
                [1, 1, D]);
            MUDD = PSDSD ./ repmat(PS0S0 + sum(PS0SD + PSDS0 + PSDSD, 3), ...
                [1, 1, D]);

            % EM updates of p0d
%             switch obj.dim
%                 case 'col'
%                     % Now update the active-to-inactive transition 
%                     % probability via EM
%                     p0d_upd = sum(sum(MU_S(1:N-1,:) - S_CORR)) / ...
%                         sum(sum(MU_S(1:N-1,:)));
%                     p0d_upd = max(min(p0d_upd, 1), 0);
%                 case 'row'
                    % Now update the active-to-inactive transition 
                    % probability via EM
                    p0d_upd = 1 ./ (1 + sum(sum(MUDD, 1), 2) ./ ...
                        sum(sum(MU0D, 1), 2));
                    p0d_upd = max(min(p0d_upd, 1), 0);
%             end
            
            % If this is the first turbo iteration, avoid updating the
            % model parameters
            if obj.InitIter
                p0d_upd = obj.p0d;          % Replace w/ old value
                obj.InitIter = false;       % Clear flag
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
                    size(prop, 1), size(prop, 2), size(prop,3), min(prop(:)), ...
                    max(prop(:)));
            end
        end
    end % Private methods
   
end % classdef