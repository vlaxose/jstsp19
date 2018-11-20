% CLASS: MarkovField
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
%   to define the parameters of a two-dimensional Markov random field (MRF)
%   with a 4-connected lattice neighborhood structure, used to describe 
%   clustered sparsity in the non-zero elements of the signal matrix, X.
%   Specifically, assume that X is an N-by-T signal matrix, and that S is 
%   an N-by-T binary matrix in which S(n,t) = 0 if X(n,t) = 0, and 
%   S(n,t) = 1 otherwise.  Next, suppose that the 1's in of S tend to 
%   cluster together in a way that is well-described by a 2-D MRF.  Then an
%   object of this class can be created and used anywhere a SupportStruct 
%   object is used in order to specify the parameters of this MRF.
%
%   In order to define a 2-D MRF, three parameters are used: betaH, a
%   horizontal inverse temperature, betaV, vertical inverse temperature,
%   and alpha, a (loosely speaking) sparsity parameter.
%
%   If the user would like an expectation-maximization (EM) algorithm to
%   attempt to learn the values of beta (common for both betaH and betaV)
%   and alpha, then set the corresponding properties learn_beta and
%   learn_alpha to 'true' (the default) or 'false' to disable.
%
%   To create a MarkovField object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   MarkovField(), will create a MarkovField object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values, by 
%   using MATLAB's property/value string pairs convention, e.g.,
%   MarkovField('betaH', 0.50, 'learn_beta', 'false') will
%   construct a MarkovField object in which betaH = 0.50, and this value
%   will not be refined by an EM learning procedure.  Any parameters not 
%   explicitly set in the constructor will be set to their default values.
%
%   ** Note that the properties of this class will override any properties
%      relating to the prior activity probability, i.e., Pr{S(n,t) = 1}, in
%      a Signal class object. **
%
% PROPERTIES (State variables)
%   betaH                   Horizontal inverse temperature [Default: 0.40]
%   betaV                   Vertical inverse temperature [Default: 0.40]
%   alpha                   Sparsity parameter [Default: 0]
%   learn_beta             	Learn a common value for betaH and betaV using 
%                           a pseudo-ML algorithm?  (See DESCRIPTION
%                           for options)  [Default: 'true']
%   learn_alpha             Learn alpha using a pseudo-ML algorithm?  (See 
%                           DESCRIPTION for options)  [Default: 'true']
%   maxIter                 Maximum number of loopy belief propagation
%                           iterations to conduct at each turbo iteration
%                           [Default: 3]
%
% METHODS (Subroutines/functions)
%	MarkovField()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   MarkovField('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class MarkovField, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'MRF' when obj is a MarkovField
%         object
%   MarkovFieldCopyObj = copy(obj)
%       - Create an independent copy of the MarkovField object, obj
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
%         (e.g., 'beta'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., '3D inverse temperature'), 
%         and value is a numeric scalar containing the most recent EM 
%         update. [Hidden method]
%

%
% Coded by: Subhojit Som, Georgia Institute of Tech.
%           Philip Schniter, The Ohio State Univ.
% E-mail: subhojit@gatech.edu; schniter@ece.osu.edu
% Adapted by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created class file from Som/Schniter code (03/12/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef MarkovField < SupportStruct

    properties
        % Markov random field properties
        betaH = 0.4;            % Horizontal inverse temperature
        betaV = 0.4;            % Vertical inverse temperature
        alpha = 0;              % Sparsity parameter
        learn_beta = 'true';
        learn_alpha = 'true';
        maxIter = 3;            % Max # of loopy BP iters per turbo iter
    end % properties
       
    properties (Hidden)
        version = 'mmse';
    end
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'MRF';        % Markov random field type identifier
    end
    
    methods
        % *****************************************************************
        %                       CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = MarkovField(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(varargin{i}, varargin{i+1});
                end
            end
        end
        
        
        % *****************************************************************
        %                        SET METHODS
        % *****************************************************************
        
        % Set method for horizontal inverse temp (betaH)
        function obj = set.betaH(obj, betaH)
            if numel(betaH) ~= 1
                error('betaH must be a scalar')
            else
                obj.betaH = betaH;
            end
        end
        
        % Set method for vertical inverse temp (betaV)
        function obj = set.betaV(obj, betaV)
            if numel(betaV) ~= 1
                error('betaV must be a scalar')
            else
                obj.betaV = betaV;
            end
        end
        
        % Set method for sparsity parameter (alpha)
        function obj = set.alpha(obj, alpha)
            if numel(alpha) ~= 1
                error('alpha must be a scalar')
            else
                obj.alpha = alpha;
            end
        end
        
        % Set method for learn_beta
        function obj = set.learn_beta(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: learn_beta')
            end
            obj.learn_beta = lower(string);
        end
        
        % Set method for learn_alpha
        function obj = set.learn_alpha(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: learn_alpha')
            end
            obj.learn_alpha = lower(string);
        end
        
        % Set method for maxIter
        function obj = set.maxIter(obj, iter)
            if iter >= 1
                obj.maxIter = iter;
            else
                error('maxIter must be >= 1')
            end
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('MarkovField does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
            
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SUPPORT STRUCTURE: 2-D Markov Random Field\n')
            fprintf('            betaH: %s\n', ...
                form(obj, obj.betaH))
            fprintf('            betaV: %s\n', ...
                form(obj, obj.betaV))
            fprintf('            alpha: %s\n', ...
                form(obj, obj.alpha))
            fprintf('       learn_beta: %s\n', obj.learn_beta)
            fprintf('      learn_alpha: %s\n', obj.learn_alpha)
            fprintf('Max # of BP iters: %s\n', ...
                form(obj, obj.maxIter))
        end
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of a MarkovField object
        function MarkovFieldCopyObj = copy(obj)
            MarkovFieldCopyObj = MarkovField('betaH', obj.betaH, ...
                'betaV', obj.betaV, 'alpha', obj.alpha, 'learn_beta', ...
                obj.learn_beta, 'learn_alpha', obj.learn_alpha, ...
                'maxIter', obj.maxIter);
        end
        
        
        % *****************************************************************
        %                      ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of structure family
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
            
            % Take the outgoing PI_OUT messages, and use them to
            % perform loopy belief propagation between the S(n,t)
            % nodes that are adjacent along either columns or rows.
            [PI_IN, S_POST, beta_upd, alpha_upd] = mrf_spd(obj, PI_OUT);

            % If user has specified EM parameter learning for betaH & 
            % betaV, update their values now
            if strcmpi('true', obj.learn_beta)
                obj.betaH = beta_upd;
                obj.betaV = beta_upd;
            end
            
            % If user has specified EM parameter learning for alpha,
            % update its value now
            if strcmpi('true', obj.learn_alpha)
                obj.alpha = alpha_upd;
            end
            
            % If user has specified EM parameter learning for sparsity
            % rate, update it now
            switch TBobj.Signal.learn_sparsity_rate
                case 'scalar'
                    % Update a single scalar
                    lambda_upd = sum(sum(S_POST, 1), 2) / N / T;
                case 'row'
                    error(['Cannot learn a distinct sparsity rate for ' ...
                        'each row when support structure is a Markov ' ...
                        'random field'])
                case 'column'
                    error(['Cannot learn a distinct sparsity rate for ' ...
                        'each column when support structure is a Markov ' ...
                        'random field'])
                case 'false'
                    % Do not update the prior
                    lambda_upd = TBobj.Signal.sparsity_rate;
            end
            lambda_upd = min(max(0, lambda_upd), 1);
            TBobj.Signal.sparsity_rate = TBobj.resize(lambda_upd, N, T);
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the signal support
        % matrix, S
        %
        % INPUTS:
        % obj       	An object of the MarkovField class
        % TBobj         An object of the TurboOpt class [ignored]
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % S_TRUE        An N-by-T realization of the support matrix, S
        
        function S_TRUE = genRand(obj, ~, GenParams)
            
            % Unpack relevant parameters from GenParams
            N = GenParams.N;
            T = GenParams.T;
            
            % # of Gibbs iterations to use in simulating the MRF
            NGibbsIters = 150;
            
            % Call the genMRF method to produce S_TRUE
            S_TRUE = obj.genMRF(N, T, NGibbsIters);
            
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = strcmpi(obj.learn_alpha, 'true') + ...
                strcmpi(obj.learn_beta, 'true');
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'betaH',    '3D MRF inverse temp.',	'learn_beta';
                        'alpha',    '3D MRF sparsity param.', 'learn_alpha'};
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
        % MRF_SPD       This function will perform loopy belief propagation
        % on the factor graph defined by the two-dimensional lattice of
        % binary support indicator variables, each of which (except along
        % edges) is 4-connected.
        %
        % In addition, this function will perform expectation-maximization
        % (EM) updates of the MRF model parameters.
        %
        % SYNTAX:
        % [PI_IN, S_POST, p01_upd] = mrf_spd(obj, PI_OUT, LAMBDA)
        %
        % INPUTS:
        % obj           A MarkovField object
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
        % beta_upd		An updated estimate of betaH & betaV obtained from
        %               maximizing a pseudo-likelihood Ising function
        % alpha_upd		An updated estimate of alpha obtained from
        %               maximizing a pseudo-likelihood Ising function
        %
        function [PI_IN, S_POST, beta_upd, alpha_upd] = mrf_spd(obj, PI_OUT)
            
            % Get sizing information
            [N, T] = size(PI_OUT);
            
            % Unpack model parameters
            betaH = obj.betaH;
            betaV = obj.betaV;
            alpha = obj.alpha;
            maxIter = obj.maxIter;
            
            ebp = exp(betaH);
            ebq = exp(betaV);
            ebpi = exp(-betaH);
            ebqi = exp(-betaV);
            e0 = exp(alpha);
            e1 = exp(-alpha);
            
            % checkerboard pattern indices
            chk = (checkerboard(1, N/2, T/2) > 0);
            [blackIdx_x, blackIdx_y] = find(chk == 0);
            [whiteIdx_x, whiteIdx_y] = find(chk == 1);
            
%             La_spdec = reshape(La_spdec,[N Len 1]);
            
%             nodePot = zeros(Len,Len,2);
%             nodePot(:,:,1) = 1./(1+exp(La_spdec)); % prob that node is 0
%             nodePot(:,:,2) = 1-nodePot(:,:,1); % prob that node is 1
            nodePot = zeros(N,T,2);
            nodePot(:,:,1) = 1 - PI_OUT;    % Prob. that node is 0
            nodePot(:,:,2) = PI_OUT;        % Prob. that node is 1
            
            % Initialize messages
            msgFromRight = 0.5*ones(N,T,2);
            msgFromLeft = 0.5*ones(N,T,2);
            msgFromTop = 0.5*ones(N,T,2);
            msgFromBottom = 0.5*ones(N,T,2);
            
            prod0 = zeros(N,T);
            prod1 = prod0;
            
            for iter = 1:maxIter
                if(mod(iter,2) == 1)
                    x = blackIdx_x; y = blackIdx_y;
                else
                    x = whiteIdx_x; y = whiteIdx_y;
                end
                % Convert row-column indexing into linear indexing
                ind = sub2ind([N, T], x, y);
                ind1 = sub2ind([N, T, 2], x, y, ones(numel(x),1));
                ind2 = sub2ind([N, T, 2], x, y, 2*ones(numel(x),1));
                
                % update messages from left 
                prod0(:,2:end) = e0*nodePot(:,1:end-1,1) .* ...
                    msgFromLeft(:,1:end-1,1) .* msgFromTop(:,1:end-1,1) .* ...
                    msgFromBottom(:,1:end-1,1);
                prod1(:,2:end) = e1*nodePot(:,1:end-1,2) .* ...
                    msgFromLeft(:,1:end-1,2) .* msgFromTop(:,1:end-1,2) .* ...
                    msgFromBottom(:,1:end-1,2);
                p0 = prod0*ebp + prod1*ebpi;
                p1 = prod0*ebpi + prod1*ebp;
                sump0p1 = p0+p1;
                
                msgFromLeft(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromLeft(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromLeft(:,1,:) = 0.5;
                
                 % update messages from right 
                prod0(:,1:end-1) = e0*nodePot(:,2:end,1) .* ...
                    msgFromRight(:,2:end,1) .* msgFromTop(:,2:end,1) .* ...
                    msgFromBottom(:,2:end,1);
                prod1(:,1:end-1) = e1*nodePot(:,2:end,2) .* ...
                    msgFromRight(:,2:end,2) .* msgFromTop(:,2:end,2) .* ...
                    msgFromBottom(:,2:end,2);
                p0 = prod0*ebp + prod1*ebpi;
                p1 = prod0*ebpi + prod1*ebp;
                sump0p1 = p0 + p1;
                
                msgFromRight(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromRight(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromRight(:,end,:) = 0.5;
                
                % update messages from top 
                prod0(2:end,:) = e0*nodePot(1:end-1,:,1) .* ...
                    msgFromLeft(1:end-1,:,1) .* msgFromTop(1:end-1,:,1) .* ...
                    msgFromRight(1:end-1,:,1);
                prod1(2:end,:) = e1*nodePot(1:end-1,:,2) .* ...
                    msgFromLeft(1:end-1,:,2) .* msgFromTop(1:end-1,:,2) .* ...
                    msgFromRight(1:end-1,:,2);
                p0 = prod0*ebq + prod1*ebqi;
                p1 = prod0*ebqi + prod1*ebq;
                sump0p1 = p0 + p1;
                
                msgFromTop(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromTop(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromTop(1,:,:) = 0.5;
                
                 % update messages from bottom 
                prod0(1:end-1,:) = e0*nodePot(2:end,:,1) .* ...
                    msgFromRight(2:end,:,1) .* msgFromLeft(2:end,:,1) .* ...
                    msgFromBottom(2:end,:,1);
                prod1(1:end-1,:) = e1*nodePot(2:end,:,2) .* ...
                    msgFromRight(2:end,:,2) .* msgFromLeft(2:end,:,2) .* ...
                    msgFromBottom(2:end,:,2);
                p0 = prod0*ebq + prod1*ebqi;
                p1 = prod0*ebqi + prod1*ebq;
                sump0p1 = p0 + p1;
                
                msgFromBottom(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromBottom(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromBottom(end,:,:) = 0.5;
                
            end
            
            
            % compute extrinsic likelihood, marginal potential and s_hat
            msgProds = msgFromLeft .* msgFromRight .* msgFromTop .* ...
                msgFromBottom;
            msgProds(:,:,1) = msgProds(:,:,1)*e0;
            msgProds(:,:,2) = msgProds(:,:,2)*e1;
            Le_spdec = log(msgProds(:,:,2)./msgProds(:,:,1));
            PI_IN = 1 ./ (1 + exp(-Le_spdec));
            
            msgProds = msgProds.*nodePot;
            sumMsgProds = sum(msgProds,3);
            S_POST = msgProds(:,:,2) ./ sumMsgProds;    % Pr{S(n,t) = 1 | Y}
            S_HAT = double(S_POST > 1/2);
            
            % Compute parameter updates (will require MATLAB's Optimization
            % Toolbox).  Currently learns a single value of beta
            options = optimset('GradObj', 'on', 'Hessian', ...
                'user-supplied', 'MaxFunEvals', 20);
            lb = [-1; 0];   % Lower bounds [beta; alpha]
            ub = [1; 1];    % Upper bounds [beta; alpha]
            [updates] = fmincon(@pseudoLF, [betaH; alpha], [], [], [], ...
                [], lb, ub, [], options, S_HAT, N, T);
            beta_upd = updates(1);
            alpha_upd = updates(2);
            
            
            % This is the pseudo likelihood function for Ising model
            %
            % Copyright (c) Subhojit Som and Philip Schniter, 2012
            % Email: som.4@osu.edu, schniter@ece.osu.edu

            function [ f, g, H ] = pseudoLF(params, S_HAT, N, T)
                beta = params(1);
                alpha = params(2);

                S_HAT = (S_HAT-0.5)*2;

                horizontalZ = zeros(N,T);
                verticalZ   = zeros(N,T);

                horizontalZ(:,2:end-1) = S_HAT(:,1:end-2) + S_HAT(:,3:end);
                horizontalZ(:,1) = horizontalZ(:,2);
                horizontalZ(:,end) = horizontalZ(:,end-1);

                verticalZ(2:end-1,:) = S_HAT(1:end-2,:) + S_HAT(3:end,:);
                verticalZ(1,:) = verticalZ(2,:);
                verticalZ(end,:) = verticalZ(end-1,:);

                hvZ = horizontalZ + verticalZ;
                Z = hvZ*beta - alpha;

                f = sum(sum(- S_HAT.*(Z) + log(cosh(Z))));


                g = [sum(sum(tanh(Z).*hvZ - S_HAT.*hvZ )); sum(sum(-tanh(Z) + S_HAT))];

                H = NaN*ones(2);
                H(1,1) = sum(sum((1-(tanh(Z)).^2).*(hvZ.^2)));
                H(2,2) = sum(sum((1-(tanh(Z)).^2)));
                H(1,2) = sum(sum(-(1-(tanh(Z)).^2).*(hvZ)));
                H(2,1) = H(1,2);
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