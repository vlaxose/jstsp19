% CLASS: JointSparse
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: SupportStruct
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class, which is a sub-class of the SupportStruct class, can be
%   used to define a form of support structure in which the N-by-T signal
%   matrix, X, is row-sparse, i.e., every column of X shares the same
%   support.  This type of structure can be found, for instance, in the
%   standard multiple measurement vector (MMV), or joint-sparse recovery, 
%   problem.
%
%   Mathematically, we associate with the N-by-T signal matrix, X, a hidden
%   discrete-valued length-N vector S, whose entries can assume D+1
%   discrete values, {0, 1, ..., D}.  When S(n) = 0, X(n,t) = 0 for all 
%   t = 1, ..., T.  When S(n) = d, (d = 1, ..., D), row n of X is assumed
%   to be non-zero, drawn from a distribution indexed by the value of d.
%   The choice of this distribution depends on the signal prior defined by
%   the Signal class.  As a concrete example, if the entries of X are
%   marginally Bernoulli-Gaussian distributed, (see BernGauss.m), then S(n)
%   can assume two values: 0 or 1.  In this case, pdf(X(n,t) | S(n) = 1) =
%   Normal(THETA(n,t), PHI(n,t)).
%
%   ** Note that this class assumes that there exists a property in the 
%   Signal class called sparsity_rate, which contains the prior 
%   probabilities Pr{S(n) = d} for d = 1, ..., D, i.e., Pr{S(n) = d} =
%   Signal.sparsity_rate(n,1,d) (d = 1, ..., D). **
%
% PROPERTIES (State variables)
%   No user-accessible properties for instances of this class
%
% METHODS (Subroutines/functions)
%	JointSparse()
%       - Default constructor.
%   S_TRUE = genRand(TBobj, GenParams)
%       - This method will generate a realization of the signal support
%         matrix, S, given a TurboOpt object (TBobj) and a GenParams
%         object.
%   get_type(obj)
%   	- Returns the character string 'JS' to indicate objects of the
%   	  JointSparse class
%   print(obj)
%   	- Prints the support structure type to the command window
%   JointSparseCopyObj = copy(obj)
%       - Create an independent copy of a JointSparse object, obj
%   [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
%       - This method accepts as inputs an object of the TurboOpt class, 
%         TBobj, and PI_OUT, an N-by-T-by-D tensor of outgoing messages
%         to the S(n) variable nodes from GAMP's most recent iteration.  
%         Using these inputs, UpdateSupport will produce two outputs, PI_IN
%         and S_POST.  PI_IN is an N-by-T-by-D tensor of messages that 
%         will be passed from the S(n) variable nodes to the GAMP factor 
%         nodes connected to X.  Note that the third dimension of the 
%         tensor is D, and not D+1, (see DESCRIPTION above for the 
%         definition of D).  This is because the "missing" element of 
%         PI_IN(n,t), which corresponds to the message related to 
%         S(n,t) = 0, can be implicitly obtained from the D remaining
%         messages, since all D+1 messages for each n must sum to 1 in 
%         order to form a valid pmf.  So, the message related to S(n,t) = d
%         (d = 1, ..., D), is given by PI_IN(n,t,d).  S_POST is an
%         N-by-1-by-D tensor of marginal posteriors, i.e.,
%         Pr{S(n) = d | Y} = S_POST(n,1,d), for d = 1, ..., D, while
%         Pr{S(n,t) = 0 | Y} = 1 - sum(S_POST(n,1,:)).  In addition to
%         returning these variables, UpdateSupport will update the prior
%         activity probabilities, Pr{S(n) = d}, using an expectation-
%         maximization method, if the user has requested to do so in the
%         Signal class. [Hidden method]
%   EMreport(obj)
%       - Returns an empty report (not needed for this class)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created (12/16/11; JAZ)
%       - Added genRand method (01/02/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef JointSparse < SupportStruct
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'JS';      % Joint sparse type identifier
    end % properties
    
    properties (Hidden)
        version = 'mmse';
    end
   
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = JointSparse()
            % Default constructor
        end
        
        
        % *****************************************************************
        %                          SET METHODS
        % *****************************************************************
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('JointSparse does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        function print(obj)
            fprintf('SUPPORT STRUCTURE: Joint Sparsity (Row Sparsity)\n')
        end
        
        
        % *****************************************************************
        %                         COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of a JointSparse object
        function JointSparseCopyObj = copy(obj)
            JointSparseCopyObj = JointSparse();
        end
        
        
        % *****************************************************************
        %                      ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of column support
        % structure is present
        function type_id = get_type(obj)
            type_id = obj.type;
        end
    end % methods
    
    methods (Hidden)
        % *****************************************************************
        %            	SUPPORT MESSAGES UPDATE METHOD
        % *****************************************************************
        
        function [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
            
            % Check for missing property
            if ~any(strcmp(properties(TBobj.Signal), 'sparsity_rate'))
                error('Signal object is missing sparsity_rate property')
            end
            
            % Start by obtaining size information from PI_OUT
            N = size(PI_OUT, 1);
            T = size(PI_OUT, 2);
            D = size(PI_OUT, 3);
            
            % Get priors, Pr{S(n) = d} = Lambda
            Lambda = TBobj.Signal.sparsity_rate;
            if size(Lambda, 2) == T
                % In most functions, TBobj.Signal.sparsity_rate must by
                % N-by-T-by-D in dimension, but here, the second
                % dimension should in reality equal 1
                Lambda = Lambda(:,1,:);
            end
            
            PI_IN = NaN(N,T,D);
            occur = 0;
            
            % Compute incoming messages to GAMP for next round
            for t = 1:T
                inc_ind = [1:t-1, t+1:T];     % Timesteps in the PI_OUT prod.
                if D == 1       % Take advantage of numerical stabilizer
                    exp_arg = log(1 - Lambda) + sum(log(1 - PI_OUT(:,inc_ind)), 2) - ...
                        log(Lambda) - sum(log(PI_OUT(:,inc_ind)), 2);
                    PI_IN(:,t) = (1 + exp(exp_arg)).^(-1);
                else
                    PI_IN(:,t,:) = Lambda.*prod(PI_OUT(:,inc_ind,:), 2);
                    if any(reshape(PI_IN(:,t,:), N*D, 1, 1) == 0) && ...
                        ~all(Lambda(:) == 0 | Lambda(:) == 1)
                        % This is not a support-aware genie, but we are
                        % encountering probs. equal to zero.  Report this,
                        % and set the zeros to realmins
                        if occur == 0,  % Suppress duplicate warnings
                            fprintf(['JointSparse.m: Numerical precision ' ...
                                'difficulties (PI_IN)\n'])
                            occur = 1;
                        end
                        EXTRACT = PI_IN(:,t,:);
                        EXTRACT(EXTRACT == 0) = realmin;
                        PI_IN(:,t,:) = EXTRACT;
                    end
                    NORM = sum(PI_IN(:,t,:), 3) + (1 - sum(Lambda, 3)) .* ...
                        prod(1 - sum(PI_OUT(:,inc_ind,:), 3), 2);
                    PI_IN(:,t,:) = PI_IN(:,t,:) ./ repmat(NORM, [1, 1, D]);
                end
            end
            
            % Compute posteriors, Pr{S(n) | Y}
            S_POST = (Lambda .* prod(PI_OUT, 2));
            NORM = sum(S_POST, 3) + (1 - sum(Lambda, 3)) .* ...
                    prod(1 - sum(PI_OUT, 3), 2);
            S_POST = S_POST ./ repmat(NORM, [1, 1, D]);
            
            % If user has specified EM parameter learning for sparsity
            % rate, update it now
            switch TBobj.Signal.learn_sparsity_rate
                case 'scalar'
                    % Update a single scalar
                    lambda_upd = sum(S_POST, 1) / N;
                case 'row'
                    error('Incompatible option for Signal.learn_sparsity_rate')
                case 'column'
                    error('Incompatible option for Signal.learn_sparsity_rate')
                case 'false'
                    % Do not update the prior
                    lambda_upd = TBobj.Signal.sparsity_rate;
            end
            TBobj.Signal.sparsity_rate = TBobj.resize(lambda_upd, N, T, D);
        end
        
        
        % *****************************************************************
        %                   GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the signal support
        % matrix, S
        %
        % INPUTS:
        % obj       	An object of the BernGauss class
        % TBobj         An object of the TurboOpt class
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % S_TRUE        An N-by-T realization of the support matrix, S
        function S_TRUE = genRand(obj, TBobj, GenParams)
            N = GenParams.N;
            T = GenParams.T;
            D = size(TBobj.Signal.sparsity_rate, 3);
            % LAMBDA will be resized to N-by-T-by-D, but all columns of
            % LAMBDA should be identical
            LAMBDA = TBobj.resize(TBobj.Signal.sparsity_rate, N, T, D);
            LAMBDASUM = cat(3, zeros(N,T), cumsum(LAMBDA, 3));
            RANDVEC = rand(N,1);
            S_TRUE = zeros(N,T);
            
            for d = 1:D
                inds = (RANDVEC > LAMBDASUM(:,1,d) & RANDVEC < ...
                    LAMBDASUM(:,1,d+1));    % Logical indexing
                S_TRUE(inds,:) = d;
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % No parameters, thus nothing to report
            Report = [];
        end
    end
   
end % classdef