% CLASS: MarkovField3D
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
%   to define the parameters of a three-dimensional Markov random field 
%   (MRF) with a 6-connected lattice neighborhood structure, used to 
%   describe clustered sparsity in the non-zero elements of a 3D signal 
%   that has been vectorized to a single column vector, X.  For simplicity,
%   we will assume in this description that X is of length Nx*Ny*Nz, and
%   that the unvectorized representation of X is as an Nx-by-Ny-by-Nz 3D
%   signal oriented on a cube lattice structure.  (In actuality, this class
%   requires the user to provide the (x,y,z) integer coordinates of each
%   entry of X, which thus determines the neighborhood structure between
%   elements of X.)  Now, assume that each entry of X, X(n), has an 
%   associated hidden binary indicator variable, S(n), that determines the
%   support of X as follows: when S(n) = 0, X(n) = 0, and when S(n) = 1,
%   X(n) ~= 0 (~=: not equal).  Next, suppose that the 1's in of S tend to 
%   cluster together in a way that is well-described by a 3D MRF.  Then an
%   object of this class can be created and used anywhere a SupportStruct 
%   object is used in order to specify the parameters of this MRF.
%
%   In order to define a 3D MRF, four parameters are used: betax, an
%   x-axis inverse temperature, betay, a y-axis inverse temperature,
%   betaz, a z-axis inverse temperature, and alpha, a (loosely speaking) 
%   sparsity parameter.  Additionally, since the signal, X, is assumed to
%   be vectorized from its 3D representation, it is necessary for the user
%   to specify the location (in 3-space) of each element X(n) on a lattice.
%   This information is stored in the N-by-3 matrix variable named
%   coordinates, where N is the length of X.  The nth row of coordinates
%   specifies the (x,y,z) integer location of X(n) in 3-space.
%   Neighbors are defined by consecutive integers, i.e., the neighbors of
%   an interior element X(n) located at integer location (x,y,z) are the
%   elements located at (x+1,y,z), (x-1,y,z), (x,y+1,z), (x,y-1,z),
%   (x,y,z+1), and (x,y,z-1).  If an element of X is missing from any of
%   these locations, a dummy element that conveys no information is
%   inserted.
%
%   If the user would like an expectation-maximization (EM) algorithm to
%   attempt to learn the values of beta (common for betax, betay and betaz)
%   and alpha, then set the corresponding properties learn_beta and
%   learn_alpha to 'true' (the default) or 'false' to disable.
%
%   To create a MarkovField3D object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   MarkovField3D(coordinates), will create a MarkovField3D object 
%   initialized with all default values for each parameter/property except
%   coordinates.  The alternative constructor allows the user to initialize 
%   any subset of the parameters (although coordinates must be included as
%   a mandatory input) with the remaining parameters initialized to their 
%   default values, by using MATLAB's property/value string pairs 
%   convention, e.g., MarkovField3D('coordinates', coord, 'learn_beta', 
%   'false') will construct a MarkovField3D object in which the property
%   "coordinates" is set to coord, and betax, betay, and betaz are not
%   refined using EM learning.  Any parameters not explicitly set in the 
%   constructor will be set to their default values.
%
%   ** Note that the properties of this class will override any properties
%      relating to the prior activity probability, i.e., Pr{S(n) = 1}, in
%      a Signal class object. **
%
% PROPERTIES (State variables)
%   coordinates             An N-by-3 matrix of integer coordinates for
%                           each entry of the length-N signal X, in (x,y,z)
%                           form
%   betax                   x-axis inverse temperature [Default: 0.40]
%   betay                   y-axis inverse temperature [Default: 0.40]
%   betaz                   z-axis inverse temperature [Default: 0.40]
%   alpha                   Sparsity parameter [Default: 0]
%   learn_beta             	Learn a common value for betax, betay, and 
%                           betaz using a pseudo-ML algorithm?  (See 
%                           DESCRIPTION for options)  [Default: 'true']
%   learn_alpha             Learn alpha using a pseudo-ML algorithm?  (See 
%                           DESCRIPTION for options)  [Default: 'true']
%   maxIter                 Maximum number of loopy belief propagation
%                           iterations to conduct at each turbo iteration
%                           [Default: 3]
%
% METHODS (Subroutines/functions)
%	MarkovField3D()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   MarkovField3D('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class MarkovField3D, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'MRF' when obj is a MarkovField3D
%         object
%   MarkovFieldCopyObj = copy(obj)
%       - Create an independent copy of the MarkovField3D object, obj
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
% Coded by: Subhojit Som
%           Philip Schniter, The Ohio State Univ.
% E-mail: ?; schniter@ece.osu.edu
% Adapted by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created class file from Som/Schniter code (03/12/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef MarkovField3D < SupportStruct

    properties
        % Markov random field properties
        betax = 0.4;            % x-axis inverse temperature
        betay = 0.4;            % y-axis inverse temperature
        betaz = 0.4;            % z-axis inverse temperature
        alpha = 0;              % Sparsity parameter
        coordinates;            % Integer (x,y,z) coordinates of each 
                                % element of X
        learn_beta = 'true';
        learn_alpha = 'true';
        maxIter = 3;            % Max # of loopy BP iters per turbo iter
    end % properties
       
    properties (Hidden)
        version = 'mmse';
        EMcnt = 0;
    end
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = '3DMRF';        % 3D Markov random field type identifier
    end
    
    methods
        % *****************************************************************
        %                       CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = MarkovField3D(varargin)
            if nargin == 1 
                % Constructor called using only coordinates arg
                obj.set('coordinates', varargin{1});
            elseif nargin > 1 && mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                coordsetflag = false;
                for i = 1 : 2 : nargin - 1
                    obj.set(varargin{i}, varargin{i+1});
                    if strcmpi('coordinates', varargin{i})
                        coordsetflag = true;
                    end
                end
                if ~coordsetflag
                    error('Must supply "coordinates" argument')
                end
            end
        end
        
        
        % *****************************************************************
        %                        SET METHODS
        % *****************************************************************
        
        % Set method for coordinates variable
        function obj = set.coordinates(obj, coords)
            if size(coords, 2) ~= 3
                error('"coordinates" must be an N-by-3 matrix')
            else
                coords = round(coords);             % Ensure integrality
                coords = coords - repmat(min(coords, [], 1) - 1, ...
                    size(coords, 1), 1);            % Offset coords so that
                                                    % smallest values of x
                                                    % y and z are at 1
                obj.coordinates = coords;    
            end
        end
        
        % Set method for x-axis inverse temp (betax)
        function obj = set.betax(obj, betax)
            if numel(betax) ~= 1
                error('betax must be a scalar')
            else
                obj.betax = betax;
            end
        end
        
        % Set method for y-axis inverse temp (betay)
        function obj = set.betay(obj, betay)
            if numel(betay) ~= 1
                error('betay must be a scalar')
            else
                obj.betay = betay;
            end
        end
        
        % Set method for z-axis inverse temp (betaz)
        function obj = set.betaz(obj, betaz)
            if numel(betaz) ~= 1
                error('betaz must be a scalar')
            else
                obj.betaz = betaz;
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
                obj.maxIter = round(iter);
            else
                error('maxIter must be >= 1')
            end
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('MarkovField3D does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
            
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SUPPORT STRUCTURE: 3D Markov Random Field\n')
            fprintf('       Signal coordinates: %s\n', ...
                form(obj, obj.coordinates))
            fprintf('x-axis inv. temp. (betax): %s\n', ...
                form(obj, obj.betax))
            fprintf('y-axis inv. temp. (betay): %s\n', ...
                form(obj, obj.betay))
            fprintf('z-axis inv. temp. (betaz): %s\n', ...
                form(obj, obj.betaz))
            fprintf('  sparsity param. (alpha): %s\n', ...
                form(obj, obj.alpha))
            fprintf('               learn_beta: %s\n', obj.learn_beta)
            fprintf('              learn_alpha: %s\n', obj.learn_alpha)
            fprintf('        Max # of BP iters: %s\n', ...
                form(obj, obj.maxIter))
        end
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of a MarkovField3D object
        function MarkovFieldCopyObj = copy(obj)
            MarkovFieldCopyObj = MarkovField3D('betax', obj.betax, ...
                'betay', obj.betay, 'alpha', obj.alpha, 'learn_beta', ...
                obj.learn_beta, 'learn_alpha', obj.learn_alpha, ...
                'maxIter', obj.maxIter, 'betaz', obj.betaz, ...
                'coordinates', obj.coordinates);
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
            
            % Verify that PI_OUT is an N-by-1 vector
            if numel(PI_OUT) ~= size(PI_OUT, 1)
                error('MarkovField3D expects signal to be a length-N column vector')
            else
                N = size(PI_OUT, 1);
            end
            
            % Take the outgoing PI_OUT messages, and use them to
            % perform loopy belief propagation between the S(n,t)
            % nodes that are adjacent along either columns or rows.
            [PI_IN, S_POST, beta_upd, alpha_upd] = mrf_spd(obj, PI_OUT);

            % If user has specified EM parameter learning for betaH & 
            % betaV, update their values now
            if strcmpi('true', obj.learn_beta)
                obj.betax = beta_upd;
                obj.betay = beta_upd;
                obj.betaz = beta_upd;
            end
            
            % If user has specified EM parameter learning for alpha,
            % update its value now
            if strcmpi('true', obj.learn_alpha)
                obj.set('alpha', alpha_upd);
            end
            
            % If user has specified EM parameter learning for sparsity
            % rate, update it now
            switch TBobj.Signal.learn_sparsity_rate
                case 'scalar'
                    % Update a single scalar
                    lambda_upd = sum(S_POST, 1) / N;
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
            TBobj.Signal.sparsity_rate = TBobj.resize(lambda_upd, N, 1);
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the signal support
        % matrix, S
        %
        % INPUTS:
        % obj       	An object of the MarkovField3D class
        % TBobj         An object of the TurboOpt class [ignored]
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % S_TRUE        An N-by-T realization of the support matrix, S
        
        function S_TRUE = genRand(obj, ~, GenParams)
            
            % Unpack relevant parameters from GenParams and attempt to
            % create a cubic structure for the 3-space lattice of the
            % signal, X, where each side is of length N^(1/3)
            if GenParams.T ~= 1
                error(['MarkovField3D expects an N-by-1 dimension for ' ...
                    'signal X, where N is some power of 3 (T=1)'])
            end
            N = round(GenParams.N ^ (1/3));
            if N^3 ~= GenParams.N
                error(['MarkovField3D expects an N-by-1 dimension for ' ...
                    'signal X, where N is some power of 3'])
            end
            
            % # of Gibbs iterations to use in simulating the MRF
            NGibbsIters = 1500;
            
            % Call the genMRF method to produce S_TRUE
            S_TRUE = obj.genMRF3D(N, N, N, NGibbsIters, true);
%             S_TRUE = obj.genMRF3Db(N, N, N, NGibbsIters, true);
            
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = strcmpi(obj.learn_alpha, 'true') + ...
                strcmpi(obj.learn_beta, 'true');
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'betax',    '3D MRF inverse temp.',	'learn_beta';
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
        % on the factor graph defined by the three-dimensional lattice of
        % binary support indicator variables, each of which (except along
        % boundaries) is 6-connected.
        %
        % In addition, this function will perform expectation-maximization
        % (EM) updates of the MRF model parameters.
        %
        % SYNTAX:
        % [PI_IN, S_POST, beta_upd, alpha_upd] = ...
        %       mrf_spd(obj, PI_OUT, LAMBDA)
        %
        % INPUTS:
        % obj           A MarkovField3D object
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
            N = size(PI_OUT, 1);
            
            % Enclose the 3-space defined by the coordinates property
            % within the smallest containing cube
            MaxVals = max(obj.coordinates, [], 1);
            Nx = MaxVals(1); Ny = MaxVals(2); Nz = MaxVals(3);
            
            
            % Declare model parameter-related constants        
            ebx = exp(obj.betax);
            eby = exp(obj.betay);
            ebz = exp(obj.betaz);
            ebxi = exp(-obj.betax);
            ebyi = exp(-obj.betay);
            ebzi = exp(-obj.betaz);
            e0 = exp(obj.alpha);
            e1 = exp(-obj.alpha);
            
            % checkerboard pattern indices in 3-space
            chk = NaN(Nx,Ny,Nz);
            check = checkerboard(1, Nx, Ny);
            check = check(1:Nx,1:Ny) > 0;
            for i = 1:Nz
                if mod(i, 2) == 0
                    chk(:,:,i) = ~check;
                else
                    chk(:,:,i) = check;
                end
            end
            blackIdx = find(chk == 0);
            whiteIdx = find(chk == 1);
            % Note the linear indices that belong to actual signal
            % coordinates...
            presentIdx = sub2ind([Nx, Ny, Nz], obj.coordinates(:,1), ...
                obj.coordinates(:,2), obj.coordinates(:,3));
            % ...and those that belong to dummy nodes
            missingIdx = setdiff(1:Nx*Ny*Nz, presentIdx);
            % Keep only those indices that were assigned coordinates
            blackIdx = intersect(blackIdx, presentIdx);
            whiteIdx = intersect(whiteIdx, presentIdx);
            
            
            % Initialize incoming probability messages
            nodePot = 1/2 * ones(Nx,Ny,Nz,2);
            nodePot(presentIdx) = 1 - PI_OUT;       	% Prob. that node is 0
            nodePot(presentIdx + Nx*Ny*Nz) = PI_OUT;	% Prob. that node is 1
            
            % Initialize messages
            msgFromRight = 1/2 * ones(Nx,Ny,Nz,2);
            msgFromLeft = 1/2 * ones(Nx,Ny,Nz,2);
            msgFromTop = 1/2 * ones(Nx,Ny,Nz,2);
            msgFromBottom = 1/2 * ones(Nx,Ny,Nz,2);
            msgFromFront = 1/2 * ones(Nx,Ny,Nz,2);
            msgFromBack = 1/2 * ones(Nx,Ny,Nz,2);
            
            prod0 = zeros(Nx,Ny,Nz);
            prod1 = prod0;
            
            for iter = 1:2*obj.maxIter
                % First grab indices of messages to update this round
                if(mod(iter,2) == 1)
                    ind = blackIdx;
                    ind1 = blackIdx;
                    ind2 = blackIdx + Nx*Ny*Nz;     % Offset linear idx
                else
                    ind = whiteIdx;
                    ind1 = whiteIdx;
                    ind2 = whiteIdx + Nx*Ny*Nz;     % Offset linear idx
                end
                
                % Update messages from left 
                prod0(:,2:end,:) = e0*nodePot(:,1:end-1,:,1) .* ...
                    msgFromLeft(:,1:end-1,:,1) .* msgFromTop(:,1:end-1,:,1) .* ...
                    msgFromBottom(:,1:end-1,:,1) .* msgFromFront(:,1:end-1,:,1) .* ...
                    msgFromBack(:,1:end-1,:,1);
                prod1(:,2:end,:) = e1*nodePot(:,1:end-1,:,2) .* ...
                    msgFromLeft(:,1:end-1,:,2) .* msgFromTop(:,1:end-1,:,2) .* ...
                    msgFromBottom(:,1:end-1,:,2) .* msgFromFront(:,1:end-1,:,2) .* ...
                    msgFromBack(:,1:end-1,:,2);
                p0 = prod0*ebx + prod1*ebxi;
                p1 = prod0*ebxi + prod1*ebx;
                sump0p1 = p0+p1;
                
                msgFromLeft(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromLeft(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromLeft(:,1,:,:) = 1/2;     	% Dummy edge msgs
                % The messages that are arriving from the left at nodes
                % whose left-side neighbors are dummy nodes should be
                % uninformative (1/2), thus we must correct such messages.
                % We can identify these nodes that have left-side dummy
                % neighbors using some linear indexing tricks
                correctedIdx = missingIdx + Nx;
                correctedIdx = correctedIdx(correctedIdx <= Nx*Ny*Nz);
                correctedIdx = [correctedIdx; correctedIdx + Nx*Ny*Nz];
                msgFromLeft(correctedIdx) = 1/2;    % Dummy missing msgs
                
                
                 % Update messages from right 
                prod0(:,1:end-1,:) = e0*nodePot(:,2:end,:,1) .* ...
                    msgFromRight(:,2:end,:,1) .* msgFromTop(:,2:end,:,1) .* ...
                    msgFromBottom(:,2:end,:,1) .* msgFromFront(:,2:end,:,1) .* ...
                    msgFromBack(:,2:end,:,1);
                prod1(:,1:end-1,:) = e1*nodePot(:,2:end,:,2) .* ...
                    msgFromRight(:,2:end,:,2) .* msgFromTop(:,2:end,:,2) .* ...
                    msgFromBottom(:,2:end,:,2) .* msgFromFront(:,2:end,:,2) .* ...
                    msgFromBack(:,2:end,:,2);
                p0 = prod0*ebx + prod1*ebxi;
                p1 = prod0*ebxi + prod1*ebx;
                sump0p1 = p0 + p1;
                
                msgFromRight(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromRight(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromRight(:,end,:,:) = 1/2;    	% Dummy edge msgs
                % As in the previous case, we must manually correct
                % messages at nodes whose right-side neighbors are dummy
                % nodes. Again, use linear indexing tricks
                correctedIdx = missingIdx - Nx;
                correctedIdx = correctedIdx(correctedIdx >= 1);
                correctedIdx = [correctedIdx; correctedIdx + Nx*Ny*Nz];
                msgFromRight(correctedIdx) = 1/2;   % Dummy missing msgs
                
                
                % Update messages from top 
                prod0(2:end,:,:) = e0*nodePot(1:end-1,:,:,1) .* ...
                    msgFromLeft(1:end-1,:,:,1) .* msgFromTop(1:end-1,:,:,1) .* ...
                    msgFromRight(1:end-1,:,:,1) .* msgFromFront(1:end-1,:,:,1) .* ...
                    msgFromBack(1:end-1,:,:,1);
                prod1(2:end,:,:) = e1*nodePot(1:end-1,:,:,2) .* ...
                    msgFromLeft(1:end-1,:,:,2) .* msgFromTop(1:end-1,:,:,2) .* ...
                    msgFromRight(1:end-1,:,:,2) .* msgFromFront(1:end-1,:,:,2) .* ...
                    msgFromBack(1:end-1,:,:,2);
                p0 = prod0*eby + prod1*ebyi;
                p1 = prod0*ebyi + prod1*eby;
                sump0p1 = p0 + p1;
                
                msgFromTop(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromTop(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromTop(1,:,:,:) = 1/2;          % Dummy edge msgs
                % As in the previous case, we must manually correct
                % messages at nodes whose top-side neighbors are dummy
                % nodes. Again, use linear indexing tricks
                correctedIdx = missingIdx + 1;
                correctedIdx = correctedIdx(correctedIdx <= Nx*Ny*Nz);
                correctedIdx = [correctedIdx; correctedIdx + Nx*Ny*Nz];
                msgFromTop(correctedIdx) = 1/2;  	% Dummy missing msgs
                
                
                % Update messages from bottom 
                prod0(1:end-1,:,:) = e0*nodePot(2:end,:,:,1) .* ...
                    msgFromRight(2:end,:,:,1) .* msgFromLeft(2:end,:,:,1) .* ...
                    msgFromBottom(2:end,:,:,1) .* msgFromFront(2:end,:,:,1) .* ...
                    msgFromBack(2:end,:,:,1);
                prod1(1:end-1,:,:) = e1*nodePot(2:end,:,:,2) .* ...
                    msgFromRight(2:end,:,:,2) .* msgFromLeft(2:end,:,:,2) .* ...
                    msgFromBottom(2:end,:,:,2) .* msgFromFront(2:end,:,:,2) .* ...
                    msgFromBack(2:end,:,:,2);
                p0 = prod0*eby + prod1*ebyi;
                p1 = prod0*ebyi + prod1*eby;
                sump0p1 = p0 + p1;
                
                msgFromBottom(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromBottom(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromBottom(end,:,:,:) = 1/2;     % Dummy edge msgs
                % As in the previous case, we must manually correct
                % messages at nodes whose bottom-side neighbors are dummy
                % nodes. Again, use linear indexing tricks
                correctedIdx = missingIdx - 1;
                correctedIdx = correctedIdx(correctedIdx >= 1);
                correctedIdx = [correctedIdx; correctedIdx + Nx*Ny*Nz];
                msgFromBottom(correctedIdx) = 1/2;	% Dummy missing msgs
                
                
                % Update messages from front 
                prod0(:,:,2:end) = e0*nodePot(:,:,1:end-1,1) .* ...
                    msgFromLeft(:,:,1:end-1,1) .* msgFromTop(:,:,1:end-1,1) .* ...
                    msgFromRight(:,:,1:end-1,1) .* msgFromFront(:,:,1:end-1,1) .* ...
                    msgFromBottom(:,:,1:end-1,1);
                prod1(:,:,2:end) = e1*nodePot(:,:,1:end-1,2) .* ...
                    msgFromLeft(:,:,1:end-1,2) .* msgFromTop(:,:,1:end-1,2) .* ...
                    msgFromRight(:,:,1:end-1,2) .* msgFromFront(:,:,1:end-1,2) .* ...
                    msgFromBottom(:,:,1:end-1,2);
                p0 = prod0*ebz + prod1*ebzi;
                p1 = prod0*ebzi + prod1*ebz;
                sump0p1 = p0 + p1;
                
                msgFromFront(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromFront(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromFront(:,:,1,:) = 1/2;        % Dummy edge msgs
                % As in the previous case, we must manually correct
                % messages at nodes whose front-side neighbors are dummy
                % nodes. Again, use linear indexing tricks
                correctedIdx = missingIdx + Nx*Ny;
                correctedIdx = correctedIdx(correctedIdx <= Nx*Ny*Nz);
                correctedIdx = [correctedIdx; correctedIdx + Nx*Ny*Nz];
                msgFromFront(correctedIdx) = 1/2;	% Dummy missing msgs
                
                
                % Update messages from back 
                prod0(:,:,1:end-1) = e0*nodePot(:,:,2:end,1) .* ...
                    msgFromRight(:,:,2:end,1) .* msgFromLeft(:,:,2:end,1) .* ...
                    msgFromBottom(:,:,2:end,1) .* msgFromTop(:,:,2:end,1) .* ...
                    msgFromBack(:,:,2:end,1);
                prod1(:,:,1:end-1) = e1*nodePot(:,:,2:end,2) .* ...
                    msgFromRight(:,:,2:end,2) .* msgFromLeft(:,:,2:end,2) .* ...
                    msgFromBottom(:,:,2:end,2) .* msgFromTop(:,:,2:end,2) .* ...
                    msgFromBack(:,:,2:end,2);
                p0 = prod0*ebz + prod1*ebzi;
                p1 = prod0*ebzi + prod1*ebz;
                sump0p1 = p0 + p1;
                
                msgFromBack(ind1) = p0(ind) ./ sump0p1(ind);
                msgFromBack(ind2) = p1(ind) ./ sump0p1(ind);
                msgFromBack(:,:,end,:) = 1/2;       % Dummy edge msgs
                % As in the previous case, we must manually correct
                % messages at nodes whose back-side neighbors are dummy
                % nodes. Again, use linear indexing tricks
                correctedIdx = missingIdx - Nx*Ny;
                correctedIdx = correctedIdx(correctedIdx >= 1);
                correctedIdx = [correctedIdx; correctedIdx + Nx*Ny*Nz];
                msgFromBack(correctedIdx) = 1/2;    % Dummy missing msgs
                
            end
            
            
            % Compute extrinsic likelihood, marginal potential and s_hat
            msgProds = msgFromLeft .* msgFromRight .* msgFromTop .* ...
                msgFromBottom .* msgFromFront .* msgFromBack;
            msgProds(:,:,:,1) = msgProds(:,:,:,1)*e0;
            msgProds(:,:,:,2) = msgProds(:,:,:,2)*e1;
            Le_spdec = log(msgProds(:,:,:,2)./msgProds(:,:,:,1));
            PI_IN = 1 ./ (1 + exp(-Le_spdec(presentIdx)));
            
            msgProds = msgProds.*nodePot;
            sumMsgProds = sum(msgProds, 4);
            S_POST = msgProds(:,:,:,2) ./ sumMsgProds;    % Pr{S(n) = 1 | Y}
            S_POST = S_POST(presentIdx);
            S_HAT = double(S_POST > 1/2);
            
            % *************************************************************
            % Compute the quantities that will be used to compute a
            % mean-field approximation to an expectation-maximization (EM)
            % update of the MRF parameters, alpha and beta.  In what
            % follows, we assume S(n) is drawn from {-1,1}
            PI_pad = 1/2*ones(Nx+2,Ny+2,Nz+2);	% Pad in all dimensions with dummy nodes
            PI = msgProds(:,:,:,2) ./ sumMsgProds;    % Pr{S(n) = 1 | Y}
            PI(missingIdx) = 1/2;
            PI_pad(2:Nx+1,2:Ny+1,2:Nz+1) = PI;  % Padded cube, w/ 1/2 at dummy nodes
            
            % Compute a couple different sums that appear often in the
            % mean-field update expressions
            ShiftPI_pad = 2*PI_pad - 1;     % 2*pi - 1
            NeighborhoodSum = ShiftPI_pad(1:Nx,2:Ny+1,2:Nz+1) + ...
                ShiftPI_pad(3:Nx+2,2:Ny+1,2:Nz+1) + ...
                ShiftPI_pad(2:Nx+1,1:Ny,2:Nz+1) + ...
                ShiftPI_pad(2:Nx+1,3:Ny+2,2:Nz+1) + ...
                ShiftPI_pad(2:Nx+1,2:Ny+1,1:Nz) + ...
                ShiftPI_pad(2:Nx+1,2:Ny+1,3:Nz+2);      % \sum_Neigh(n) (2*pi(q) - 1)
            % \sum_Neigh(n) (2*pi(n) - 1) (2*pi(q) - 1)
            AvgNeighborhoodSum = (2*PI - 1) .* NeighborhoodSum;
            
            
            % Compute parameter updates (will require MATLAB's Optimization
            % Toolbox).  Currently learns a single value of beta
            beta_upd = mean([obj.betax, obj.betay, obj.betaz]);
            alpha_upd = obj.alpha;
            options = optimset('GradObj', 'on', 'Hessian', ...
                'off', 'MaxFunEvals', 100, 'tolX', 1e-8, 'Display', ...
                'notify', 'Algorithm', 'interior-point');
            lb = [-10; -30];   % Lower bounds [alpha; beta]
            ub = [10; 30];    % Upper bounds [alpha; beta]
            if strcmpi(obj.learn_alpha, 'true') || ...
                    strcmpi(obj.learn_beta, 'true')
                [updates, ~, exitFlag] = fmincon(@meanfieldLF, ...
                    [obj.alpha; obj.betax], [], [], [], [], lb, ub, [], ...
                    options, PI, NeighborhoodSum, AvgNeighborhoodSum);
            end
            if obj.EMcnt >= 1
                if strcmpi(obj.learn_alpha, 'true'), alpha_upd = updates(1); end
                if strcmpi(obj.learn_beta, 'true'), beta_upd = updates(2); end
            end
            obj.EMcnt = obj.EMcnt + 1;

%             % *********************************
%             % No EM updates for the time being
%             % *********************************
%             beta_upd = mean([obj.betax, obj.betay, obj.betaz]);
%             alpha_upd = obj.alpha;
            
            
            
            % *************************************************************
            % meanfieldLF function
            % *************************************************************
            % This is the mean-field approximation of the likelihood
            % function
            %
            % INPUTS:
            %   params: A 2-by-1 vector of [beta; alpha] parameter values,
            %           at which the EM cost function and gradient are to 
            %           be computed
            %   PI:     An Nx-by-Ny-by-Nz tensor containing 
            %           Pr{s(i,j,k) = 1 | Y}
            %   NeighSum:   An Nx-by-Ny-by-Nz tensor whose (i,j,k)th
            %               element consists of \sum_q 2*pi(q) - 1, where
            %               the sum is over entries, q, that are neighbors
            %               of the (i,j,k)th voxel, and pi(q) = 
            %               Pr{s(q) = 1 | Y}.
            %   AvgNeighSum:    Similar to NeighSum, except that the
            %                   (i,j,k)th element is multiplied by
            %                   Pr{s(i,j,k) = 1 | Y}.
            %   
            % OUTPUTS:
            %   f:      Cost function at [alpha; beta]
            %   g:      Gradient of cost function at [alpha; beta]
            %
            % Coded by: Justin Ziniel, The Ohio State Univ.
            % E-mail: zinielj@ece.osu.edu
            % Last change: 04/13/13
            % Change summary: 
            %       - Created (04/13/13; JAZ)
            % Version 0.2
            
            function [f, g] = meanfieldLF(params, PI, NeighSum, AvgNeighSum)
                alpha = params(1);
                beta = params(2);
                
                % Start by computing the objective function value, which is
                % the posterior expectation of the mean field approximation
                % of the 3D MRF prior
                PosSum = -alpha + beta*NeighSum;    % s(n) = 1
                ExpPosSum = exp(PosSum);
                ExpNegSum = exp(-PosSum);
                f = alpha*(2*PI - 1) - beta*AvgNeighSum + log(ExpPosSum + ExpNegSum);
                f = sum(f(presentIdx));
                
                % Next, compute the derivative of the cost function w.r.t.
                % alpha and beta
                g_alpha = (ExpNegSum - ExpPosSum) ./ (ExpNegSum + ExpPosSum);
                g_alpha = (2*PI - 1) + g_alpha;     % deriv of f wrt alpha
                g_beta = (NeighSum.*ExpPosSum - NeighSum.*ExpNegSum) ./ ...
                    (ExpNegSum + ExpPosSum);
                g_beta = -AvgNeighSum + g_beta;     % deriv of f wrt beta
                
                g = [sum(g_alpha(presentIdx)); sum(g_beta(presentIdx))];
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