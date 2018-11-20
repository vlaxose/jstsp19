% CLASS: GaussMarkovFieldArb
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: AmplitudeStruct
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class, which is a subclass of the AmplitudeStruct class, can be 
%   used to define the parameters of a Gaussian Markov field consisting of 
%   an arbitrary neighborhood structure, for the purpose of describing
%   spatial and temporal amplitude similarity between neighboring elements
%   of a multidimensional signal that has been vectorized to a single 
%   column vector, X.  Now, assume that there is a neighborhood structure
%   defining spatial relationships between entries of X (we use X(n,t) to
%   denote the n^th spatial coefficient at timestep t).  If there is
%   spatial and temporal correlation between the amplitudes of X, then
%   a Gaussian Markov random field (GMRF) can be used to model such
%   correlation.  An object of this class can be created and used 
%   anywhere a AmplitudeStruct object is used in order to specify the 
%   parameters of this GMRF.
%
%   In order to define the strength of the spatial and temporal amplitude
%   correlations, we define two non-negative scalar parameters, beta and
%   gamma.  Spatial correlation is proportional to beta > 0, and temporal
%   correlation is proportional to gamma > 0.  Additionally, since the 
%   signal, X, is assumed to be vectorized, it is necessary for the user to 
%   provide two additional pieces of information: an adjacency matrix, 
%   AdjMtx, and a label array, LabelArr.  If the vectorized signal, X, is 
%   of length N*T, then AdjMtx is an N-by-N matrix with 1 in the (i,j)th 
%   location if nodes i and j (at any time, t) are neighbors (i.e., share 
%   an edge), and 0 otherwise.  LabelArr is an N-by-1 vector consisting of 
%   C unique integers, (1,...,C), which define a "graph coloring", i.e., 
%   any two nodes which share an edge will possess different labels.  This 
%   array is needed in order to ensure that adjacent nodes do not propagate 
%   messages simultaneously.  In summary, AdjMtx describes the *spatial*
%   connections between N different nodes, which remains static for all
%   t = 1,...,T timesteps.  It is assumed that, in addition to its spatial
%   neighbors, node X(n,t) is also connected to its temporal predecessor
%   and ancestor (i.e., X(n,t-1) and X(n,t+1)), with the strength of this
%   association governed by the temporal correlation parameter, gamma.  
%   Note that, if there is no temporal dimension to the problem (i.e., 
%   T = 1), AdjMtx will describe the entire neighborhood structure, and 
%   parameter gamma will have no importance.
%   
%   ********* EM LEARNING NOT YET IMPLEMENTED **************
%   If the user would like an expectation-maximization (EM) algorithm to
%   attempt to learn the value of beta or gamma, then set the corresponding 
%   properties learn_beta or learn_gamma to 'scalar' (to learn a scalar 
%   parameter---the default), or 'false' to disable.
%
%   To create a GaussMarkovFieldArb object, there are two constructors to 
%   choose from (see METHODS section below).  The default constructor, 
%   GaussMarkovFieldArb(AdjMtx, LabelArr), will create a GaussMarkovFieldArb 
%   object initialized with all default values for each parameter/property 
%   except AdjMtx and LabelArr.  The alternative constructor allows the 
%   user to initialize any subset of the parameters (although AdjMtx and 
%   LabelArr must be included as mandatory inputs) with the remaining 
%   parameters initialized to their default values, by using MATLAB's 
%   property/value string pairs convention, e.g., 
%   GaussMarkovFieldArb('AdjMtx', adjmtx, 'LabelArr', labelarr, ...
%   'learn_beta', 'false') will construct a GaussMarkovFieldArb object in 
%   which EM learning of beta is disabled.  Any parameters not explicitly 
%   set in the constructor will be set to their default values.
%
%   ** Note that the properties of this class will override any properties
%      relating to the prior mean or variance of amplitude coefficients
%      defined in any Signal class object. **
%
% PROPERTIES (State variables)
%   AdjMtx                  A valid N-by-N adjacency matrix, with a 1 in
%                           location (i,j) denoting that nodes i and j are
%                           neighbors (sharing an edge), and 0 otherwise
%   LabelArr                An N-by-1 array consisting of the labels
%                           1,...,C, where C is specified by the user, with
%                           the only constraint being that two adjacent
%                           nodes must have different labels
%   marginal_var            The marginal prior variance of each signal
%                           coefficient, i.e., var{X(n,t)} [Default: 1]
%   beta                    Spatial inverse temperature [Default: 0.40]
%   gamma                   Temporal inverse temperature [Default: 0.20]
%   learn_beta             	Learn a common value for beta using an 
%                           approximate mean-field EM algorithm?  (See 
%                           DESCRIPTION for options)  [Default: 'false']
%   learn_gamma             Learn a common value for gamma using an 
%                           approximate mean-field EM algorithm?
%                           [Default: 'false']
%   maxIter                 Maximum number of loopy belief propagation
%                           iterations to conduct at each turbo iteration
%                           [Default: 8]
%
% METHODS (Subroutines/functions)
%	GaussMarkovFieldArb(AdjMtx, LabelArr)
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   GaussMarkovFieldArb('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class GaussMarkovFieldArb, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'MRF' when obj is a GaussMarkovFieldArb
%         object
%   GaussMarkovFieldCopyObj = copy(obj)
%       - Create an independent copy of the GaussMarkovFieldArb object, obj
%   [MEAN_IN, VAR_IN, POST_MEAN, POST_VAR] = ...
%       UpdateAmplitude(obj, TBobj, MEAN_OUT, VAR_OUT)
%       - As required by the AmplitudeStruct base class, this method accepts
%         a TurboOpt object, TBobj, as an input, along with outgoing
%         messages from GAMP to the N*T-by-1 signal array X, in MEAN_OUT
%         and VA_OUT.  Using these, it generates incoming messages to GAMP 
%         via MEAN_IN and VAR_IN, along with marginal posteriors, 
%         p(X(n,t)|Y), which are Gaussian with mean POST_MEAN and variance
%         POST_VAR. If EM parameter learning is enabled, this method will 
%         also perform such learning [Hidden method]
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
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 07/27/13
% Change summary: 
%       - Created (07/27/13; JAZ)
% Version 0.2
%

classdef GaussMarkovFieldArb < AmplitudeStruct

    properties
        % Markov random field properties
        marginal_var = 1;       % Marginal prior variance
        beta = 0.4;             % Spatial inverse temperature
        gamma = 0.2;            % Temporal inverse temperature
        learn_beta = 'scalar';
        learn_gamma = 'scalar';
        maxIter = 8;            % Max # of loopy BP iters per turbo iter
    end % properties
    
    properties (Dependent)
        AdjMtx;                 % Adjacency matrix
        LabelArr;               % Graph coloring label array
        maxDeg;                 % Maximal graph degree
        C;                      % # of graph colors
        NbrList;                % N-by-maxDeg list of neighbors of each node
    end
       
    properties (Hidden)
        version = 'mmse';
        EMcnt = 0;
        AdjMtx_priv;
        LabelArr_priv;
        maxDeg_priv;
        C_priv;
        NbrList_priv;
        RevNbrList;             % An N-by-maxDeg linearly-indexed mapping
                                % that takes the messages stored in an
                                % Outbound Messages array, and places those
                                % elements into the correct locations in
                                % the Inbound Messages array for the next
                                % message passing cycle
        dummyMask;              % An N-by-maxDeg logical mask with 1's in 
                                % locations corresponding to a dummy node
                                % in NbrList
        inDeg;                  % An N-by-1 array containing the in-degree
                                % of each node
    end
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'ArbGMRF';       % Arbitrary Gaussian Markov random field type identifier
    end
    
    methods
        % *****************************************************************
        %                       CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = GaussMarkovFieldArb(varargin)
            if nargin == 2 && isnumeric(varargin{1}) && isnumeric(varargin{2})
                % Constructor called using only AdjMtx and LabelArr args
                obj.set('AdjMtx', varargin{1});
                obj.set('LabelArr', varargin{2});
            elseif nargin > 1 && mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                AdjMtxsetflag = false;
                Labelsetflag = false;
                for i = 1 : 2 : nargin - 1
                    obj.set(varargin{i}, varargin{i+1});
                    if strcmpi('AdjMtx', varargin{i})
                        AdjMtxsetflag = true;
                    elseif strcmpi('LabelArr', varargin{i})
                        Labelsetflag = true;
                    end
                end
                if ~AdjMtxsetflag || ~Labelsetflag
                    error('Must supply "AdjMtx" and "LabelArr" arguments')
                end
            end
        end
        
        
        % *****************************************************************
        %                        SET METHODS
        % *****************************************************************
        
        % Set method for adjacency matrix
        function obj = set.AdjMtx(obj, Mtx)
            assert(size(Mtx, 1) == size(Mtx, 2), 'AdjMtx must be N-by-N');
            obj.AdjMtx_priv = sparse(Mtx);      % Hopefully Mtx is sparse
            maxDeg = max(sum(Mtx, 2));
            obj.maxDeg_priv = maxDeg;           % Maximum node degree
            
            % Build an N-by-maxDeg list of neighbors, padding with a
            % "dummy" node as needed
            N = size(Mtx, 1);
            dummyID = N + 1;
            InDeg = zeros(N,1);     % In-degree of each node
            NbrList = dummyID * ones(N,maxDeg);
            for i = 1:N
                Idxs = find(Mtx(i,:) == 1);
                NbrList(i,1:numel(Idxs)) = Idxs;
                InDeg(i) = numel(Idxs);
            end
            obj.NbrList_priv = NbrList;
            obj.dummyMask = (NbrList == dummyID);   % 1's in dummy node locs
            obj.inDeg = InDeg;
            
            % NbrList is a bookkeeping array whose i^th row tells us the 
            % indices of the destination nodes for messages leaving node i.  
            % RevNbrList is a linearly-indexed mapping that takes the 
            % outgoing messages from the outgoing message array and places 
            % them in the correct places in the incoming message array for 
            % the next message passing iteration
            dumIdx = find(NbrList == dummyID, 1); 	% Index of 1st dummy loc
            revNbrList = dumIdx * ones(N,maxDeg);   % Default to dummy outgoing msgs
            OffsetAdjMtx = spalloc(N, N, N*maxDeg);
            for i = 1:N
                OffsetAdjMtx(:,i) = sum(Mtx(:,1:i), 2) .* Mtx(:,i);
                rowIdcs = NbrList(i,1:InDeg(i));
                revNbrList(i,1:InDeg(i)) = ...
                    N*(OffsetAdjMtx(rowIdcs,i) - 1) + rowIdcs';
            end
            obj.RevNbrList = revNbrList;
        end
        
        % Set method for graph coloring label array
        function obj = set.LabelArr(obj, Arr)
            assert(size(Arr,2) == 1, 'LabelArr must be an N-by-1 array')
            obj.LabelArr_priv = Arr;
            obj.C_priv = numel(unique(Arr));
        end
        
        % Set method for marginal prior variance
        function obj = set.marginal_var(obj, var)
			assert(isnumeric(var) && all(var(:) >= 0), 'marginal_var must be non-negative')
            obj.marginal_var = var;
        end
        
        % Set method for spatial inverse temp (beta)
        function obj = set.beta(obj, beta)
			assert(isnumeric(beta) && all(beta(:) >= 0), 'beta must be non-negative')
            obj.beta = beta;
        end
        
        % Set method for temporal inverse temp (gamma)
        function obj = set.gamma(obj, gamma)
            assert(isnumeric(gamma) && isscalar(gamma) && ...
                gamma >= 0, 'gamma must be non-negative')
            obj.gamma = gamma;
        end
        
        % Set method for learn_beta
        function obj = set.learn_beta(obj, string)
            if sum(strcmpi(string, {'scalar', 'timestep', 'false'})) == 0
                error('Invalid option: learn_beta')
            end
            obj.learn_beta = lower(string);
        end
        
        % Set method for learn_gamma
        function obj = set.learn_gamma(obj, string)
            if sum(strcmpi(string, {'scalar', 'false'})) == 0
                error('Invalid option: learn_gamma')
            end
            obj.learn_gamma = lower(string);
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
                error('GaussMarkovFieldArb does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        
        % *****************************************************************
        %                        GET METHODS
        % *****************************************************************
        
        % Get method for adjacency matrix
        function AdjMtx = get.AdjMtx(obj)
            AdjMtx = obj.AdjMtx_priv;
        end
        
        % Get method for graph coloring label array
        function LabelArr = get.LabelArr(obj)
            LabelArr = obj.LabelArr_priv;
        end
        
        % Get method for maximal node degree
        function maxDeg = get.maxDeg(obj)
            maxDeg = obj.maxDeg_priv;
        end
        
        % Get method for # of unique labels (colors)
        function C = get.C(obj)
            C = obj.C_priv;
        end
        
        % Get method for neighbor list
        function NbrList = get.NbrList(obj)
            NbrList = obj.NbrList_priv;
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('AMPLITUDE STRUCTURE: Arbitrary Gaussian Markov Random Field\n')
            fprintf('       Marginal prior variance: %s\n', ...
                form(obj, obj.marginal_var))
            fprintf('  Spatial inverse temp. (beta): %s\n', ...
                form(obj, obj.beta))
            fprintf('Temporal inverse temp. (gamma): %s\n', ...
                form(obj, obj.gamma))
            fprintf('                    learn_beta: %s\n', obj.learn_beta)
            fprintf('                   learn_gamma: %s\n', obj.learn_gamma)
            fprintf('             Max # of BP iters: %s\n', ...
                form(obj, obj.maxIter))
        end
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of a GaussMarkovFieldArb object
        function GaussMarkovFieldCopyObj = copy(obj)
            GaussMarkovFieldCopyObj = GaussMarkovFieldArb('beta', obj.beta, ...
                'learn_beta', obj.learn_beta, ...
                'maxIter', obj.maxIter, ...
                'AdjMtx', obj.AdjMtx, 'LabelArr', obj.LabelArr, ...
                'gamma', obj.gamma, 'learn_gamma', obj.learn_gamma, ...
                'marginal_var', obj.marginal_var);
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
        %                 AMPLITUDE MESSAGES UPDATE METHOD
        % *****************************************************************
        
        function [MEAN_IN, VAR_IN, POST_MEAN, POST_VAR] = ...
                UpdateAmplitude(obj, TBobj, MEAN_OUT, VAR_OUT)
            
            % Verify that MEANOUT and VAR_OUT are N*T-by-1 vectors
            assert((numel(MEAN_OUT) == size(MEAN_OUT, 1)) && ...
                (numel(VAR_OUT) == size(VAR_OUT, 1)), ...
                'GaussMarkovFieldArb expects signal to be a length-N*T column vector');
            
            % Take the outgoing MEAN_OUT and VAR_OUT messages, and use them to
            % perform loopy belief propagation between adjacent temporal
            % and spatial nodes X(n,t)
            [MEAN_IN, VAR_IN, POST_MEAN, POST_VAR, beta_upd, gamma_upd] = ...
                gmrf_spd(obj, MEAN_OUT, VAR_OUT, TBobj);

            % Update model parameters now (if learning is disabled, updated
            % values will be the same as existing values)
            obj.set('beta', beta_upd);
            obj.set('gamma', gamma_upd);
        end
        
        
        % *****************************************************************
        %                 AMPLITUDE MESSAGES INIT METHOD
        % *****************************************************************
        function InitPriors(obj, TBobj, Y, A)
            % Nothing to do
        end
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the signal
        % amplitude matrix, G. We assume that the GenParams object
        % specifies a dimension that is a power of 3, i.e., we assume
        % that G consists of an N-by-N-by-N lattice structure with 6
        % nearest neighbors (4 spatial and 2 temporal), and that
        % GenParams.N = N^3.
        %
        % INPUTS:
        % obj       	An object of the GaussMarkovFieldArb class
        % TBobj         An object of the TurboOpt class [ignored]
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % G_TRUE        An N^3 realization of the support matrix, S
        
        function G_TRUE = genRand(obj, TBobj, GenParams)
            
            % Unpack relevant parameters from GenParams and attempt to
            % create a cubic structure for the 3-space lattice of the
            % signal, X, where each side is of length N^(1/3)
            if GenParams.T ~= 1
                error(['GaussMarkovFieldArb expects an N-by-1 dimension for ' ...
                    'signal X, where N is some power of 3'])
            end
            N = round(GenParams.N ^ (1/3));
            if N^3 ~= GenParams.N
                error(['GaussMarkovFieldArb expects an N-by-1 dimension for ' ...
                    'signal X, where N is some power of 3'])
            end
            
            % # of Gibbs iterations to use in simulating the GMRF
            NGibbsIters = 1500;
            
            % Call the genGMRF method to produce G_TRUE
            G_TRUE = obj.genGMRF3D(TBobj, N, NGibbsIters);
            G_TRUE = G_TRUE(:);
            
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = any(strcmpi(obj.learn_beta, {'scalar', 'timestep'})) + ...
                strcmpi(obj.learn_gamma, 'scalar');
            Report = cell(Nparam, 3);   % Declare Report array
            Params = {  'beta',     'MRF spatial inverse temp.', 'learn_beta';
                        'gamma',    'MRF temporal inverse temp.', 'learn_gamma'};
            % Iterate through each parameter, adding to Report array as
            % needed
            j = 0;
            for i = 1:size(Params, 1)
                switch obj.(Params{i,3})
                    case {'scalar', 'timestep'}
                        j = j + 1;
                        Report{j,1} = Params{i,1};
                        Report{j,2} = Params{i,2};
                        Report{j,3} = squeeze(obj.(Params{i,1}));
                    otherwise
                        % Don't add this parameter to the report
                end
            end
        end
    end
    
    methods (Access = private)
        % GMRF_SPD      This function will perform loopy belief propagation
        % on the factor graph defined by the adjacency matrix provided 
        % during object construction.
        %
        % In addition, this function will perform expectation-maximization
        % (EM) updates of the GMRF model parameters.
        %
        % SYNTAX:
        % [MEAN_IN, VAR_IN, POST_MEAN, POST_VAR, beta_upd, gamma_upd] = ...
        %       gmrf_spd(obj, MEAN_OUT, VAR_OUT, TBobj)
        %
        % INPUTS:
        % obj           A GaussMarkovFieldArb object
        % MEAN_OUT      An N*T-by-1 vector of outgoing (from GAMP) Gaussian
        %               message means
        % VAR_OUT       An N*T-by-1 vector of outgoing (from GAMP) Gaussian
        %               message variances
        % TBobj 		The parent TurboOpt object
        %
        % OUTPUTS:
        % MEAN_IN       An N*T-by-1 vector of incoming (to GAMP) Gaussian
        %               message means
        % VAR_IN        An N*T-by-1 vector of incoming (to GAMP) Gaussian
        %               message means
        % POST_MEAN     An N*T-by-1 vector of posterior amplitude means
        % POST_VAR      An N*T-by-1 vector of posterior amplitude variances
        % beta_upd		An updated estimate of beta obtained via a
        %               mean-field approximate EM update
        % gamma_upd		An updated estimate of gamma obtained via a
        %               mean-field approximate EM update
        %
        function [MEAN_IN, VAR_IN, POST_MEAN, POST_VAR, beta_upd, gamma_upd] = ...
                gmrf_spd(obj, MEAN_OUT, VAR_OUT, TBobj)
            
            % Extract key graph-related variables
            maxDeg = obj.maxDeg;        % Maximal node degree
            NbrList = obj.NbrList;      % List of neighbors for each node
            LabelArr = obj.LabelArr;    % Graph coloring label array
            UniqLbls = unique(LabelArr);% Unique labels
            C = obj.C;                  % Total number of unique labels
            
            % Reshape MEAN_OUT and VAR_OUT into N-by-1-by-T arrays
            N = size(NbrList, 1);       % # of spatial nodes (per timestep)
            T = numel(MEAN_OUT) / N;      % # of timesteps
            assert(T - round(T) == 0, ...
                'Gaussian Markov field must contain a total of N*T nodes');
            PRIOR_VAR = TBobj.resize(obj.marginal_var, N, 1, T);
            MEAN_OUT = reshape(MEAN_OUT, N, 1, T);
            VAR_OUT = reshape(VAR_OUT, N, 1, T);
            PrecisionOutGAMP = 1 ./ VAR_OUT;
            BETA = TBobj.resize(obj.beta, N, 1, T);
            GAMMA = TBobj.resize(obj.gamma, N, 1, T);
            
            % Expand certain matrix indexing arrays along the third
            % (temporal) dimension, accounting for shifts in linearly
            % indexed quantities accordingly
            NbrListRep = NaN(N,maxDeg,T);
            RevNbrListRep = NaN(N,maxDeg,T);
            for t = 1:T
                RevNbrListRep(:,:,t) = (t - 1)*N*maxDeg + obj.RevNbrList;
                NbrListRep(:,:,t) = (t - 1)*N + NbrList;
            end
            dummyMaskRep = repmat(obj.dummyMask, [1, 1, T]);
            
            % Initialize diagonal elements of the precision matrix
%             DiagPrecision = (1./VAR_OUT) + obj.beta * sum(dummyMaskRep, 2);
            ALPHA = (1 - (BETA .* sum(dummyMaskRep, 2) + ...
                2*GAMMA) .* PRIOR_VAR) ./ PRIOR_VAR;
            ALPHA(:,1,1) = ALPHA(:,1,1) + GAMMA(:,1,1);     % Correction for t = 1
            ALPHA(:,1,T) = ALPHA(:,1,T) + GAMMA(:,1,T);     % Correction for t = T
            assert(all(ALPHA(:) >= 0), ['Invalid parameterization. ' ...
                'Decrease beta, gamma, or maginal_var'])
            DiagPrecision = ALPHA + BETA .* sum(dummyMaskRep, 2) + ...
                2 * GAMMA;
            DiagPrecision(:,1,1) = DiagPrecision(:,1,1) - GAMMA(:,1,1);
            DiagPrecision(:,1,T) = DiagPrecision(:,1,T) - GAMMA(:,1,T);
            
            % Initialize incoming and outgoing messages.
            SpatialPrecisionIn = zeros(N,maxDeg,T); 	% Inbound spatial precision matrix
            SpatialPrecisionOut = zeros(N,maxDeg,T);    % Outbound spatial precision matrix
            PrevPrecisionIn = zeros(N,1,T);             % Prior timestep inbound precisions
            LatePrecisionIn = zeros(N,1,T);             % Later timestep inbound precisions
            SpatialMeanIn = zeros(N,maxDeg,T);          % Inbound spatial mean matrix
            SpatialMeanOut = zeros(N,maxDeg,T);         % Outbound spatial mean matrix
            PrevMeanIn = zeros(N,1,T);                  % Prior timestep mean precisions
            LateMeanIn = zeros(N,1,T);                  % Later timestep mean precisions
            
            for iter = 1:2*C*obj.maxIter
                % First grab indices of spatial nodes to update this round
                UpdNodes = (LabelArr == UniqLbls(mod(iter-1,C)+1));
                % Next, decide whether to update odd timesteps (UpdTime =
                % 1 : 2 : T), or even nodes (UpdTime = 2 : 2 : T)
                if T > 1
                    UpdTime = (mod(floor((iter - 1)/C), 2) + 1) : 2 : T;
                else
                    UpdTime = 1;    % Only 1 timestep to update
                end
                
                % *************** Spatial message updates *****************
                % Now cycle through each outbound message being dispatched
                % from a given node (many will be outbound to the dummy
                % node, and are therefore irrelevant and will be
                % overwritten)
                SpatialPrecisionOut = SpatialPrecisionIn(RevNbrListRep);
                SpatialMeanOut = SpatialMeanIn(RevNbrListRep);
                for out = 1:maxDeg
                    InclIdx = [1:out-1, out+1:maxDeg];  % Neighbor columns to include
                    
                    % *** Equation references that follow are to the report
                    % by Xinhua Zhang entitled "Message Passing Formula in
                    % Gaussian MRF" ***
                    
                    % Quantity P_0 (defined just before Eq. (3))
                    PrecisionSum = DiagPrecision(UpdNodes,1,UpdTime) + ...
                        sum(SpatialPrecisionIn(UpdNodes,InclIdx,UpdTime), 2) + ...
                        PrevPrecisionIn(UpdNodes,1,UpdTime) + ...
                        LatePrecisionIn(UpdNodes,1,UpdTime) + ...
                        PrecisionOutGAMP(UpdNodes,1,UpdTime);
                    
                    % Quantity \mu_0 (defined just before Eq. (3))
                    MeanSum = (PrecisionOutGAMP(UpdNodes,1,UpdTime) .* ...
                        MEAN_OUT(UpdNodes,1,UpdTime)) + ...
                        sum(SpatialPrecisionIn(UpdNodes,InclIdx,UpdTime).*...
                        SpatialMeanIn(UpdNodes,InclIdx,UpdTime), 2) + ...
                        (PrevPrecisionIn(UpdNodes,1,UpdTime) .* ...
                        PrevMeanIn(UpdNodes,1,UpdTime)) + ...
                        (LatePrecisionIn(UpdNodes,1,UpdTime) .* ...
                        LateMeanIn(UpdNodes,1,UpdTime));
                    MeanSum = MeanSum ./ PrecisionSum;
                    
                    % Quantity P_{ij} (defined after Eq. (6))
                    SpatialPrecisionOut(UpdNodes,out,UpdTime) = ...
                        -BETA(UpdNodes,1,UpdTime).^2 ./ PrecisionSum;
                    
                    % Quantity \mu_{ij} (defined after Eq. (6))
                    SpatialMeanOut(UpdNodes,out,UpdTime) = ...
                        BETA(UpdNodes,1,UpdTime) .* MeanSum ./ ...
                        SpatialPrecisionOut(UpdNodes,out,UpdTime);
                    
                    % If BETA = 0, then 0/0 NaN's can occur.  Fix them
                    BETA_EXTRACT = BETA(UpdNodes,1,UpdTime);
                    if any(BETA_EXTRACT(:)) == 0
                        SpatialMeanOut(isnan(SpatialMeanOut)) = 0;
                    end
                end
                % Replace outgoing messages to dummy nodes with 
                % uninformative values
                SpatialMeanOut(dummyMaskRep) = 0;
                SpatialPrecisionOut(dummyMaskRep) = 0;
                
                % Now copy the outbound messages over to the inbound
                % messages for the next message passing cycle
                SpatialMeanIn = SpatialMeanOut(RevNbrListRep);
                SpatialPrecisionIn = SpatialPrecisionOut(RevNbrListRep);
                
                
                % ************** Temporal message updates *****************
                if T > 1
                    % Start by updating messages coming from previous timesteps
                    UpdTimeP = UpdTime(UpdTime < T);    % No updates leave last time
                    % Quantity P_0 (defined just before Eq. (3))
                    PrecisionSum = DiagPrecision(UpdNodes,1,UpdTimeP) + ...
                        sum(SpatialPrecisionIn(UpdNodes,:,UpdTimeP), 2) + ...
                        PrevPrecisionIn(UpdNodes,1,UpdTimeP) + ...
                        PrecisionOutGAMP(UpdNodes,1,UpdTimeP);
                    % Quantity \mu_0 (defined just before Eq. (3))
                    MeanSum = (PrecisionOutGAMP(UpdNodes,1,UpdTimeP) .* ...
                        MEAN_OUT(UpdNodes,1,UpdTimeP)) + ...
                        sum(SpatialPrecisionIn(UpdNodes,:,UpdTimeP).*...
                        SpatialMeanIn(UpdNodes,:,UpdTimeP), 2) + ...
                        (PrevPrecisionIn(UpdNodes,1,UpdTimeP) .* ...
                        PrevMeanIn(UpdNodes,1,UpdTimeP));
                    MeanSum = MeanSum ./ PrecisionSum;
                    % Quantity P_{ij} (defined after Eq. (6))
                    PrevPrecisionIn(UpdNodes,1,UpdTimeP+1) = ...
                        -GAMMA(UpdNodes,1,UpdTimeP).^2 ./ PrecisionSum;
                    % Quantity \mu_{ij} (defined after Eq. (6))
                    PrevMeanIn(UpdNodes,1,UpdTimeP+1) = ...
                        GAMMA(UpdNodes,1,UpdTimeP) .* MeanSum ./ ...
                        PrevPrecisionIn(UpdNodes,1,UpdTimeP+1);
                    % If GAMMA = 0, then 0/0 NaN's can occur.  Fix them
                    GAM_EXTRACT = GAMMA(UpdNodes,1,UpdTimeP);
                    if any(GAM_EXTRACT(:)) == 0
                        PrevMeanIn(isnan(PrevMeanIn)) = 0;
                    end
                    
                    % Finally, update messages coming from later timesteps
                    UpdTimeL = UpdTime(UpdTime > 1);	% No updates leave initial time
                    % Quantity P_0 (defined just before Eq. (3))
                    PrecisionSum = DiagPrecision(UpdNodes,1,UpdTimeL) + ...
                        sum(SpatialPrecisionIn(UpdNodes,:,UpdTimeL), 2) + ...
                        LatePrecisionIn(UpdNodes,1,UpdTimeL) + ...
                        PrecisionOutGAMP(UpdNodes,1,UpdTimeL);
                    % Quantity \mu_0 (defined just before Eq. (3))
                    MeanSum = (PrecisionOutGAMP(UpdNodes,1,UpdTimeL) .* ...
                        MEAN_OUT(UpdNodes,1,UpdTimeL)) + ...
                        sum(SpatialPrecisionIn(UpdNodes,:,UpdTimeL).*...
                        SpatialMeanIn(UpdNodes,:,UpdTimeL), 2) + ...
                        (LatePrecisionIn(UpdNodes,1,UpdTimeL) .* ...
                        LateMeanIn(UpdNodes,1,UpdTimeL));
                    MeanSum = MeanSum ./ PrecisionSum;
                    % Quantity P_{ij} (defined after Eq. (6))
                    LatePrecisionIn(UpdNodes,1,UpdTimeL-1) = ...
                        -GAMMA(UpdNodes,1,UpdTimeL).^2 ./ PrecisionSum;
                    % Quantity \mu_{ij} (defined after Eq. (6))
                    LateMeanIn(UpdNodes,1,UpdTimeL-1) = ...
                        GAMMA(UpdNodes,1,UpdTimeL) .* MeanSum ./ ...
                        LatePrecisionIn(UpdNodes,1,UpdTimeL-1);
                    % If GAMMA = 0, then 0/0 NaN's can occur.  Fix them
                    GAM_EXTRACT = GAMMA(UpdNodes,1,UpdTimeL);
                    if any(GAM_EXTRACT(:)) == 0
                        LateMeanIn(isnan(LateMeanIn)) = 0;
                    end
                end
            end
            
            
            % Compute extrinsic likelihood (outgoing message)
            ExtrinsicPrecision = sum(SpatialPrecisionIn, 2) + ...
                PrevPrecisionIn + LatePrecisionIn + (1 ./ PRIOR_VAR);
            MeanSum = ( sum(SpatialPrecisionIn .* SpatialMeanIn, 2) + ...
                (PrevPrecisionIn .* PrevMeanIn) + (LatePrecisionIn .* ...
                LateMeanIn) );  % No contribution from prior b/c zero mean
            ExtrinsicMean = MeanSum ./ ExtrinsicPrecision;
            MEAN_IN = ExtrinsicMean(:);     % Vectorize N-by-1-by-T output
            VAR_IN = 1 ./ ExtrinsicPrecision(:);    % Variance = inverse precision
            
            % Lastly, compute the amplitude posterior (a Gaussian)
            PostPrecision = ExtrinsicPrecision + PrecisionOutGAMP;
            PostMean = (MeanSum + PrecisionOutGAMP .* MEAN_OUT) ./ ...
                PostPrecision;
            POST_MEAN = PostMean(:);            % Vectorize N-by-1-by-T output
            POST_VAR = 1 ./ PostPrecision(:);   % Ditto
            
%             % *************************************************************
%             % Compute the quantities that will be used to compute a
%             % mean-field approximation to an expectation-maximization (EM)
%             % update of the MRF parameters, alpha and beta.  In what
%             % follows, we assume S(n) is drawn from {-1,1}
%             PI = reshape(S_POST, N, 1, T);      % Pr{S(n) = 1 | Y}
%             
%             % Compute a couple different sums that appear often in the
%             % mean-field update expressions
%             ShiftPI = 2*PI - 1;         % 2*pi - 1
%             ShiftPI = [ShiftPI; reshape(zeros(1,T), 1, 1, T)];	% Append dummy node posterior
%             
%             % For indexing reasons, need to append a row of dummy indices
%             % to each timestep of NbrListRep
%             NbrListRep = [NbrListRep; repmat(reshape(N+1 : N+1 : ...
%                 T * (N+1), 1, 1, T), 1, maxDeg)];
%             % \sum_Neigh(n,t) (2*pi(q,t) - 1)
%             NeighborhoodSum = sum(ShiftPI(NbrListRep), 2);
%             NeighborhoodSum = NeighborhoodSum(1:N,1,:);     % Remove dummy row
%             ShiftPI = ShiftPI(1:N,1,:);                     % Remove dummy row
%             
%             % \sum_Neigh(n,t) (2*pi(n,t) - 1) (2*pi(q,t) - 1)
%             AvgNeighborhoodSum = ShiftPI .* NeighborhoodSum;
%             
%             % (2*pi(n,t-1) - 1) + (2*pi(n,t+1) - 1)
%             if T > 1
%                 TemporalSum = circshift(ShiftPI, [0, 0, 1]) + ...
%                     circshift(ShiftPI, [0, 0, -1]);
%                 TemporalSum(:,1,1) = ShiftPI(:,1,2);    % t = 1 only has predecessor
%                 TemporalSum(:,1,T) = ShiftPI(:,1,T-1);  % t = T only has ancestor
%             else
%                 TemporalSum = zeros(N,1,1);
%             end
%             
%             % (2*pi(n,t) - 1) ( (2*pi(n,t-1) - 1) + (2*pi(n,t+1) - 1) )
%             AvgTemporalSum = ShiftPI .* TemporalSum;
%             
%             
%             % Compute parameter updates (will require MATLAB's Optimization
%             % Toolbox).
			if strcmpi(obj.learn_beta, 'timestep') && isscalar(obj.beta)
            	beta_upd = TBobj.resize(obj.beta, 1, 1, T);
			else
				beta_upd = obj.beta;
            end
            gamma_upd = obj.gamma;
%             options = optimset('GradObj', 'on', 'Hessian', ...
%                 'off', 'MaxFunEvals', 1000, 'tolX', 1e-8, 'Display', ...
%                 'notify', 'Algorithm', 'interior-point');
%             lb = [-10*ones(numel(alpha_upd),1); -30*ones(numel(beta_upd),1); -30];
%             ub = [10*ones(numel(alpha_upd),1); 30*ones(numel(beta_upd),1); 30];
%             if any(strcmpi(obj.learn_alpha, {'scalar', 'timestep'})) || ...
%                     any(strcmpi(obj.learn_beta, {'scalar', 'timestep'})) || ...
%                     strcmpi(obj.learn_gamma, 'scalar')
%                 try
%                     [updates, ~, exitFlag] = fmincon(@meanfieldLF, ...
%                         [squeeze(alpha_upd); squeeze(beta_upd); gamma_upd], ...
% 						[], [], [], [], lb, ub, [], options, PI, NeighborhoodSum, ...
%                         AvgNeighborhoodSum, TemporalSum, AvgTemporalSum, ...
% 						[numel(alpha_upd), numel(beta_upd), 1], TBobj);
%                     
%                     if obj.EMcnt >= 0
%                         if strcmpi(obj.learn_alpha, 'scalar')
%                             alpha_upd = updates(1);
%                             idx = 2;
%                         elseif strcmpi(obj.learn_alpha, 'timestep')
%                             alpha_upd = reshape(updates(1:T), [1, 1, T]);
%                             idx = T + 1;
%                         else
%                             idx = 2;
%                         end
%                         if strcmpi(obj.learn_beta, 'scalar')
%                             beta_upd = updates(idx);
%                             idx = idx + 1;
%                         elseif strcmpi(obj.learn_beta, 'timestep')
%                             beta_upd = reshape(updates(idx:idx+T-1), [1, 1, T]);
%                             idx = idx + T;
%                         else
%                             idx = idx + 1;
%                         end
%                         if strcmpi(obj.learn_gamma, 'scalar'), gamma_upd = updates(idx); end
%                     end
%                 catch ME
%                     fprintf('Error updating MRF parameters: %s\n', ...
%                         ME.message)
%                 end
%             end
%             obj.EMcnt = obj.EMcnt + 1;
            
            
            
            % *************************************************************
            % meanfieldLF function
            % *************************************************************
            % This is the mean-field approximation of the likelihood
            % function
            %
            % INPUTS:
            %   params: A Q-by-1 vector of [alpha; beta; gamma] parameter 
			%			values at which the EM cost function and gradient 
			% 			are to be computed, where Q is determined by the
			% 			number of unique alpha and beta values being learned
			% 			(e.g., scalar vs per-timestep parameters)
            %   PI:     An N-by-1-by-T tensor containing Pr{s(n,t) = 1 | Y}
            %   NeighSum:   An N-by-1-by-T tensor whose (n,t)^th
            %               element consists of \sum_q 2*pi(q,t) - 1, where
            %               the sum is over entries, q, that are neighbors
            %               of the (n,t)^th voxel, and pi(q,t) = 
            %               Pr{s(q,t) = 1 | Y}.
            %   AvgNeighSum:    Similar to NeighSum, except that the
            %                   (n,t)^th element is multiplied by
            %                   Pr{s(n,t) = 1 | Y}.
            %   TempSum:    An N-by-1-by-T tensor whose (n,t)^th
            %               element consists of (2*pi(n,t-1) - 1) + 
            %               (2*pi(n,t+1) - 1)
            %   AvgTempSum: Similar to TempSum, except that the (n,t)^th
            %               element is multiplied by Pr{s(n,t) = 1 | Y}.
			%	Dims:		A 1-by-3 vector indicating the number of
			%				unique alpha, beta, and gamma values to learn
			%	TBobj:		A parent TurboOpt object
            %   
            % OUTPUTS:
            %   f:      Cost function at [alpha; beta; gamma]
            %   g:      Gradient of cost function at [alpha; beta; gamma]
            %
            % Coded by: Justin Ziniel, The Ohio State Univ.
            % E-mail: zinielj@ece.osu.edu
            % Last change: 07/16/13
            % Change summary: 
            %       - Created (07/13/13; JAZ)
            % Version 0.2
            
            function [f, g] = meanfieldLF(params, PI, NeighSum, ...
                    AvgNeighSum, TempSum, AvgTempSum, Dims, TBobj)
				% Extract parameter values and size info
                Alpha = reshape(params(1:Dims(1)), [1, 1, Dims(1)]);
                Beta = reshape(params(Dims(1)+1:Dims(1)+Dims(2)), [1, 1, Dims(2)]);
                Gamma = params(Dims(1)+Dims(2)+1);
				[N, ~, T] = size(NeighSum);

				% Expand parameters to N-by-1-by-T dimensions
				Alpha = TBobj.resize(Alpha, N, 1, T);
				Beta = TBobj.resize(Beta, N, 1, T);
				Gamma = TBobj.resize(Gamma, N, 1, T);
                
                % Start by computing the objective function value, which is
                % the posterior expectation of the mean field approximation
                % of the 3D MRF prior
                PosSum = -Alpha + Beta.*NeighSum + Gamma.*TempSum;	% s(n,t) = 1
                ExpPosSum = exp(PosSum);
                ExpNegSum = exp(-PosSum);
                f = Alpha.*(2*PI - 1) - Beta.*AvgNeighSum - Gamma.*AvgTempSum + ...
                    log(ExpPosSum + ExpNegSum);
                f = sum(f(:));
                
                % Next, compute the derivative of the cost function w.r.t.
                % alpha and beta
                g_alpha = (ExpNegSum - ExpPosSum) ./ (ExpNegSum + ExpPosSum);
                g_alpha = (2*PI - 1) + g_alpha;     % deriv of f wrt alpha
                g_beta = (NeighSum.*ExpPosSum - NeighSum.*ExpNegSum) ./ ...
                    (ExpNegSum + ExpPosSum);
                g_beta = -AvgNeighSum + g_beta;     % deriv of f wrt beta
                g_gamma = (TempSum.*ExpPosSum - TempSum.*ExpNegSum) ./ ...
                    (ExpNegSum + ExpPosSum);
                g_gamma = -AvgTempSum + g_gamma;     % deriv of f wrt gamma
                
				g = NaN(sum(Dims),1);	% Allocate space for gradient output
				if Dims(1) == 1
					g(1) = sum(g_alpha(:));
				else
					g(1:T) = squeeze(sum(g_alpha, 1));
				end
				if Dims(2) == 1
					g(Dims(1)+1) = sum(g_beta(:));
				else
					g(Dims(1)+1:Dims(1)+T) = squeeze(sum(g_beta, 1));
				end
				g(end) = sum(g_gamma(:));
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
