% CLASS: GaussMarkov
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: AmplitudeStruct
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class, which is a sub-class of the AmplitudeStruct class, can be 
%   used to define the parameters of a first-order Gauss-Markov random
%   process that describes the statistics of the amplitudes of the non-zero
%   elements of the signal matrix, X, along either columns or rows.  
%   Specifically, let X be an N-by-T signal matrix, and let GAMMA be an
%   N-by-T matrix of amplitudes.  Conditioned on a particular coefficient,
%   X(n,t), being non-zero, we enforce that X(n,t) = GAMMA(n,t).  In other
%   words, pdf(X(n,t) | X(n,t)~=0, GAMMA(n,t)) = delta(X(n,t) - GAMMA(n,t))
%   where delta(.) is the Dirac delta function.  Next, suppose that the 
%   amplitudes in GAMMA exhibit a positive correlation across either rows
%   or columns that is well-described by a first-order Gauss-Markov
%   random process.  Then an object of this class can be created and used 
%   anywhere an AmplitudeStruct object is used in order to specify the 
%   parameters of the Gauss-Markov process.
%   
%   We uniquely specify a Gauss-Markov random process through the use of
%   three parameters: alpha, PHI, and THETA.  With these parameters, the
%   dynamical model of the random process is as follows:  If each row of 
%   GAMMA forms a Gauss-Markov process (across columns), then we assume
%   GAMMA(n,t) = (1 - alpha(n))*GAMMA(n,t-1) + alpha(n)*E(n,t) + ...
%                alpha(n)*THETA(n),
%   where pdf(E(n,t)) = Normal(0, Rho(n)).  If each column of GAMMA forms a
%   Gauss-Markov process, then we assume
%   GAMMA(n,t) = (1 - alpha(t))*GAMMA(n-1,t) + alpha(t)*E(n,t) + ...
%                alpha(t)*THETA(t),
%   where pdf(E(n,t)) = Normal(0, Rho(t)).  alpha is a vector that defines 
%   the amount of correlation between neighboring elements of GAMMA, and 
%   its values are always in the interval [0,1].  A value of alpha << 1 
%   implies high correlation between neighboring elements of GAMMA.  Rho is
%   a vector whose entries define the variances of the Gaussian 
%   perturbation process, E(n,t), and it is set automatically based on the
%   value of alpha to ensure that the steady-state variance of the
%   Gauss-Markov process is PHI.  Note that the parameters THETA and PHI
%   are not specified as properties in this class, but are instead passed
%   to GaussMarkov class methods as needed.
%   
%   The property "dim", which specifies whether each row or each column of 
%   GAMMA forms a Gauss-Markov chain, holds the character string 'row' or 
%   'col' respectively.
%
%   If the user would like an expectation-maximization (EM) algorithm to
%   attempt to learn a scalar value of alpha from the data, then the
%   property "learn_alpha" should be set to 'true' (the default choice), 
%   otherwise, set it to 'false'.
%
%   If the user would like a scalar value of alpha to be initialized
%   automatically from the data, set the property "init_params" to 'true'.
%
%   To create a GaussMarkov object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   GaussMarkov(), will create a GaussMarkov object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values, by 
%   using MATLAB's property/value string pairs convention, e.g.,
%   GaussMarkov('alpha', 0.01, 'learn_alpha', 'false') will
%   construct a GaussMarkov object in which alpha(n) (since "dim" defaults
%   to 'row') = 0.01 for all n, and it will not be refined by the EM 
%   parameter learning procedure.  Any parameter not explicitly set in the
%   constructor will be set to its default value.
%
%   ** Note that this class assumes that there exist properties in the 
%   Signal class object called active_mean and active_var, which contain
%   the prior means and variances of the GAMMA variables **
%
% PROPERTIES (State variables)
%   alpha                   A scalar, or a length-T or length-N vector that
%                           specifies the correlation in the Gauss-Markov
%                           process across rows or columns [Default: 0.10]
%   learn_alpha           	Learn alpha using an EM algorithm?  (See 
%                           DESCRIPTION for options)  [Default: 'true']
%   dim                     Is each row of GAMMA a Gauss-Markov process
%                           ('row'), or each column ('col')? [Default: 
%                           'row']
%   init_params             Initialize alpha automatically using Y and A 
%                           ('true'), or not ('false'). [Default: 'false']
%
% METHODS (Subroutines/functions)
%   GaussMarkov()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   GaussMarkov('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class GaussMarkov, obj, has been constructed
%   print()
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'GM' when obj is a GaussMarkov
%         object
%   GaussMarkovCopyObj = copy(obj)
%       - Creates an independent copy of the GaussMarkov object, obj
%   [ETA_IN, KAPPA_IN, POST_MEAN, POST_VAR] = UpdateAmplitude(obj, ...
%           TBobj, ETA_OUT, KAPPA_OUT)
%       - Provides the Signal class object that called this method with
%         updated messages about the active means and variances that are
%         needed to generate new EstimIn and EstimOut objects for GAMP's
%         next iteration, using model parameters stored in the TurboOpt 
%         object, TBobj, along with N-by-T-by-D tensors of means and
%         variances provided in ETA_OUT and KAPPA_OUT respectively.  Third
%         dimension (D) corresponds to the number of distinct Gaussians
%         that could provide X with the active means.  ETA_IN and KAPPA_IN
%         will therefore be of dimension N-by-T-by-D as well. POST_MEAN and 
%         POST_VAR are the posterior means and variances of the amplitude
%         variables (N-by-T-by-D dimensional) [Hidden method]
%   [THETA_UPD, PHI_UPD] = LearnAmplitudeParams(obj, TBobj, varargin)
%       - This method is called by objects of the Signal base class
%         whenever they have apriori means (THETA) and variances (PHI) that
%         they wish to update using an EM algorithm.  THETA_UPD and PHI_UPD
%         will be of dimension N-by-T-by-D. [Hidden method]
%   InitPriors(obj, TBobj, Y, A)
%       - If init_params equals 'true', then initialize alpha as a scalar
%         using simple heuristics involving Y and A. [Hidden method]
%   GAMMA_TRUE = genRand(obj, TBobj, GenParams)
%       - Generate a realization of the Gauss-Markov process using a
%         TurboOpt object (TBobj) and a GenParams object [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'alpha'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Gauss-Markov alpha'), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created (12/10/11; JAZ)
%       - Added genRand method (01/03/12; JAZ)
%       - Implemented EM learning of THETA and PHI (01/13/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef GaussMarkov < AmplitudeStruct

    properties
        % First-order Gauss-Markov process properties
        alpha = 0.10;           % Process correlation = 1 - alpha
        learn_alpha = 'true';	% Learn alpha using EM alg. by default
        dim = 'row';            % Each row forms a Gauss-Markov process
        init_params = 'false';  % Do not initialize from data
    end % properties
    
    properties (Hidden)
        version = 'mmse';
    end
       
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'GM';            % 1st-order Gauss-Markov type identifier
    end
    
    properties (Access = private, Dependent)
        D;      % The number of Gaussian mixture components
    end
    
    properties (Access = private)
        bypass1 = true;      % Don't do EM learning after 1st GAMP run
        bypass2 = true;      % Don't do EM learning after 1st GAMP run
        data = 'real';
    end
    
    methods
        
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = GaussMarkov(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(lower(varargin{i}), varargin{i+1});
                end
            end
        end
        
        
        % *****************************************************************
        %                       SET METHODS
        % *****************************************************************
        
        % Set method for active-to-inactive transition prob (p01)
        function obj = set.alpha(obj, alpha)
            if size(alpha, 1) ~= 1 && size(alpha, 2) ~= 1
                % Either the user, or another method, is attempting to set
                % alpha equal to a matrix (or tensor, if D > 1).  If it is
                % the user, we want to alert them that alpha cannot
                % consist of a matrix of unique values, and that alpha must
                % have uniform entries along either rows or columns
                ST = dbstack;   % Find the calling function
                if sum(strcmpi({'GaussMarkov.UpdateAmplitude', ...
                        'GaussMarkov.InitPriors'}, ST(2).name)) == 0
                    % The set method was called either from the command
                    % window, or a user-created script, thus throw an
                    % exception
                    error(['alpha must be singleton along either the ' ...
                        'first or second dimension'])
                end
            elseif size(alpha, 1) == 1 && size(alpha, 2) > 1
                % Automatically set dim = 'col'
                obj.dim = 'col';
            elseif size(alpha, 1) > 1 && size(alpha, 2) == 1
                % Automatically set dim = 'row'
                obj.dim = 'row';
            end
            if any(alpha(:) < 0) || any(alpha(:) > 1)
                error('alpha must be in the interval [0,1]')
            end
            obj.alpha = alpha;
        end
        
        % Set method for learn_alpha
        function obj = set.learn_alpha(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: learn_alpha')
            end
            obj.learn_alpha = lower(string);
        end
        
        % Set method for dim
        function obj = set.dim(obj, string)
            if sum(strcmpi(string, {'row', 'col'})) == 0
                error('Invalid option: dim')
            end
            if size(obj.alpha, 1) > 1 && strcmpi(string, 'col')
                error(['alpha cannot have a non-singleton first ' ...
                    'dimension if dim = ''col'''])
            elseif size(obj.alpha, 2) > 1 && strcmpi(string, 'row')
                error(['alpha cannot have a non-singleton second ' ...
                    'dimension if dim = ''row'''])
            end
            obj.dim = lower(string);
        end
        
        % Set method for init_params
        function obj = set.init_params(obj, string)
            if sum(strcmpi(string, {'true', 'false'})) == 0
                error('Invalid option: init_params')
            end
            obj.init_params = lower(string);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('GaussMarkov does not support max-sum GAMP')
            else
                error('Invalid option: version')
            end
        end
        
        % *****************************************************************
        %                           GET METHOD
        % *****************************************************************
        function D = get.D(obj)
            D = size(obj.alpha, 3);     % # of Gaussian mixture components
        end
        
        
        % *****************************************************************
        %                           PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('AMPLITUDE STRUCTURE: Gauss-Markov\n')
            switch obj.dim
                case 'row'
                    fprintf('   Structure of each row: 1st-order Gauss-Markov process\n')
                    fprintf('Structure of each column: No structure\n')
                case 'col'
                    fprintf('Structure of each column: 1st-order Gauss-Markov process\n')
                    fprintf('Structure of each row: No structure\n')
            end
            fprintf('                   alpha: %s\n', ...
                form(obj, obj.alpha))
            fprintf('             learn_alpha: %s\n', obj.learn_alpha)
            fprintf('             init_params: %s\n', obj.init_params)
            fprintf('# of component Gaussians: %d\n', obj.D)
        end
        
        
        % *****************************************************************
        %                           COPY METHOD
        % *****************************************************************
        
        % Create an independent copy of a GaussMarkov object
        function GaussMarkovCopyObj = copy(obj)
            GaussMarkovCopyObj = GaussMarkov('alpha', obj.alpha, ...
                'learn_alpha', obj.learn_alpha, 'dim', obj.dim, ...
                'init_params', obj.init_params);
        end
        
        
        % *****************************************************************
        %                       ACCESSORY METHODS
        % *****************************************************************
        
        % This function allows one to query which type of signal family
        % this class is by returning a character string type identifier
        function type_id = get_type(obj)
            type_id = obj.type;
        end            
    end % methods
    
    methods (Hidden)
        
        % *****************************************************************
        %           	 AMPLITUDE MESSAGES UPDATE METHOD
        % *****************************************************************
        function [ETA_IN, KAPPA_IN, POST_MEAN, POST_VAR] = ...
                UpdateAmplitude(obj, TBobj, ETA_OUT, KAPPA_OUT)
            
            % Start by performing the sum-product forward/backward pass for
            % each of the component Gaussians simultaneously
            [ETA_IN, KAPPA_IN, MsgStateStruct, POST_MEAN, POST_VAR] = ...
                fwd_bwd_msg_pass(obj, TBobj, ETA_OUT, KAPPA_OUT);
            
            % Now, if user has specified EM parameter estimation of alpha,
            % do so now
            if strcmpi('true', obj.learn_alpha)
                obj.alpha = alpha_em_update(obj, TBobj, MsgStateStruct);
            end
            
            % Update the Signal class model parameters using the built-in
            % EM algorithm in this class file
            obj.LearnAmplitudeParams(TBobj, MsgStateStruct);
        end
        
        
        % *****************************************************************
        %                   INITIALIZE ALPHA METHOD
        % *****************************************************************
        
        % This function is called by TurboOpt as it initializes GAMP's
        % EstimIn and EstimOut objects.  In addition, it will initialize
        % model parameters.  At the conclusion of this script, the property
        % alpha should be an N-by-T-by-D, regardless of how it has been 
        % initialized
        function InitPriors(obj, TBobj, Y, A)
            
            % Get size information
            [M, T] = size(Y);
            switch TBobj.commonA
                case true, 
                    [M, N] = A.size();
                case false, 
                    [M, N] = A{1}.size();
            end
            
            % Set the "data" hidden property of this class to match the
            % "data" property contained in the signal class
            try
                switch TBobj.Signal.data
                    case 'real'
                        obj.data = 'real';
                    case 'complex'
                        obj.data = 'complex';
                end
            catch
                ST = dbstack;
                warning(['The %s signal class appears to be missing' ...
                    'the required property "data"'], ST(2).name);
            end
            
            switch obj.init_params
                case 'false'
                    % Use user-specified parameter initialization, and
                    % resize if needed to make it N-by-T-by-D in dimension
                    obj.alpha = TBobj.resize(obj.alpha, N, T, obj.D);
                case 'true'
                    % Make a crude guess of the noise variance, psi
                    psi = norm(Y, 'fro')^2 / (M*T*1001);

                    % Now make a guess of a scalar alpha
                    switch obj.dim
                        case 'row'
                            ytrace = sum(diag(Y(:,1) * Y(:,1)'));
                            yxtrace = abs(sum(diag(Y(:,1) * Y(:,2)')));
                        case 'col'
                            ytrace = sum(diag(Y(1,:) * Y(1,:)'));
                            yxtrace = abs(sum(diag(Y(1,:)*Y(2,:)')));
                    end
                    alpha_init = max(min(1 - yxtrace/(ytrace - M*psi), ...
                        0.5), 0.025);

                    TBobj.AmplitudeStruct.alpha = ...
                        TBobj.resize(alpha_init, N, T, obj.D);
            end
        end
        
        
        % *****************************************************************
        %           GENERATE GAUSS-MARKOV PROCESS REALIZATION
        % *****************************************************************
        
        function GAMMA_TRUE = genRand(obj, TBobj, GenParams)
            % Get size information
            N = GenParams.N;
            T = GenParams.T;
            D = obj.D;
            
            % Resize relevant model parameters
            THETA = TBobj.resize(TBobj.Signal.active_mean, N, T, D);
            PHI = TBobj.resize(TBobj.Signal.active_var, N, T, D);
            ALPHA = TBobj.resize(obj.alpha, N, T, D);
            
            GAMMA_TRUE = NaN([N, T, D]);
            RHO = (2 - ALPHA) .* PHI ./ ALPHA;
            
            % Initialize the Gauss-Markov processes
            switch obj.dim
                case 'row'   % Each row forms a Gauss-Markov process
                    switch GenParams.type
                        case 'real'
                            GAMMA_TRUE(:,1,:) = THETA(:,1,:) + ...
                                sqrt(PHI(:,1,:)) .* randn(N,1,D);
                        case 'complex'
                            GAMMA_TRUE(:,1,:) = THETA(:,1,:) + ...
                                sqrt(PHI(:,1,:)/2) .* randn(N,1,D) + ...
                                1j*sqrt(PHI(:,1,:)/2) .* randn(N,1,D);
                    end
                case 'col'  % Each column forms a Gauss-Markov process
                    switch GenParams.type
                        case 'real'
                            GAMMA_TRUE(1,:,:) = THETA(1,:,:) + ...
                                sqrt(PHI(1,:,:)) .* randn(1,T,D);
                        case 'complex'
                            GAMMA_TRUE(1,:,:) = THETA(1,:,:) + ...
                                sqrt(PHI(1,:,:)/2) .* randn(1,T,D) + ...
                                1j*sqrt(PHI(1,:,:)/2) .* randn(1,T,D);
                    end
            end
            
            % Evolve the Gauss-Markov processes
            switch obj.dim
                case 'row'  % Each row forms a Gauss-Markov process
                    for t = 2:T
                        % Unperturbed process
                        GAMMA_TRUE(:,t,:) = (1 - ALPHA(:,t,:)) .* ...
                            GAMMA_TRUE(:,t-1,:) + ALPHA(:,t,:) .* ...
                            THETA(:,t,:);
                        % Now add perturbation
                        switch GenParams.type
                            case 'real'
                                GAMMA_TRUE(:,t,:) = GAMMA_TRUE(:,t,:) + ...
                                    ALPHA(:,t,:) .* sqrt(RHO(:,t,:)) .* ...
                                    randn(N,1,D);
                            case 'complex'
                                GAMMA_TRUE(:,t,:) = GAMMA_TRUE(:,t,:) + ...
                                    ALPHA(:,t,:) .* (sqrt(RHO(:,t,:)/2) .* ...
                                    randn(N,1,D) + 1j*sqrt(RHO(:,t,:)/2) .* ...
                                    randn(N,1,D));
                        end
                    end
                case 'col'  % Each column forms a Gauss-Markov process
                    for n = 2:N
                        % Unperturbed process
                        GAMMA_TRUE(n,:,:) = (1 - ALPHA(n,:,:)) .* ...
                            GAMMA_TRUE(n-1,:,:) + ALPHA(n,:,:) .* ...
                            THETA(n,:,:);
                        % Now add perturbation
                        switch GenParams.type
                            case 'real'
                                GAMMA_TRUE(n,:,:) = GAMMA_TRUE(n,:,:) + ...
                                    ALPHA(n,:,:) .* sqrt(RHO(n,:,:)) .* ...
                                    randn(1,T,D);
                            case 'complex'
                                GAMMA_TRUE(n,:,:) = GAMMA_TRUE(n,:,:) + ...
                                    ALPHA(n,:,:) .* (sqrt(RHO(n,:,:)/2) .* ...
                                    randn(N,1,D) + 1j*sqrt(RHO(n,:,:)/2) .* ...
                                    randn(N,1,D));
                        end
                    end
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = double(strcmpi(obj.learn_alpha, 'true'));
            Report = cell(obj.D*Nparam, 3);   % Declare Report array
            Params = {  'alpha',    'Gauss-Markov alpha',	'learn_alpha'};
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
        % *****************************************************************
        %                     	HELPER METHODS
        % *****************************************************************
        
        % FWD_BWD_MSG_PASS              Method for performing a single
        % forward/backward pass of the sum-product algorithm for amplitude
        % variables that form Gauss-Markov processes across either rows or
        % columns of the N-by-T amplitude matrices.  If there is more than
        % one set of amplitude variables from which the active elements of
        % X are drawn, and they all behave as independent Gauss-Markov
        % processes, then this method can be used to perform simultaneous
        % inference on each of the component Gaussians.  To do so, all
        % inputs related to the amplitude variables, e.g., ETA_OUT,
        % KAPPA_OUT, alpha, apriori means and variances, etc., should
        % be N-by-T-by-D tensors, where D is the number of component
        % Gaussians.
        %
        % SYNTAX:
        % [ETA_IN, KAPPA_IN, MsgStateStruct, POST_MEAN, POST_VAR] = ...
        %       fwd_bwd_msg_pass(obj, TBobj, ETA_OUT, KAPPA_OUT)
        %
        % INPUTS:
        % obj               An object of the GaussMarkov class
        % TBobj             An object of the TurboOpt class
        % ETA_OUT           An N-by-T-by-D tensor of incoming messages from
        %                   the GAMP factor nodes to the amplitude variable
        %                   nodes that contains the means of the Gaussian
        %                   messages.  Here D is the number of component
        %                   Gaussians that collectively form the "active"
        %                   distributions for the N-by-T signal matrix X
        % KAPPA_OUT         An N-by-T-by-D tensor of incoming messages from
        %                   the GAMP factor nodes to the amplitude variable
        %                   nodes that contains the variances of the 
        %                   Gaussian messages.  Here D is the number of 
        %                   component Gaussians that collectively form the 
        %                   "active" distributions for the N-by-T signal 
        %                   matrix X
        %
        % OUTPUTS:
        % ETA_IN            An N-by-T-by-D tensor of outgoing messages from
        %                   the amplitude variable nodes to the GAMP factor
        %                   nodes that contains the means of the Gaussian
        %                   messages
        % KAPPA_IN          An N-by-T-by-D tensor of outgoing messages from
        %                   the amplitude variable nodes to the GAMP factor
        %                   nodes that contains the variances of the
        %                   Gaussian messages
        % MsgStateStruct    Structure of messages that are useful for EM
        %                   parameter update procedure for alpha
        % POST_MEAN         An N-by-T-by-D tensor of posterior means
        % POST_VAR          An N-by-T-by-D tensor of posterior variances
        %
        function [ETA_IN, KAPPA_IN, MsgStateStruct, POST_MEAN, POST_VAR] = ...
                fwd_bwd_msg_pass(obj, TBobj, ETA_OUT, KAPPA_OUT)
            
            % Begin by getting size information, and verifying validity of
            % the ETA_OUT and KAPPA_OUT inputs
            [N, T, D] = size(ETA_OUT);      % # of Gaussian components: D
            
            if D ~= obj.D
                error(['The number of Gaussian components declared in ' ...
                    'the Signal sub-class does not agree with the ' ...
                    'number declared in the GaussMarkov class'])
            end
            
            if any(KAPPA_OUT(:) < 0) || ~isreal(KAPPA_OUT)
                error('Invalid input: KAPPA_OUT')
            end
            
            % Now get the steady-state means and variances for each
            % component Gaussian.  The property names under which these
            % variables are stored will depend on the signal prior, so we
            % differentiate here
            switch TBobj.Signal.get_type()
                case {'BG', 'MoG'}
                    % Bernoulli-Gaussian or Bernoulli/Mixture-of-Gaussians
                    THETA = TBobj.Signal.active_mean;
                    PHI = TBobj.Signal.active_var;
                otherwise
                    error('Unrecognized signal prior type')
            end
            
            if size(THETA, 3) ~= obj.D
                error(['The number of Gaussian components declared in ' ...
                    'the Signal sub-class does not agree with the ' ...
                    'number declared in the GaussMarkov class'])
            end
            
            % Now we need to compute the perturbation process variance RHO,
            % which is set based on the values of alpha and PHI to ensure
            % the process has a steady-state variance of PHI
            RHO = (2 - obj.alpha) .* PHI ./ obj.alpha;
            
            % Initialize certain initial and terminal messages
            THETA_FWD = NaN([N, T, D]);
            PHI_FWD = NaN([N, T, D]);
            THETA_BWD = NaN([N, T, D]);
            PHI_BWD = NaN([N, T, D]);
            switch obj.dim
                case 'row'  % Each row is a Gauss-Markov process
                    THETA_FWD(:,1,:) = THETA(:,1,:);
                    PHI_FWD(:,1,:) = PHI(:,1,:);
                    THETA_BWD(:,T,:) = zeros([N, 1, D]);
                    PHI_BWD(:,T,:) = inf([N, 1, D]);
                case 'col'  % Each column is a Gauss-Markov process
                    THETA_FWD(1,:,:) = THETA(1,:,:);
                    PHI_FWD(1,:,:) = PHI(1,:,:);
                    THETA_BWD(N,:,:) = zeros([1, T, D]);
                    PHI_BWD(N,:,:) = inf([1, T, D]);
            end
            
            % First execute the forward portion of the forward/backward
            % pass.  Need to differentiate whether the Gauss-Markov process
            % is defined for rows or columns of the amplitude matrix
            % (or matrices, for D > 1)
            switch obj.dim
                case 'row'  % Each row is a Gauss-Markov process
                    for t = 1:T-1
                        TempVar = (1./PHI_FWD(:,t,:) + 1./KAPPA_OUT(:,t,:)).^(-1);
                        THETA_FWD(:,t+1,:) = (1 - obj.alpha(:,t+1,:)) .* ...
                            TempVar .* (THETA_FWD(:,t,:)./PHI_FWD(:,t,:) + ...
                            ETA_OUT(:,t,:)./KAPPA_OUT(:,t,:)) + ...
                            obj.alpha(:,t+1,:) .* THETA(:,t+1,:);
                        PHI_FWD(:,t+1,:) = (1 - obj.alpha(:,t+1,:)).^2 .* ...
                            TempVar + obj.alpha(:,t+1,:).^2 .* RHO(:,t+1,:);
                    end
                case 'col'  % Each column is a Gauss-Markov process
                    for n = 1:N-1
                        TempVar = (1./PHI_FWD(n,:,:) + 1./KAPPA_OUT(n,:,:)).^(-1);
                        THETA_FWD(n+1,:,:) = (1 - obj.alpha(n+1,:,:)) .* ...
                            TempVar .* (THETA_FWD(n,:,:)./PHI_FWD(n,:,:) + ...
                            ETA_OUT(n,:,:)./KAPPA_OUT(n,:,:)) + ...
                            obj.alpha(n+1,:,:) .* THETA(n+1,:,:);
                        PHI_FWD(n+1,:,:) = (1 - obj.alpha(n+1,:,:)).^2 .* ...
                            TempVar + obj.alpha(n+1,:,:).^2 .* RHO(n+1,:,:);
                    end
            end
            
            % Now execute the backward portion of the forward/backward pass
            switch obj.dim
                case 'row'  % Each row is a Gauss-Markov process
                    for t = T:-1:2
                        TempVar = (1./PHI_BWD(:,t,:) + 1./KAPPA_OUT(:,t,:)).^(-1);
                        THETA_BWD(:,t-1,:) = 1./(1 - obj.alpha(:,t-1,:)) .* ...
                            (TempVar .* (THETA_BWD(:,t,:)./PHI_BWD(:,t,:) + ...
                            ETA_OUT(:,t,:)./KAPPA_OUT(:,t,:)) - ...
                            obj.alpha(:,t-1,:) .* THETA(:,t-1,:));
                        PHI_BWD(:,t-1,:) = (1 - obj.alpha(:,t-1,:)).^-2 .* ...
                            (TempVar + obj.alpha(:,t-1,:).^2 .* RHO(:,t-1,:));
                    end
                case 'col'  % Each column is a Gauss-Markov process
                    for n = N:-1:2
                        TempVar = (1./PHI_BWD(n,:,:) + 1./KAPPA_OUT(n,:,:)).^(-1);
                        THETA_BWD(n-1,:,:) = 1./(1 - obj.alpha(n-1,:,:)) .* ...
                            (TempVar .* (THETA_BWD(n,:,:)./PHI_BWD(n,:,:) + ...
                            ETA_OUT(n,:,:)./KAPPA_OUT(n,:,:)) - ...
                            obj.alpha(n-1,:,:) .* THETA(n-1,:,:));
                        PHI_BWD(n-1,:,:) = (1 - obj.alpha(n-1,:,:)).^-2 .* ...
                            (TempVar + obj.alpha(n-1,:,:).^2 .* RHO(n-1,:,:));
                    end
            end
            
            % Now the forward/backward pass is complete.  Combine the
            % forward and backward messages to yield the outgoing messages
            KAPPA_IN = (PHI_FWD.^(-1) + PHI_BWD.^(-1)).^(-1);
            ETA_IN = (THETA_FWD./PHI_FWD + THETA_BWD./PHI_BWD);
            ETA_IN = KAPPA_IN .* ETA_IN;
            
            % Compute the posterior means and variances as well
            POST_VAR = (1./KAPPA_OUT + 1./PHI_FWD + 1./PHI_BWD).^(-1);
            POST_MEAN = (ETA_OUT ./ KAPPA_OUT + THETA_FWD ./ PHI_FWD + ...
                THETA_BWD ./ PHI_BWD);
            POST_MEAN = POST_VAR .* POST_MEAN;
            
            % Finally, pack up a structure that is useful for EM parameter
            % learning
            MsgStateStruct.ETA_OUT = ETA_OUT;
            MsgStateStruct.KAPPA_OUT = KAPPA_OUT;
            MsgStateStruct.PHI_FWD = PHI_FWD;
            MsgStateStruct.PHI_BWD = PHI_BWD;
            MsgStateStruct.RHO = RHO;
            MsgStateStruct.THETA = THETA;
            MsgStateStruct.THETA_BWD = THETA_BWD;
            MsgStateStruct.THETA_FWD = THETA_FWD;
            MsgStateStruct.POST_MEAN = POST_MEAN;
            MsgStateStruct.POST_VAR = POST_VAR;

        end	%fwd_bwd_msg_pass
        
        
        % *****************************************************************
        %           LEARN SIGNAL MODEL PRIOR PARAMETERS METHOD
        % *****************************************************************
        
        function LearnAmplitudeParams(obj, TBobj, MsgStateStruct)
            % Switch based on the type of signal prior
            switch TBobj.Signal.get_type()
                case {'BG', 'MoG'}
                    % Unpack the state of messages
                    ETA_OUT = MsgStateStruct.ETA_OUT;
                    KAPPA_OUT = MsgStateStruct.KAPPA_OUT;
                    PHI_FWD = MsgStateStruct.PHI_FWD;
                    PHI_BWD = MsgStateStruct.PHI_BWD;
                    RHO = MsgStateStruct.RHO;
                    THETA = MsgStateStruct.THETA;
                    THETA_BWD = MsgStateStruct.THETA_BWD;
                    THETA_FWD = MsgStateStruct.THETA_FWD;
                    MU_GAMMA = MsgStateStruct.POST_MEAN;
                    V_GAMMA = MsgStateStruct.POST_VAR;
                    
                    ALPHA = obj.alpha;  % N-by-T-by-D matrix of alphas
                    [N, T, D] = size(MU_GAMMA);
                    
                    % For now, we don't support groups, but keep the group
                    % indexing for future use.  Also check to make sure
                    % that the user's parameter learning preferences make
                    % sense
                    switch obj.dim
                        case 'row'  % Each row forms Gauss-Markov process
                            g_ind = 1:N;        % Group indices (all)
                            Ng = numel(g_ind);
                            AHr = g_ind;    % Ahead row index
                            BHr = g_ind;    % Behind row index
                            AHc = 2:T;      % Ahead column index
                            BHc = 1:T-1;    % Behind column index
                            if strcmp(TBobj.Signal.learn_active_mean, ...
                                    'column') || strcmp('column', ...
                                    TBobj.Signal.learn_active_var)
                                error(['Cannot learn unique active means and ' ...
                                    'variances for each column of X ' ...
                                    'when each row forms a Gauss-Markov ' ...
                                    'process'])
                            end
                        case 'col'  % Each column forms Gauss-Markov process
                            g_ind = 1:T;    % Group indices (all)
                            Ng = numel(g_ind);
                            AHr = 2:N;      % Ahead row index
                            BHr = 1:N-1;    % Behind row index
                            AHc = g_ind;   	% Ahead column index
                            BHc = g_ind;    % Behind column index
                            if strcmp(TBobj.Signal.learn_active_mean, ...
                                    'row') || strcmp('row', ...
                                    TBobj.Signal.learn_active_var)
                                error(['Cannot learn unique active means and ' ...
                                    'variances for each row of X ' ...
                                    'when each column forms a Gauss-Markov ' ...
                                    'process'])
                            end
                    end
                    
                    if ~strcmp(TBobj.Signal.learn_active_var, 'false')
                    % ---------------------------------------------
                    %       Updated active variance, PHI
                    % ---------------------------------------------
                    % Start by computing E[GAMMA(n,t)'*GAMMA(n,t-1)|Y] (or 
                    % E[GAMMA(n,t)'*GAMMA(n-1,t)|Y])
                    Q = (1./KAPPA_OUT(AHr,AHc,:) + 1./PHI_BWD(AHr,AHc,:)).^(-1);
                    R = (ETA_OUT(AHr,AHc,:)./KAPPA_OUT(AHr,AHc,:) + ...
                        THETA_BWD(AHr,AHc,:)./PHI_BWD(AHr,AHc,:));
                    Q_BAR = (1./KAPPA_OUT(BHr,BHc,:) + 1./PHI_FWD(BHr,BHc,:)).^(-1);     
                    R_BAR = (ETA_OUT(BHr,BHc,:)./KAPPA_OUT(BHr,BHc,:) + ...
                        THETA_FWD(BHr,BHc,:)./PHI_FWD(BHr,BHc,:));        
                    Q_TIL = (1./Q_BAR + ((1-ALPHA(BHr,BHc,:)).^2)./(Q + ...
                        (ALPHA(BHr,BHc,:).^2 .* RHO(BHr,BHc,:)))).^(-1);        
                    M_BAR = (1 - ALPHA(BHr,BHc,:)).*(Q.*R - ...
                        (ALPHA(BHr,BHc,:).*THETA(BHr,BHc,:))) ./ (Q + ...
                        (ALPHA(BHr,BHc,:).^2 .* RHO(BHr,BHc,:))) + R_BAR;        
                    GAMMA_CORR = (Q./(Q + (ALPHA(BHr,BHc,:).^2.*RHO(BHr,BHc,:)))) .* ...
                        ((1-ALPHA(BHr,BHc,:)).*(Q_TIL + abs(Q_TIL.*M_BAR).^2) + ...
                        (ALPHA(BHr,BHc,:).*THETA(BHr,BHc,:)).*Q_TIL.*conj(M_BAR) + ...
                        (ALPHA(BHr,BHc,:).^2.*RHO(BHr,BHc,:)).*Q_TIL.*conj(M_BAR).*R);
                    
                    % Now compute E[|GAMMA(n,t) - (1-alpha)*GAMMA(n,t-1) -
                    % alpha*THETA|^2 | Y] (or similar for column GM process)
                    E1 = V_GAMMA(AHr,AHc,:) + abs(MU_GAMMA(AHr,AHc,:)).^2 - ...
                        2*(1 - ALPHA(BHr,BHc,:)) .* real(GAMMA_CORR) - ...
                        2*ALPHA(BHr,BHc,:) .* real(conj(THETA(AHr,AHc,:)) .* ...
                        MU_GAMMA(AHr,AHc,:)) + (1 - ALPHA(BHr,BHc,:)).^2 .* ...
                        (V_GAMMA(BHr,BHc,:) + abs(MU_GAMMA(BHr,BHc,:)).^2) + ...
                        2*ALPHA(BHr,BHc,:) .* (1 - ALPHA(BHr,BHc,:)) .* ...
                        real(conj(THETA(BHr,BHc,:)) .* MU_GAMMA(BHr,BHc,:)) + ...
                        ALPHA(BHr,BHc,:).^2 .* abs(THETA(BHr,BHc,:)).^2;
                    
                    % Now compute E[|GAMMA(n,0) - THETA|^2 | Y], (or
                    % similar for column GM process)
                    switch obj.dim
                        case 'row'  % Each row forms Gauss-Markov process
                            E2 = V_GAMMA(g_ind,1,:) + ...
                                abs(MU_GAMMA(g_ind,1,:)).^2 - ...
                                2*real(conj(THETA(g_ind,1,:)) .* ...
                                MU_GAMMA(g_ind,1,:)) + abs(THETA(g_ind,1,:)).^2;
                        case 'col'  % Each column forms Gauss-Markov process
                            E2 = V_GAMMA(1,g_ind,:) + ...
                                abs(MU_GAMMA(1,g_ind,:)).^2 - ...
                                2*real(conj(THETA(1,g_ind,:)) .* ...
                                MU_GAMMA(1,g_ind,:)) + abs(THETA(1,g_ind,:)).^2;
                    end
                    end     % Even bother computing these quantities?
                    
                    % We are now ready to compute the update to PHI.
                    switch TBobj.Signal.learn_active_var
                        case 'false'
                            PHI_UPD = TBobj.Signal.active_var;
                        case 'scalar'
                            switch obj.dim
                                case 'row'
                                    PHI_UPD = 1 ./ (Ng*T*ALPHA(1,1,:).* ...
                                        (2 - ALPHA(1,1,:))) .* ...
                                        sum(sum(E1, 1), 2) + (1 / (Ng*T))* ...
                                        sum(E2, 1);
                                case 'col'
                                    PHI_UPD = 1 ./ (Ng*N*ALPHA(1,1,:).* ...
                                        (2 - ALPHA(1,1,:))) .* ...
                                        sum(sum(E1, 1), 2) + (1 / (Ng*N))* ...
                                        sum(E2, 2);
                            end
                        case 'row'
                            PHI_UPD = 1 ./ (T*ALPHA(:,1,:).* ...
                                        (2 - ALPHA(:,1,:))) .* ...
                                        sum(E1, 2) + (1/T)*E2;
                        case 'column'
                            PHI_UPD = 1 ./ (N*ALPHA(1,:,:).* ...
                                        (2 - ALPHA(1,:,:))) .* ...
                                        sum(E1, 1) + (1/N)*E2;
                    end
                    
                    if any(PHI_UPD(:) <= 0)
                        warning(['Negative EM update for active variance. ' ...
                            'Returning previous value instead'])
                        PHI_UPD = TBobj.Signal.active_var;
                    end
                    
                    if obj.bypass1	% Bypass updates on the first GAMP run
                        PHI_UPD = TBobj.Signal.active_var;
                        obj.bypass1 = false;
                    end
                    
                    % Pass along the updated estimate of PHI
                    TBobj.Signal.active_var = TBobj.resize(PHI_UPD, N, T, D);
                    PHI = TBobj.Signal.active_var;

                    % Now, update "current" value of RHO for subsequent
                    % EM estimator
                    RHO = (2 - ALPHA) .* PHI ./ ALPHA;


                    % ------------------------------------
                    %   Updated amplitude mean, THETA
                    % ------------------------------------
                    % First compute E[GAMMA(n,t) - (1-alpha)*THETA(n,t-1) |
                    % Y] * (1/alpha/rho)
                    E1 = MU_GAMMA(AHr,AHc,:) - (1 - ALPHA(BHr,BHc,:)) .* ...
                        MU_GAMMA(BHr,BHc,:);
                    E1 = (1./(ALPHA(BHr,BHc,:) .* RHO(BHr,BHc,:))) .* E1;
                    
                    % Now do the EM update of THETA
                    switch TBobj.Signal.learn_active_mean
                        case 'false'
                            THETA_UPD = TBobj.Signal.active_mean;
                        case 'scalar'
                            switch obj.dim
                                case 'row'  % Each row forms GM process
                                    THETA_UPD = sum(sum(E1, 1), 2) + ...
                                        sum(MU_GAMMA(g_ind,1,:) ./ ...
                                        PHI(g_ind,1,:), 1);
                                    THETA_UPD = THETA_UPD ./ ...
                                        ( sum(sum(1./RHO(BHr,BHc,:), 1), 2) + ...
                                        sum(1./PHI(g_ind,1,:), 1) );
                                case 'col'  % Each column forms GM process
                                    THETA_UPD = sum(sum(E1, 1), 2) + ...
                                        sum(MU_GAMMA(1,g_ind,:) ./ ...
                                        PHI(1,g_ind,:), 2);
                                    THETA_UPD = THETA_UPD ./ ...
                                        ( sum(sum(1./RHO(BHr,BHc,:), 1), 2) + ...
                                        sum(1./PHI(1,g_ind,:), 2) );
                            end
                        case 'row'
                            THETA_UPD = sum(E1, 2) + (MU_GAMMA(g_ind,1,:) ./ ...
                                PHI(g_ind,1,:));
                            THETA_UPD = THETA_UPD ./ ...
                                ( sum(1./RHO(BHr,BHc,:), 2) + 1./PHI(g_ind,1,:) );
                        case 'column'
                            THETA_UPD = sum(E1, 1) + (MU_GAMMA(1,g_ind,:) ./ ...
                                PHI(1,g_ind,:));
                            THETA_UPD = THETA_UPD ./ ...
                                ( sum(1./RHO(BHr,BHc,:), 1) + 1./PHI(1,g_ind,:) );
                    end
                    
                    if obj.bypass2	% Bypass updates on the first GAMP run
                        THETA_UPD = TBobj.Signal.active_mean;
                        obj.bypass2 = false;
                    end
                    
                    % Pass along the updated estimate of THETA
                    TBobj.Signal.active_mean = TBobj.resize(THETA_UPD, N, T, D);
                    
                otherwise
                    ST = dbstack;
                    warning(['The GaussMarkov class does not support EM ' ...
                        'parameter learning for the %s signal class'], ...
                        ST(2).name)
            end
        end
        
        
        % ALPHA_EM_UPDATE           Function for updating the Gauss-Markov
        % process innovation rate, alpha, using an EM learning procedure
        %
        % OUTPUTS:
        % Alpha_upd         A 1-by-1-by-D EM update of the Gauss-Markov 
        %                   process correlation parameter(s), one for each
        %                   Gaussian active component
        %
        function Alpha_upd = alpha_em_update(obj, TBobj, MsgStateStruct)
            
            % Start by unpacking the MsgStateStruct structure
            ETA_OUT = MsgStateStruct.ETA_OUT;
            KAPPA_OUT = MsgStateStruct.KAPPA_OUT;
            PHI_FWD = MsgStateStruct.PHI_FWD;
            PHI_BWD = MsgStateStruct.PHI_BWD;
            RHO = MsgStateStruct.RHO;
            THETA = MsgStateStruct.THETA;
            THETA_BWD = MsgStateStruct.THETA_BWD;
            THETA_FWD = MsgStateStruct.THETA_FWD;
            V_GAMMA = MsgStateStruct.POST_VAR;
            MU_GAMMA = MsgStateStruct.POST_MEAN;
            
            % Get size information
            [N, T, D] = size(ETA_OUT);
            
            % This function could support operations on different groups,
            % (i.e., different alpha's for different subsets of indices),
            % but for now we are going to use only a single group
            switch obj.dim
                case 'row'  % Each row is a Gauss-Markov process
                    g_ind = 1:N;    % Indices of group members
                    N_g = N;        % Number of group members
                case 'col'  % Each column is a Gauss-Markov process
                    g_ind = 1:T;    % Indices of group members
                    N_g = T;        % Number of group members
            end
            
            Alpha = obj.alpha;  % Current N-by-T-by-D alpha matrix
            Alpha_upd = Alpha;
            
            for h = 1:D     % Each component Gaussian separately
                
                % Start by computing E[GAMMA(n,t)'*GAMMA(n,t-1)|Y] (or 
                % E[GAMMA(n,t)'*GAMMA(n-1,t)|Y], depending)
                switch obj.dim
                    case 'row'  % Each row is a Gauss-Markov process
                        Q = (1./KAPPA_OUT(g_ind,2:T,h) + 1./PHI_BWD(g_ind,2:T,h)).^(-1);
                        R = (ETA_OUT(g_ind,2:T,h)./KAPPA_OUT(g_ind,2:T,h) + ...
                            THETA_BWD(g_ind,2:T,h)./PHI_BWD(g_ind,2:T,h));
                        Q_BAR = (1./KAPPA_OUT(g_ind,1:T-1,h) + 1./PHI_FWD(g_ind,1:T-1,h)).^(-1);     
                        R_BAR = (ETA_OUT(g_ind,1:T-1,h)./KAPPA_OUT(g_ind,1:T-1,h) + ...
                            THETA_FWD(g_ind,1:T-1,h)./PHI_FWD(g_ind,1:T-1,h));        
                        Q_TIL = (1./Q_BAR + ((1-Alpha(g_ind,1:T-1,h)).^2)./(Q + ...
                            (Alpha(g_ind,1:T-1,h).^2.*RHO(g_ind,1:T-1,h)))).^(-1);        
                        M_BAR = (1 - Alpha(g_ind,1:T-1,h)).*(Q.*R - ...
                            (Alpha(g_ind,1:T-1,h).*THETA(g_ind,1:T-1,h))) ./ (Q + ...
                            (Alpha(g_ind,1:T-1,h).^2.*RHO(g_ind,1:T-1,h))) + R_BAR;        
                        GAMMA_CORR = (Q./(Q + (Alpha(g_ind,1:T-1,h).^2.*RHO(g_ind,1:T-1,h)))) .* ...
                            (((1-Alpha(g_ind,1:T-1,h))).*(Q_TIL + abs(Q_TIL.*M_BAR).^2) + ...
                            ((Alpha(g_ind,1:T-1,h).*THETA(g_ind,1:T-1,h))).*Q_TIL.*conj(M_BAR) + ...
                            ((Alpha(g_ind,1:T-1,h).^2.*RHO(g_ind,1:T-1,h))).*Q_TIL.*conj(M_BAR).*R);
                    case 'col'  % Each column is a Gauss-Markov process
                        Q = (1./KAPPA_OUT(2:N,g_ind,h) + 1./PHI_BWD(2:N,g_ind,h)).^(-1);
                        R = (ETA_OUT(2:N,g_ind,h)./KAPPA_OUT(2:N,g_ind,h) + ...
                            THETA_BWD(2:N,g_ind,h)./PHI_BWD(2:N,g_ind,h));
                        Q_BAR = (1./KAPPA_OUT(1:N-1,g_ind,h) + 1./PHI_FWD(1:N-1,g_ind,h)).^(-1);     
                        R_BAR = (ETA_OUT(1:N-1,g_ind,h)./KAPPA_OUT(1:N-1,g_ind,h) + ...
                            THETA_FWD(1:N-1,g_ind,h)./PHI_FWD(1:N-1,g_ind,h));        
                        Q_TIL = (1./Q_BAR + ((1-Alpha(1:N-1,g_ind,h)).^2)./(Q + ...
                            (Alpha(1:N-1,g_ind,h).^2.*RHO(1:N-1,g_ind,h)))).^(-1);        
                        M_BAR = (1 - Alpha(1:N-1,g_ind,h)).*(Q.*R - ...
                            (Alpha(1:N-1,g_ind,h).*THETA(1:N-1,g_ind,h))) ./ (Q + ...
                            (Alpha(1:N-1,g_ind,h).^2.*RHO(1:N-1,g_ind,h))) + R_BAR;        
                        GAMMA_CORR = (Q./(Q + (Alpha(1:N-1,g_ind,h).^2.*RHO(1:N-1,g_ind,h)))) .* ...
                            (((1-Alpha(1:N-1,g_ind,h))).*(Q_TIL + abs(Q_TIL.*M_BAR).^2) + ...
                            ((Alpha(1:N-1,g_ind,h).*THETA(1:N-1,g_ind,h))).*Q_TIL.*conj(M_BAR) + ...
                            ((Alpha(1:N-1,g_ind,h).^2.*RHO(1:N-1,g_ind,h))).*Q_TIL.*conj(M_BAR).*R);
                end

                % Compute the values of the coefficients of the cubic polynomial
                % whose solution gives the appropriate value of alpha
                switch obj.dim
                    case 'row'  % Each row is a Gauss-Markov process
                        mult_a = -2*N_g*(T-1);
                        mult_b = (RHO(g_ind,2:T,h)).^(-1) .* (2*real(GAMMA_CORR) - ...
                            2*real(conj(THETA(g_ind,2:T,h)).*MU_GAMMA(g_ind,2:T,h)) - ...
                            2*(V_GAMMA(g_ind,1:T-1,h) + abs(MU_GAMMA(g_ind,1:T-1,h)).^2) + ...
                            2*real(conj(THETA(g_ind,1:T-1,h)).*MU_GAMMA(g_ind,1:T-1,h)));
                        mult_b = sum(sum(mult_b));
                        mult_c = (RHO(g_ind,2:T,h)).^(-1) .* ((V_GAMMA(g_ind,2:T,h) + ...
                            abs(MU_GAMMA(g_ind,2:T,h)).^2) - 2*real(GAMMA_CORR) + ...
                            (V_GAMMA(g_ind,1:T-1,h) + abs(MU_GAMMA(g_ind,1:T-1,h)).^2));
                        mult_c = 2 * sum(sum(mult_c));
                        mult_d = -2*N_g;
                        mult_e = (2./RHO(g_ind,1,h)) .* (V_GAMMA(g_ind,1,h) + ...
                            abs(MU_GAMMA(g_ind,1,h)).^2 + abs(THETA(g_ind,1,h)).^2 - ...
                            2*real(conj(THETA(g_ind,1,h)).*MU_GAMMA(g_ind,1,h)));
                        mult_e = sum(mult_e);
                    case 'col'  % Each column is a Gauss-Markov process
                        mult_a = -2*N_g*(N-1);
                        mult_b = (RHO(2:N,g_ind,h)).^(-1) .* (2*real(GAMMA_CORR) - ...
                            2*real(conj(THETA(2:N,g_ind,h)).*MU_GAMMA(2:N,g_ind,h)) - ...
                            2*(V_GAMMA(1:N-1,g_ind,h) + abs(MU_GAMMA(1:N-1,g_ind,h)).^2) + ...
                            2*real(conj(THETA(1:N-1,g_ind,h)).*MU_GAMMA(1:N-1,g_ind,h)));
                        mult_b = sum(sum(mult_b));
                        mult_c = (RHO(2:N,g_ind,h)).^(-1) .* ((V_GAMMA(2:N,g_ind,h) + ...
                            abs(MU_GAMMA(2:N,g_ind,h)).^2) - 2*real(GAMMA_CORR) + ...
                            (V_GAMMA(1:N-1,g_ind,h) + abs(MU_GAMMA(1:N-1,g_ind,h)).^2));
                        mult_c = 2 * sum(sum(mult_c));
                        mult_d = -2*N_g;
                        mult_e = (2./RHO(1,g_ind,h)) .* (V_GAMMA(1,g_ind,h) + ...
                            abs(MU_GAMMA(1,g_ind,h)).^2 + abs(THETA(1,g_ind,h)).^2 - ...
                            2*real(conj(THETA(1,g_ind,h)).*MU_GAMMA(1,g_ind,h)));
                        mult_e = sum(mult_e);
                end
                
                % Now find the roots of the cubic polynomial of alpha
                try
                    % We can support both real- and complex-valued alpha
                    % updates, but for now the rest of the stuff is only
                    % working for real-valued quantities
                    switch obj.data
                        case 'real'
                            % Real-valued update
%                         alpha_roots = roots([-N_g*T, mult_b/2, mult_c/2]);
                            alpha_roots = roots([-mult_a/2, (mult_a - (mult_b/2 + ...
                                mult_e/2) + mult_d/2), ((mult_b + mult_e) - mult_c/2), ...
                                mult_c]);
                        case 'complex'
                          	% Complex-valued update
%                         alpha_roots = roots([-2*N_g*T, mult_b, mult_c]);
                            alpha_roots = roots([-mult_a, (2*mult_a - (mult_b + ...
                                mult_e) + mult_d), (2*(mult_b + mult_e) - mult_c), ...
                                2*mult_c]);
                    end
                    if isempty(alpha_roots(alpha_roots > 0 & alpha_roots < 1))
                        warning(['Unable to find valid root in computing alpha for ' ...
                            'Gaussian component ' num2str(h) ', thus returning previous estimate'])
                        % Return previous estimate
                        Alpha_upd(:,:,h) = obj.alpha(:,:,h);
                    else
                        % Clip allowable range for alpha
                        alpha_upd = alpha_roots(alpha_roots > 0 & ...
                            alpha_roots < 1);
                        alpha_upd = max(min(alpha_upd, 0.99), 0.001);
                        Alpha_upd(:,:,h) = repmat(alpha_upd, N, T);
                    end
                catch
                    % Either NaN or inf arguments were passed to roots fxn,
                    % suggesting that the EM procedure is diverging.  We can try to
                    % salvage it by just returning the previous estimate of alpha,
                    % but no guarantees here...
                    warning(['NaN or Inf arguments encountered during alpha update' ...
                        ' for component ' num2str(h) ', thus returning previous estimate'])
                end
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