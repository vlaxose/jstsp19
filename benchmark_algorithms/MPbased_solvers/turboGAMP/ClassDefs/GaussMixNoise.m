% CLASS: GaussMixNoise
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Observation
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used to define the statistics of the separable
%   observation channel p(y(m,t) | z(m,t)), m = 1, ..., M, t = 1, ..., T,
%   for the case of a binary Gaussian mixture distribution, i.e.,
%     p(y(m,t) | z(m,t)) = (1 - PI(m,t)) Normal(y(m,t); z(m,t), NU0(m,t)) +
%                               PI(m,t) Normal(y(m,t); z(m,t), NU1(m,t)).
%   This model is useful for modeling noise that is typically small, (with
%   variance NU0), but with probability PI can be large (with variance
%   NU1).  By default, this class assumes real-valued noise.  For a
%   complex-valued noise model, set data = 'complex'.
%
%   The distribution above is specified by the parameters PI, NU0, and NU1,
%   however this class will instead use the slightly different inputs of
%   PI, NU0, and NUratio = NU1/NU0.  Acceptable inputs include a scalar, 
%   an M-by-1 row vector, a 1-by-T column vector, or an M-by-T matrix, for
%   any of the input parameters.  In the first three cases, the input is 
%   replicated to yield an M-by-T matrix of parameter values.
%
%   To learn PI, NU0, and NUratio from the data using an expectation-
%   maximization (EM) learning algorithm, the properties learn_pi, 
%   learn_nu0, and learn_nuratio can be set to 'scalar' to learn a single 
%   common value of each parameter, 'row' to learn a unique parameter value
%   for each index m = 1, ..., M, or 'column' to learn a unique parameter
%   value for each index t = 1, ..., T.  To prevent the EM algorithm from 
%   learning any subset of these parameters, set the corresponding learn_*
%   parameter to 'false'.
%
%   To create a GaussMixNoise object, one of two constructors may be used.
%   The default constructor, GaussMixNoise(), will set all properties to 
%   their default values.  The alternative constructor adopts MATLAB's
%   property/value string pair convention to allow the user to specify any
%   subset of properties, with remaining properties set to their default
%   values.  For example, GaussMixNoise('PI', 0.20, 'learn_pi', 'column') 
%   will construct a GaussMixNoise object that specifies an initial
%   weight matrix PI(m,t) = 0.20 for all m,t, and indicates that a unique
%   weight is to be learned using the EM algorithm for each column of PI.
%
% PROPERTIES (State variables)
%   PI                  The prior mixing weight  [Default: 0.10]
%   NU0                 The small component variance  [Default: 1e-2]
%   NUratio             Ratio of big-to-small variances, NU1/NU0  [Default:
%                       10]
%   learn_pi            Specifies options for learning the mixing weight
%                       using an EM learning algorithm.  Acceptable
%                       settings are 'scalar', 'row', 'column', or 'false'.
%                       [Default: 'scalar']
%   learn_nu0        	Specifies options for learning small variance NU0
%                       using an EM learning algorithm.  Acceptable
%                       settings are 'scalar', 'row', 'column', or 'false'.
%                       [Default: 'scalar']
%   learn_nuratio      	Specifies options for learning the big-to-small
%                       variance ratio, NUratio, using an EM learning 
%                       algorithm.  Acceptable settings are 'scalar', 
%                       'row', 'column', or 'false'.  [Default: 'scalar']
%   data                Real-valued ('real') or complex-valued ('complex')
%                       noise? [Default: 'real']
%
% METHODS (Subroutines/functions)
%   GaussMixNoise()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   GaussMixNoise('ParameterName1', Value1, 'ParameterName2', Value2)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class GaussNoise, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%       - Returns the character string 'GMN' when obj is a GaussMixNoise
%         object
%   GaussMixNoiseCopyObj = copy(obj)
%       - Creates an independent copy of a GaussMixNoise object, obj
%   EstimOut = UpdatePriors(obj, GAMPState, Y)
%     	- Given the final state of the message passing variables that were
%         output from GAMP after its most recent execution, produce a new 
%         object of the EstimOut base class that will be used to specify 
%         the Gaussian mixture noise "prior" on the next iteration of GAMP. 
%         TBobj is an object of the TurboOpt class, GAMPState is an object 
%         of the GAMPState class, and Y is an M-by-T matrix of 
%         measurements.  If TBobj.commonA is false, then this method 
%         returns a 1-by-T cell array of EstimOut objects.  [Hidden method]
%   EstimOut = InitPriors(TBobj)
%    	- Provides an initial EstimOut object for use by GAMP the first
%         time. TBobj is a TurboOpt object.  If TBobj.commonA is false, 
%         then this method returns a 1-by-T cell array of EstimOut objects.
%         [Hidden method]
%   Y = genRand(TBobj, GenParams, Z)
%       - Produce a realization of y(t), t = 1,...,T, given a TurboOpt
%         object (TBobj), a GenParams object, and tranform coefficient, Z,
%         (either as an M-by-T matrix, or a 1-by-T cell array of vectors)
%         as inputs [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'NU0'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Small noise variance'), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/30/13
% Change summary: 
%       - Created (02/16/12; JAZ)
%       - Added support for complex-valued noise (05/22/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method implementation (01/30/13; JAZ)
% Version 0.2
%

classdef GaussMixNoise < Observation
    
    properties
        PI = 0.10;      % The prior mixing weight  [Dflt: 0.10]
        NU0 = 1e-2;     % The small component variance  [Dflt: 1e-2]
        NUratio = 10;   % Ratio of big-to-small variances, NU1/NU0  [Dflt: 10]
        learn_pi = 'scalar';        % Learn mixing weight option [Dflt: 'scalar']
        learn_nu0 = 'scalar';       % Learn small variance, NU0 [Dflt: 'scalar']
        learn_nuratio = 'scalar';  	% Learn ratio NU1/NU0 [Dflt: 'scalar']
        data = 'real';              % Real- or compex-valued noise [Dflt: 'real']
        version = 'mmse';
    end % properties
    
    properties (Constant, Hidden)
        type = 'GMN';   % Noise distribution identifier string
    end
   
    methods 
        
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = GaussMixNoise(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(varargin{i}, varargin{i+1});
                end
            end
        end
        
        
        % *****************************************************************
        %                         SET METHODS
        % *****************************************************************
        
        % Set method for mixing weight (PI)
        function obj = set.PI(obj, PI)
           if any(PI(:)) < 0 || any(PI(:)) > 1
              error('Mixing weights must be in interval [0,1]')
           else
              obj.PI = PI;
           end
        end
        
        % Set method for learn_pi
        function obj = set.learn_pi(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_pi')
            end
            obj.learn_pi = lower(string);
        end
        
        % Set method for small component variance (NU0)
        function obj = set.NU0(obj, NU0)
           if any(NU0(:)) < 0
              error('Noise variances must be non-negative')
           else
              obj.NU0 = NU0;
           end
        end
        
        % Set method for learn_nu0
        function obj = set.learn_nu0(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_nu0')
            end
            obj.learn_nu0 = lower(string);
        end
        
        % Set method for big-to-small variance ratio (NU1/NU0)
        function obj = set.NUratio(obj, NUratio)
           if any(NUratio(:)) < 1
              error('Please set NUratio >= 1')
           else
              obj.NUratio = NUratio;
           end
        end
        
        % Set method for learn_nuratio
        function obj = set.learn_nuratio(obj, string)
            if ~check_input(obj, string)
                error('Invalid option: learn_nuratio')
            end
            obj.learn_nuratio = lower(string);
        end
        
        % Set method for version
        function obj = set.version(obj, version)
            if strcmpi(version, 'mmse')
                obj.version = lower(version);
            elseif strcmpi(version, 'map')
                error('GaussMixNoise does not support MAP message passing')
            else
                error('Invalid option: version')
            end
        end
        
        
        % *****************************************************************
        %                         PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('NOISE PRIOR: Gaussian mixture\n')
            fprintf('                  Mixture weight (PI): %s\n', ...
                form(obj, obj.PI))
            fprintf('       Small component variance (NU0): %s\n', ...
                form(obj, obj.NU0))
            fprintf('Big-to-small variance ratio (NUratio): %s\n', ...
                form(obj, obj.NUratio))
            fprintf('                             learn_pi: %s\n', ...
                obj.learn_pi)
            fprintf('                            learn_nu0: %s\n', ...
                obj.learn_nu0)
            fprintf('                        learn_nuratio: %s\n', ...
                obj.learn_nuratio)
            fprintf('                                 data: %s\n', ...
                obj.data)
        end
        
        
        % *****************************************************************
        %                         COPY METHOD
        % *****************************************************************
        
        % Creates an independent copy of a GaussMixNoise object
        function GaussMixNoiseCopyObj = copy(obj)
            GaussMixNoiseCopyObj = GaussMixNoise('PI', obj.PI, ...
                'NU0', obj.NU0, 'NUratio', obj.NUratio, 'learn_pi', ...
                obj.learn_pi, 'learn_nu0', obj.learn_nu0, ...
                'learn_nuratio', obj.learn_nuratio, 'data', obj.data);
        end
        
        % *****************************************************************
        %                        ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of noise family
        % this class is by returning a character string type identifier
        function type_id = get_type(obj)
            type_id = obj.type;
        end
        
    end % methods
    
    methods (Hidden)
        
        % *****************************************************************
        %      	         UPDATE GAMP NOISE "PRIOR" METHOD
        % *****************************************************************
        
      	% Update the EstimOut object for a binary Gaussian mixture prior
        function EstimOut = UpdatePriors(obj, TBobj, GAMPState, Y, ...
                ~, ~)
            
            [M, T] = size(Y);
            PI = obj.PI;
            NU0 = obj.NU0;
            NU1 = obj.NU0 .* obj.NUratio;
            
            % Unpack the GAMPState object
            [~, ~, ~, ~, ~, ~, ZMSGHAT, ZMSGVAR] = GAMPState.getState();
            
            % Compute messages for some latent variables, BETA, that
            % indicate which component Gaussian produced which noise
            % sample.  Useful for closed-form EM updates
            YminusZ = abs(Y - ZMSGHAT).^2;
            PI_OUT = ((ZMSGVAR + NU1)./(ZMSGVAR + NU0)).^(1/2) .* ...
                exp(-(1/2)*YminusZ .* ((ZMSGVAR + NU0).^(-1) - ...
                (ZMSGVAR + NU1).^(-1)));
            MULT = (1 + ((1-PI)./PI).*PI_OUT).^(-1);
            MULT2 = (1 + (PI./(1-PI)).*PI_OUT.^(-1)).^(-1);
            PI_OUT = (1 + PI_OUT).^(-1);
            BETA_HAT = PI.*PI_OUT ./ ((1 - PI).*(1 - PI_OUT) + PI.*PI_OUT);
            CON1 = (1./NU1 + 1./ZMSGVAR).^(-1);     % Needed constant
            CON2 = CON1 .* (Y ./ NU1 + ZMSGHAT ./ ZMSGVAR);
            CON3 = (1./NU0 + 1./ZMSGVAR).^(-1);     % Needed constant
            CON4 = CON3 .* (Y ./ NU0 + ZMSGHAT ./ ZMSGVAR);
            
            % Compute E[Z(m,t)|Y] and var{Z(m,t)|Y}
            K = (1./NU0 + 1./ZMSGVAR).^(-1);
            Kb = (1./NU1 + 1./ZMSGVAR).^(-1);
            L = K .* (Y./NU0 + ZMSGHAT./ZMSGVAR);
            Lb = K .* (Y./NU1 + ZMSGHAT./ZMSGVAR);
            AR = PI_OUT.^(-1);
            ZHAT = L./(1+(PI./(1-PI)).*AR) + Lb./(1+((1-PI)./PI).*AR.^(-1));
            ZVAR = (abs(L).^2+K)./(1+(PI./(1-PI)).*AR) + ...
                (abs(Lb).^2+Kb)./(1+((1-PI)./PI).*AR.^(-1)) - abs(ZHAT).^2;
            
            % If user has specified, attempt to learn the binary Gaussian
            % mixture parameters.
            % ------------- Learning PI ---------------
            switch obj.learn_pi
                case 'false'
                    PI_upd = obj.PI;        % Use old parameter values
                    PI = PI_upd;
                case 'scalar'
                    PI_upd = sum(sum(BETA_HAT)) / M / T;
                    PI = TBobj.resize(PI_upd, M, T);
                case 'row'
                    PI_upd = sum(BETA_HAT, 2) / T;
                    PI = TBobj.resize(PI_upd, M, T);
                case 'column'
                    PI_upd = sum(BETA_HAT, 1) / M;
                    PI = TBobj.resize(PI_upd, M, T);
            end
            % ------------- Learning NU0 ---------------
            BETA_HAT = PI.*PI_OUT ./ ((1 - PI).*(1 - PI_OUT) + PI.*PI_OUT);
            switch obj.learn_nu0
                case 'false'
                    NU0_upd = obj.NU0;      % Use old parameter values
                case 'scalar'
%                     NU0_num = (abs(Y).^2).*(1 - BETA_HAT) - 2*Y.*ZHAT + ...
%                         2*PI.*Y.*CON2 + abs(ZHAT).^2 + ZVAR - ...
%                         PI.*(abs(CON2).^2 + CON1);
%                     NU0_num = (abs(Y).^2).*(1 - BETA_HAT) - 2*Y.*ZHAT + ...
%                         2*MULT.*Y.*CON2 + abs(ZHAT).^2 + ZVAR - ...
%                         MULT.*(abs(CON2).^2 + CON1);
                    NU0_num = (abs(Y).^2).*(1 - BETA_HAT) - ...
                        2*MULT2.*Y.*CON4 + MULT2.*(abs(CON4).^2 + CON3);
                    NU0_num = sum(sum(NU0_num));
                    NU0_upd = NU0_num / sum(sum(1 - BETA_HAT));
                case 'row'
                    NU0_num = (abs(Y).^2).*(1 - BETA_HAT) - ...
                        2*MULT2.*Y.*CON4 + MULT2.*(abs(CON4).^2 + CON3);
                    NU0_num = sum(NU0_num, 2);
                    NU0_upd = NU0_num ./ sum(1 - BETA_HAT, 2);
                case 'column'
                    NU0_num = (abs(Y).^2).*(1 - BETA_HAT) - ...
                        2*MULT2.*Y.*CON4 + MULT2.*(abs(CON4).^2 + CON3);
                    NU0_num = sum(NU0_num, 1);
                    NU0_upd = NU0_num ./ sum(1 - BETA_HAT, 1);
            end
            % ------------- Learning NUratio ---------------
            NU0_upd = max(NU0_upd, 1e-8);   % Clip minimal value
            NU0 = TBobj.resize(NU0_upd, M, T);
            PI_OUT = ((ZMSGVAR + NU1)./(ZMSGVAR + NU0)).^(1/2) .* ...
                exp(-(1/2)*YminusZ .* ((ZMSGVAR + NU0).^(-1) - ...
                (ZMSGVAR + NU1).^(-1)));
            MULT = (1 + ((1-PI)./PI).*PI_OUT).^(-1);
            PI_OUT = (1 + PI_OUT).^(-1);
            BETA_HAT = PI.*PI_OUT ./ ((1 - PI).*(1 - PI_OUT) + PI.*PI_OUT);
            switch obj.learn_nuratio
                case 'false'
                    NUratio_upd = obj.NUratio;  % Use old parameter values
                case 'scalar'
%                     NU1_num = (abs(Y).^2).*BETA_HAT - 2*PI.*Y.*CON2 + ...
%                         PI.*(abs(CON2).^2 + CON1);
                    NU1_num = (abs(Y).^2).*BETA_HAT - 2*MULT.*Y.*CON2 + ...
                        MULT.*(abs(CON2).^2 + CON1);
                    NU1_num = sum(sum(NU1_num));
                    NU1_upd = NU1_num / sum(sum(BETA_HAT));
                    NUratio_upd = TBobj.resize(NU1_upd, M, T) ./ ...
                        TBobj.resize(NU0_upd, M, T);
                case 'row'
                    NU1_num = (abs(Y).^2).*BETA_HAT - 2*MULT.*Y.*CON2 + ...
                        MULT.*(abs(CON2).^2 + CON1);
                    NU1_num = sum(NU1_num, 2);
                    NU1_upd = NU1_num / sum(BETA_HAT, 2);
                    NUratio_upd = TBobj.resize(NU1_upd, M, T) ./ ...
                        TBobj.resize(NU0_upd, M, T);
                case 'column'
                    NU1_num = (abs(Y).^2).*BETA_HAT - 2*MULT.*Y.*CON2 + ...
                        MULT.*(abs(CON2).^2 + CON1);
                    NU1_num = sum(NU1_num, 1);
                    NU1_upd = NU1_num / sum(BETA_HAT, 1);
                    NUratio_upd = TBobj.resize(NU1_upd, M, T) ./ ...
                        TBobj.resize(NU0_upd, M, T);
            end
            % If something went wrong, make sure NUratio is at least
            % greater than 1
            NUratio_upd = max(NUratio_upd, 1 + 1e-1);
            
            % Place an M-by-T matrix of updated parameters back into object
            % structure
            obj.PI = TBobj.resize(PI_upd, M, T);
            obj.NU0 = TBobj.resize(NU0_upd, M, T);
            obj.NUratio = TBobj.resize(NUratio_upd, M, T);
            
            % Construct a GaussMixEstimOut object
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimOut object and run
                    % matrix GAMP
                    switch obj.data
                        case 'real'
                            EstimOut = GaussMixEstimOut(Y, obj.NU0, ...
                                obj.NUratio.*obj.NU0, obj.PI);
                        case 'complex'
                            EstimOut = CGaussMixEstimOut(Y, obj.NU0, ...
                                obj.NUratio.*obj.NU0, obj.PI);
                    end
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimOut objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimOut = cell(1,T);
                    switch obj.data
                        case 'real'
                            for t = 1:T
                                EstimOut{t} = GaussMixEstimOut(Y(:,t), ...
                                    obj.NU0(:,t), obj.NUratio(:,t) .* ...
                                    obj.NU0(:,t), obj.PI(:,t));
                            end
                        case 'complex'
                            for t = 1:T
                                EstimOut{t} = CGaussMixEstimOut(Y(:,t), ...
                                    obj.NU0(:,t), obj.NUratio(:,t) .* ...
                                    obj.NU0(:,t), obj.PI(:,t));
                            end
                    end
            end
        end
        
        
        % *****************************************************************
        %          	   INITIALIZE GAMP NOISE "PRIOR" METHOD
        % *****************************************************************
        
        % Initialize EstimOut object for AWGN noise prior
        function EstimOut = InitPriors(obj, TBobj, Y)
            
            % First get the # of timesteps
            [M, T] = size(Y);
            
            % Resize user-provided parameters to make M-by-T
            obj.PI = TBobj.resize(obj.PI, M, T);
            obj.NU0 = TBobj.resize(obj.NU0, M, T);
            obj.NUratio = TBobj.resize(obj.NUratio, M, T);
            
            % Form EstimOut object
            switch TBobj.commonA
                case true
                    % There is one common A matrix for all timesteps, thus
                    % we can use a matrix-valued EstimOut object and run
                    % matrix GAMP
                    switch obj.data
                        case 'real'
                            EstimOut = GaussMixEstimOut(Y, obj.NU0, ...
                                obj.NUratio.*obj.NU0, obj.PI);
                        case 'complex'
                            EstimOut = CGaussMixEstimOut(Y, obj.NU0, ...
                        obj.NUratio.*obj.NU0, obj.PI);
                    end
                case false
                    % Each timestep has a different matrix A(t), thus we
                    % must run vector GAMP with separate EstimOut objects
                    % for each timestep.  Return a cell array of such
                    % objects
                    EstimOut = cell(1,T);
                    switch obj.data
                        case 'real'
                            for t = 1:T
                                EstimOut{t} = GaussMixEstimOut(Y(:,t), ...
                                    obj.NU0(:,t), obj.NUratio(:,t).*obj.NU0(:,t), ...
                                    obj.PI(:,t));
                            end
                        case 'complex'
                            for t = 1:T
                                EstimOut{t} = CGaussMixEstimOut(Y(:,t), ...
                                    obj.NU0(:,t), obj.NUratio(:,t).*obj.NU0(:,t), ...
                                    obj.PI(:,t));
                            end
                    end
            end
        end
        
        % *****************************************************************
        %                   GENERATE NOISE REALIZATION
        % *****************************************************************
        
        function Y = genRand(obj, TBobj, GenParams, Z)
            % Get size information
            if isnumeric(Z)
                [M, T] = size(Z);
            elseif isa(Z, 'cell')
                T = numel(Z);
                M = size(Z{1}, 1);      % Assume time-invariant M
            else
                error('Z must either be a numeric or cell array')
            end
            
            PI = TBobj.resize(obj.PI, M, T);                %#ok<*PROP> % Mix weight
            NU0 = TBobj.resize(obj.NU0, M, T);              % Sm. variance
            NU1 = TBobj.resize(obj.NUratio, M, T) .* NU0;   % Lg. variance
            
            % Generate a noise realization
            switch GenParams.type
                case 'real'
                    if strcmp(obj.data, 'complex')
                        warning(['Changing GaussMixNoise.data to ' ...
                            '''real'' to match GenParams.type'])
                        obj.data = 'real';
                    end
                    MASK = (rand(M,T) > PI);
                    NOISE = MASK .* (sqrt(NU0) .* randn(M,T)) + ...
                        not(MASK) .* (sqrt(NU1) .* randn(M,T));
                case 'complex'
                    if strcmp(obj.data, 'real')
                        warning(['Changing GaussMix.data to ' ...
                            '''real'' to match GenParams.type'])
                            obj.data = 'complex';
                    end
                    MASK = (rand(M,T) > PI);
                    NOISE = MASK .* (sqrt(NU0/2) .* randn(M,T) + ...
                        1j*sqrt(NU0/2) .* randn(M,T)) + ...
                        not(MASK) .* (sqrt(NU1/2) .* randn(M,T) + ...
                        1j*sqrt(NU1/2) .* randn(M,T));
            end
            
            % Add noise to transform coefficients to get channel outputs
            if isnumeric(Z)
                Y = Z + NOISE;
            else
                Y = NaN(M,T);
                for t = 1:T
                    Y(:,t) = Z{t} + NOISE(:,t);
                end
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % Determine number of parameters being learned
            Nparam = ~strcmpi(obj.learn_pi, 'false') + ...
                ~strcmpi(obj.learn_nu0, 'false') + ...
                ~strcmpi(obj.learn_nuratio, 'false');
            Report = cell(obj.D*Nparam, 3);   % Declare Report array
            Params = {  'PI',       'GM noise mixing prob',         'learn_pi'; 
                        'NU0',      'GM noise small variance',   	'learn_nu0';
                        'NU_ratio',	'GM noise big-to-small ratio',  'learn_nuratio'};
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
        %                        HELPER METHODS
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
    end % Check for valid string method
    
   
end % classdef