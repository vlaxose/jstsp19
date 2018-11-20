classdef PBiGAMPOpt
    %PBiGAMPOpt Options for the P-BiG-AMP optimizer
    
    properties
        
        %***** General Options
        
        % Print progress
        verbose = false;
        
        %Return additional outputs useful for EM learning. Can be disabled
        %to reduce the memory required for the output structure
        saveEM = true;
        
        %Save diagnostic information. This option should only be used for
        %debugging purposes, as it will increase code run time.
        diagnostics = false;
        
        %Number of iterations
        nit = 1500;
        
        %Minimum number of iterations- sometimes useful with warm starting
        %approaches
        nitMin = 30; %0 for no effect
        
        %Convergence tolerance based on change in zhat. Iterations are 
        % terminated when norm(zhat-zhatOld)/norm(zhat)<tol.
        %Set to -1 to disable
        tol = 1e-8;

        %Convergence tolerance based on norm of zhat. Iterations are 
        % terminated when norm(zhat)<normTol, which is typically a sign
        % that the algorithm is trapped in a local minimum.
        %Set to -1 to disable
        normTol = -1;

        %Error function. This is a function handle that acts on zhat to
        %compute an error metric, typically a normalized residual error.
        error_function = @(q) inf;
        
        %Error functions for B and C, similar to above.
        error_functionB = @(q) inf;
        error_functionC = @(q) inf;
       
        %autoTuning disabled if error_function > errTune
        errTune = inf;
        
        %***** Initialization
        
        %Provide initial guesses. If these are set to
        %empty, then the appropriate method of the input estimator is
        %called to initialize these values. This functionality is useful
        %for warm starting the algorithm when not providing the full state.
        bhat0 = [];
        chat0 = [];
        bvar0 = [];
        cvar0 = [];
        shat0 = [];

        %The full state for warm-starting
        state = [];

        %Return the state structure for warm-starting in estFin. Can be
        %disabled to reduce the memory burden.
        saveState = false; 
        
        
        %***** Step Size Control
        
        %Logical flag to include a step size in the pvar/zvar calculation.
        %This momentum term often improves numerical performance. On by
        %defualt.
        pvarStep = true;
        
        %Initial step size, or fixed size for non-adaptive steps
        step = 0.05;
        
        % Enable adaptive step size
        adaptStep = true;
        
        % Minimum step size
        stepMin = 0.05;
        
        % Maximum step size
        stepMax = 0.5;
        
        % Multiplicative step size increase, when successful
        stepIncr = 1.1;
        
        % Multiplicative step size decrease, when unsuccessful
        stepDecr = 0.5;
        
        %Maximum number of allowed failed steps before we decrease stepMax,
        %inf for no effect
        maxBadSteps = inf;
        
        %Amount to decrease stepMax after maxBadSteps failed steps, 1 for
        %no effect
        maxStepDecr = 0.5;
        
        %Create a window for the adaptive step size test. Setting this to
        %zero causes it to have no effect. For postive integer values,
        %creats a moving window of this length when checking the step size
        %acceptance criteria. The new value is only required to be better
        %than the worst in this window, i.e. the step size criteria is not
        %required to monotonicaly increase.
        stepWindow = 1;
        
        % Iterations are termined when the step size becomes smaller
        % than this value. Set to -1 to disable
        stepTol = -1;
        
        %This is a filter value on the steps. It slows down the early steps
        %in a smooth way. Set to less than 1 for no effect, In particular,
        %the steps are computed as step1 = step*it/(it + stepFilter)
        stepFilter = 0;
        
        
        %***** Variance Control
        
        %Switch to enable uniform variance mode. Replaces all variances
        %with scalar approximations. The ParametricZ object must support
        %this functionality for it to work correctly. 
        uniformVariance = false;
        gBind = []; % change indices for EstimInConcat on b
        gCind = []; % change indices for EstimInConcat on c
        
        %Minimum variances. See code for details of use.
        bvarMin = 0;
        cvarMin = 0;
        pvarMin = 1e-13;
        zvarToPvarMax = 0.99;   % prevents negative svar when <= 1
        
        %Variance threshold for rvar and qvar, set large for no effect
        varThresh = 1e6;
        
        
        
    end
    
    methods
        
        %Constructor
        function opt = PBiGAMPOpt(varargin)
            if nargin == 0
                % No custom parameters values, thus create default object
                return
            elseif mod(nargin, 2) == 0
                % User is providing property/value pairs
                names = fieldnames(opt);    % Get names of class properties
                
                % Iterate through every property/value pair, assigning
                % user-specified values.  Note that the matching is NOT
                % case-sensitive
                for i = 1 : 2 : nargin - 1
                    if any(strcmpi(varargin{i}, names))
                        % User has specified a valid property
                        propName = names(strcmpi(varargin{i}, names));
                        opt.(propName{1}) = varargin{i+1};
                    else
                        % Unrecognized property name
                        error('PBiGAMPOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The PBiGAMPOpt constructor requires arguments ' ...
                    'be provided in pairs, e.g., PiGAMPOpt(''adaptStep'',' ...
                    ' false, ''nit'', 50)'])
            end
        end
    end
    
end

