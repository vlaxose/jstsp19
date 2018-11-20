classdef BiGAMPOpt
    % Options for the BiG-AMP optimizer.
    
    properties
        
        %***** General Options
        
        %Flag to control sparse mode. In sparse mode, the gOut object
        %should expect and return vectors containing only the observed
        %entries of Z. In sparse mode, rowLocations and columnLocations
        %must be set. In this mode, sparse matrix computations are used to
        %avoid storing full matrices of the MxL size. When the fraction of
        %observed entries is small, this will also result in a
        %computational savings.
        %Sparse mode requires you to first mex the support
        %function by executing: mex sparseMult2.c -largeArrayDims
        sparseMode = false;
        
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
        
        %Specify a convergence tolerance. Iterations are terminated when
        %the norm of the differnece in two iterates divided by the norm of
        %the current iterate falls below this threshold. Set to -1 to
        %disable
        tol = 1e-8;
        
        %Error function. This is a function handle that acts on Zhat to
        %compute an NMSE error.
        error_function = @(q) inf;
        
        %Error functions for A and X
        error_functionA = @(q) inf;
        error_functionX = @(q) inf;
        error_functionX2 = @(q) inf;
        
        %Option to "normalize" variances for computation. May improve
        %robustness in some cases. Use not recommended.
        varNorm = false;
        
        
        %***** Initialization
        
        %Provide initial guesses for xhat0,xvar0,shat0. If these are set to
        %empty, then the appropriate method of the input estimator is
        %called to initialize these values. This functionality is useful
        %for warm starting the algorithm when not providing the full state.
        xhat0 = [];
        Ahat0 = [];
        xvar0 = [];
        Avar0 = [];
        shat0 = [];
        
        %These are used by the X2 variant of BiG-AMP
        x2hat0 = [];
        x2var0 = [];
        
        %***** Step Size Control
        
        %Logical flag to include a step size in the pvar/zvar calculation.
        %This momentum term often improves numerical performance. On by
        %defualt.
        pvarStep = true;
        
        %Initial step size, or fixed size for non-adaptive steps
        step = 0.05;
        
        % Enable adaptive step size
        adaptStep = true;
        
        % Disable computing the cost via Bethe free energy
        adaptStepBethe = false;
        
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
        %with scalar approximations
        uniformVariance = false;
        
        %Minimum variances. See code for details of use.
        pvarMin = 1e-13;
        xvarMin = 0;
        AvarMin = 0;
        zvarToPvarMax = 0.99;   % prevents negative svar, should be near 1
        
        %Variance threshold for rvar and qvar, set large for no effect
        varThresh = 1e6;
        
        %This was included for testing and should not be modified
        gainMode = 1;
        
    end
    
    methods
        
        % Constructor with default options
        function opt = BiGAMPOpt(varargin)
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
                        error('BiGAMPOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The BiGAMPOpt constructor requires arguments ' ...
                    'be provided in pairs, e.g., BiGAMPOpt(''adaptStep'',' ...
                    ' false, ''nit'', 50)'])
            end
        end
    end
    
end
