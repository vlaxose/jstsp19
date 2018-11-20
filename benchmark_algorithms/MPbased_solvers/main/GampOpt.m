classdef GampOpt
    % Options for the GAMP optimizer.
    
    properties
        nit = 200;          % number of iterations
        
        % Remove mean from A by creating a new matrix with one additional
        % row and column.
        removeMean = false;
        removeMeanExplicit = true; % implements new matrix explicitly 
        
        % Relative minimum variance on output and input.
        % This prevents poor conditioning.
        pvarMin = 1e-12;
        xvarMin = 1e-12;	% really this should be named "rvarMin"!
        zvarToPvarMax = inf;    % maximum allowed zvar-to-pvar ratio
        
        % Enable adaptive stepsize
        adaptStep = true;
        
        % Disable computing the cost via Bethe free energy
        adaptStepBethe = false;
        
        %Create a window for the adaptive stepsize test. Setting this to
        %zero causes it to have no effect. For postive integer values,
        %creates a moving window of this length when checking the stepsize
        %acceptance criteria. The new value is only required to be better
        %than the worst in this window, i.e. the stepsize criteria is not
        %required to monotonicaly increase. As with other algorithms, this
        %modification tends to improve convergence speed
        stepWindow = 20;
        
        %Set to true to use a Barzilai Borwein type rule to choose a new
        %stepsize after each succesful step. Otherwise, the stepsize is set to
        %the previous successful stepsize
        bbStep = false;
        
        %Option to use identical variances across coefficients. Allows
        %alg to run about twice as fast, but may impact performance
        uniformVariance = false;
        
        % Print progress
        verbose = false;
        
        %Specify a convergence tolerance. Iterations are terminated when
        %the norm of the differnece in two iterates divided by the norm of
        %the current iterate falls below this threshold. Set to -1 to
        %disable
        tol = 1e-4;
        
        % Initial stepsize
        step = 1;
        
        % Minimum stepsize.  If stepsize is initialized below this value,
        % then the iteration will be automatically considered successful.
        stepMin = 0;
        
        % Maximum allowed stepsize
        stepMax = 1;
        
        % Multiplicative stepsize increase, when successful
        stepIncr = 1.1;
        
        % Multiplicative stepsize decrease, when unsuccessful
        stepDecr = 0.5;
        
        %Maximum number of allowed failed steps before we decrease stepMax,
        %inf for no effect; 10 is a decent value to play with
        maxBadSteps = inf;
        
        %Amount to decrease stepMax after maxBadSteps failed steps, 1 for
        %no effect
        maxStepDecr = 0.8;
        
        % Iterations are termined when the stepsize becomes smaller
        % than this value. Set to -1 to disable
        stepTol = 1e-10;
        
        % If desired, a custom stopping criterion can be used to terminate
        % GAMP and/or display useful quantities during runtime.  
	% There are two versions:
        stopFcn = [];  
        % called as stop = stopFcn(val, xhat, xhatPrev, Axhat)
        %   val: the current objective function value
        %   xhat,xhatPrev: the estimate of X at the current,previous iteration
        %   Axhat: the current transform vector (zhat(k) = A*xhat(k)). 
        % A "true" return value indicates GAMP should terminate.
        stopFcn2 = [];
        % called as:  stop = stopFcn2(gampstate);
        %   gampstate: a structure whose fields describe the current state.
        % A stop value of "true" indicates GAMP should terminate.
        % For full details, see gampEst.m.

        % Robust GAMP options
        % Logical flag to include a stepsize in the pvar calculation.
        % This momentum term often improves numerical performance. On by
        % defualt.
        pvarStep = true;
        
        % Logical flag to include a stepsize in the rvar calculation.
	% Experimental; off by default
        rvarStep = false;
        
        % Option to "normalize" variances for computation. May improve
        % robustness in some cases, but fundamentally changes the quantities
        % that are slowed down when step<1, so only use with non-adaptive step=1
        varNorm = false;
        
        % Provide initial guesses for xhat0,xvar0,shat0. If these are set to
        % empty, then the appropriate method of the input estimator is
        % called to initialize these values. This functionality is useful
        % for warm starting the algorithm.
        xhat0 = [];
        xvar0 = [];
        shat0 = [];
        
        % The following are used for warm-starting the algorithm:
        svar0 = [];
        pvarOpt0 = [];
        rvarOpt0 = [];
        A2xvarOpt0 = [];
        xhatPrev0 = [];
        xhatDampPrev0 = [];
        scaleFac = 1;
        valOpt0 = [];
        valIn0 = [];
        failCount0 = 0;

        % Automatic determination of xvar0.  This is useful if you have a 
	% good point estimate xhat0, but don't know how to set xvar0.
	% Off by default.
	xvar0auto = false;

        % History Interval: save memory by decimating the saved history
        % e.g. histIntvl=20 will save iteration 20,40,...
        histIntvl = 1; % defaults to save every iteration

        %The following selects what output format is used, and whether there
        %is a warning about the change in outputs
        legacyOut = true;
        warnOut = false; % NO LONGER USED: CAN REMOVE 
    end
    
    methods
        
        % Constructor with default options
        function opt = GampOpt(varargin)
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
                        error('GampOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The GampOpt constructor requires arguments ' ...
                    'be provided in pairs, e.g., GampOpt(''adaptStep'',' ...
                    ' false, ''nit'', 50)'])
            end
        end

        function other = warmStartCopy(this,estFin,varargin)
        %
        %   go2 = go.warmStartCopy(estFin[,'key1',val1,...])
        %
        % returns a copy of this GampOpt with specified changes:
        %
        % 1) warm start values from the provided estFin struct (given by gampEst with legacyOut=false)
        %
        % 2) any additional key,value pairs, e.g.,
        %       go1 = GampOpt('legacyOut',false,...)
        %       ...
        %       estFin = gampEst(... , go1);
        %       go2 = go1.warmStartCopy(estFin,'tol',1e-7,'nit',20);
        %       ...
        %       estFin2 = gampEst(... , go2);
        %
        % Note: this is a legacy method, replaced by .warmStart

            warning('This is a legacy method, replaced by .warmStart')
         
            other = this;
            % populate the warm start fields used by gampEst
            if ~isempty(estFin)

                % old gampEst fields
                other.xhat0 = estFin.xhat;
                other.xvar0 = estFin.xvar;
                other.shat0 = estFin.shat;
                other.svar0 = estFin.svar;
                if isfield(estFin,'xhatPrev')
                    other.xhatPrev0 = estFin.xhatPrev;
                end

                % fields related to adaptive stepsizes and variance normalization
                other.scaleFac = estFin.scaleFac;
                other.step = estFin.step;
                other.stepMax = estFin.stepMax;

                % recently added gampEst fields
                other.xhatDampPrev0 = nan;
                other.pvarOpt0 = nan;
                other.rvarOpt0 = nan;
                other.A2xvarOpt0 = nan;
                other.failCount0 = 0;
%               other.valOpt0 = [];
                other.valIn0 = -inf;
            end
            % override any additional fields
            for i = 1:2:length(varargin)
                other.(varargin{i}) = varargin{i+1};
            end
        end

        function other = warmStart(this,estFin,varargin)
        %
        %   go2 = go.warmStart(estFin[,'key1',val1,...])
        %
        % returns a copy of this GampOpt with specified changes:
        %
        % 1) warm start values from the provided estFin struct (given by gampEst with legacyOut=false)
        %
        % 2) any additional key,value pairs, e.g.,
        %       go1 = GampOpt('legacyOut',false,...)
        %       ...
        %       estFin = gampEst(... , go1);
        %       go2 = go1.warmStart(estFin,'tol',1e-7,'nit',20);
        %       ...
        %       estFin2 = gampEst(... , go2);

            if (this.removeMean)
                warning('Mean-removal-augmented state variables are not preserved in .warmStart.')
	    end

            other = this;
            % populate the warm start fields used by gampEst
            if ~isempty(estFin)
                % basic gampEst fields
                other.xhatDampPrev0  = estFin.xhatDamp; % used only by gampEstBasic
                other.xhatPrev0 = estFin.xhatDamp; % used only by gampEst
                other.xhat0 = estFin.xhatNext;
                other.xvar0 = estFin.xvarNext;
                other.shat0 = estFin.shatNext;
                other.svar0 = estFin.svarNext;
                other.pvarOpt0 = estFin.pvarOpt;
                other.rvarOpt0 = estFin.rvarOpt;
                other.A2xvarOpt0 = estFin.A2xvarOpt; % used only by gampEst

                % fields related to adaptive stepsizes and variance normalization
                other.scaleFac = estFin.scaleFac;
                other.step = estFin.step;
                other.stepMax = estFin.stepMax;
                other.failCount0 = estFin.failCount;
                other.valIn0 = estFin.valIn;
                other.valOpt0 = estFin.valOpt;
            end
            % override any additional fields
            for i = 1:2:length(varargin)
                other.(varargin{i}) = varargin{i+1};
            end
        end

    end
    
end
