classdef AmpOpt
    % Options for the AMP optimizer.
    
    properties
        nit = 200;          % number of iterations

        % Method to compute variance at denoiser input. 
        rvarMethod = 'median'; % in {'mean','median','wvar'}	

        % Minimum variance at denoiser input. 
        rvarMin = 1e-12;	
        
        % Print progress
        verbose = false;
        
        %Specify a convergence tolerance. Iterations are terminated when
        %the norm of the differnece in two iterates divided by the norm of
        %the current iterate falls below this threshold. Set to -1 to
        %disable
        tol = 1e-4;
        
        % If desired, a custom stopping criterion can be used to terminate
        % GAMP and/or display useful quantities during runtime.  
        stopFcn = [];  
        % called as stop = stopFcn(val, xhat, xhatPrev, Axhat)
        %   val: the current objective function value
        %   xhat,xhatPrev: the estimate of X at the current,previous iteration
        %   Axhat: the current transform vector (zhat(k) = A*xhat(k)). 
        % A "true" return value indicates GAMP should terminate.

        % Provide initial guesses. If these are set to
        % empty, then the appropriate method of the input estimator is
        % called to initialize these values. This functionality is useful
        % for warm starting the algorithm.
        xhat0 = [];

        % The following are used for warm-starting the algorithm:
        div0 = [];
        vhatPrev0 = [];

        % History Interval: save memory by decimating the saved history
        % e.g. histIntvl=20 will save iteration 20,40,...
        histIntvl = 1; % defaults to save every iteration

        % Automatically checks whether norm(A,'fro')^2 ~= size(x,1)
        checkA = true;
         
        % Automatically normalizes A & y when checkA fails
        normalizeA = true;

        % Run S-AMP from [Cakmak,Winther,Fleury]?
        Stransform = false;
        wvar = []; % measurement noise variance, required for S-AMP
        evalsAAh = []; % eigenvalues of A*A', computed if not specified here

        % The following options are not yet implemented in ampEst.m!
        removeMean = false;
        removeMeanExplicit = true; % implements new matrix explicitly 
        stepMin = 0;
        stepMax = 1;
        stepIncr = 1.1;
        stepDecr = 0.5;
    end
    
    methods
        
        % Constructor with default options
        function opt = AmpOpt(varargin)
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
                        error('AmpOpt: %s is an unrecognized option', ...
                            num2str(varargin{i}));
                    end
                end
                return
            else
                error(['The AmpOpt constructor requires arguments ' ...
                    'be provided in pairs, e.g., AmpOpt(''adaptStep'',' ...
                    ' false, ''nit'', 50)'])
            end
        end

    end
    
end
