classdef GampOpt
    % Options for the GAMP optimizer.
    
    properties
        step = 1;           % step size
        nit = 20;           % number of iterations
        
        % Remove mean from A by creating a new matrix with one additional
        % row and column.
        removeMean = false;
        
        % Relative minimum variance on output and input.
        % This prevents poor conditioning. The lower limit is relative to
        % the initial variance
        pvarMin = 1e-10;
        xvarMin = 1e-10;
        
        % Enable adaptive step size
        adaptStep = true;
        
        %Create a window for the adaptive step size test. Setting this to
        %zero causes it to have no effect. For postive integer values,
        %creats a moving window of this length when checking the step size
        %acceptance criteria. The new value is only required to be better
        %than the worst in this window, i.e. the step size criteria is not
        %required to monotonicaly increase. As with other algorithms, this
        %modification tends to improve convergence speed
        stepWindow = 0;
        
        %Set to true to use a Barzilai Borwein type rule to choose a new
        %step size after each succesful step. Otherwise, the step is set to
        %the previous succesful step size
        bbStep = false;
        
        % Print progress
        verbose = false;
        
        %Specify a convergence tolerance. Iterations are terminated when
        %the norm of the differnece in two iterates divided by the norm of
        %the current iterate falls below this threshold. Set to -1 to
        %disable
        tol = -1;
        
        % Minimum step size
        stepMin = 0;
        
        % Maximum step size
        stepMax = inf;


        % Iterations are termined when the step size becomes smaller
        % than this value. Set to -1 to disable
        stepTol = -1;
        
        %Provide initial guesses for xhat0,xvar0,shat0. If these are set to
        %empty, then the appropriate method of the input estimator is
        %called to initialize these values. This functionality is useful
        %for warm starting the algorithm.
        xhat0 = [];
        xvar0 = [];
        shat0 = [];
        
    end
    
    methods
        
        % Constructor with default options
        function opt = GampOpt()
        end
    end
    
end
