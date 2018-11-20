classdef FistaOpt
    % Options for the FISTA optimizer.
    
    properties
        nit = 100;           % number of iterations
        verbose = false;    % Print progress
        
        %Specify a convergence tolerance. Iterations are terminated when
        %the norm of the differnece in two iterates divided by the norm of
        %the current iterate falls below this threshold. Set to -1 to
        %disable
        tol = -1;

        %Specify the lambda parameter. The cost function minimized by FISTA
        %is given as argmin_x ||Ax - b||_2^2 + |Lam .* x|. Can be either
        %scalar or vector
        lam = 1e-2;
        
        %Specify the Lipschitz constant. Should be twice the spectral norm
        %of A. If set to empty, the code estimates it using the power
        %iteration.
        lip = [];
        
    end
    
    methods
        
        % Constructor with default options
        function opt = FistaOpt()
        end
    end
    
end