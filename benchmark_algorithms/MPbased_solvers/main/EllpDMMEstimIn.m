classdef EllpDMMEstimIn < EstimIn
    % EllpDMMEstimIn: Performs Donoho/Maleki/Montanari-style thresholding
    % when using GAMP to solve "min_x 1/2/var*norm(y-A*x,2)^2 + lambda*norm(x,p)^p"  
    % for 0<p<=1.  Warning: the val output is currently set to zero

    
    properties
	%DMM threshold gain "alpha", which is NOT the "lambda" in the cost function. 
	%Here, the soft threshold is set as thresh = alpha * sqrt(mean(mur));
        alpha;
        
        %Value of p defining the ell-p norm (0<p<=1, where p=1 for soft thresholding)
        p;
    end
    
    methods
        % Constructor
        function obj = EllpDMMEstimIn(alpha,p)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.alpha = alpha;
                obj.p = p;
            end
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(~)
	    %Set these to reasonable constants
            mean0 = 0; 
            var0  = 1e-2;
            valInit = -Inf;
        end
        
        % Carry out soft thresholding
        function [xhat,xvar,val] = estim(obj,rhat,rvar)
            
            
            %Compute the threshold
            thresh = obj.alpha * sqrt(mean(rvar));
            
            %Estimate the signal
            xhat = max(0,abs(rhat)-thresh.*abs(rhat).^(obj.p-1)) .* sign(rhat);
                
            %Estimate the variance
            xvar = rvar .* (1 - thresh.*(obj.p-1).*abs(rhat).^(obj.p-2)) .* (xhat~=0);
            
            %For now, let's set the val output to zero. Not clear
	    %how to properly set this without knowledge of "lambda"
            val = zeros(size(xhat));
            
            
        end
        
    end
    
end

