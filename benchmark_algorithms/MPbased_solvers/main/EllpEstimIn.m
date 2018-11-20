classdef EllpEstimIn < EstimIn
    % EllpEstimIn:  Inputs a soft thresholding scalar input function.
    % Allows GAMP to be used to solve "min_x 1/2/var*norm(y-A*x,2)^2 + lambda*norm(x,p)^p"
    % for 0<p<=1. 
    
    % Public properties.  Can be set with the set / get interface.
    properties
        %lambda = the gain on the ellp term in the MAP cost.
        %The soft threshold is set according to the expression thresh = lambda * mur;
        lambda=1;
        
        %Value of p defining the ell-p norm (0<p<=1, where p=1 for soft thresholding)
        p=1;
        
        % Dimension -- needed to support concatenation
        % Empty implies dimension is autofit to input vector
        nx = [];
        ncol = 1;
        
        % Initial values
        mean0 = 0; 
        var0  = 5e-4;
        valInit = -inf; % Set to [] to autocompute this
        varMax = inf;
    end
    
    methods
        % Constructor
        function obj = EllpEstimIn(lambda, p, mean0, var0)
            obj = obj@EstimIn;
            if (nargin >= 1)
                obj.lambda = lambda;
            end
            if (nargin >= 2)
                obj.p = p;
            end
            if (nargin >= 3)
                obj.mean0 = mean0;
            end
            if (nargin >= 4)
                obj.var0 = var0;
            end
            
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = obj.mean0;
            var0 = obj.var0;
            
            % Resize to fit output
            if (~isempty(obj.nx) && length(obj.mean0) == 1)
                mean0 = repmat(mean0, obj.nx, obj.ncol);
                var0 = repmat(var0, obj.nx, obj.ncol);
            end
            
            % Autocompute the initial value if requested.
            % I believe this should always be used...
            if (isempty(obj.valInit))
                valInit = -obj.lambda*abs(mean0).^obj.p;
            else
	        valInit = obj.valInit;
            end
        end
        
        % Carry out soft thresholding
        function [xhat,xvar,val] = estim(obj,rhat,rvar)
            
            %Compute the thresh
            thresh = obj.lambda .* rvar;
            
            %Estimate the signal
            xhat = max(0,abs(rhat)-thresh.*abs(rhat).^(obj.p-1)) .* sign(rhat);
            
            %Estimate the variance
            xvar = rvar .* (1 - thresh.*(obj.p-1).*abs(rhat).^(obj.p-2)) .* (xhat~=0);
            xvar = min(xvar, obj.varMax);
            
            %Output negative cost 
            %val = -1*obj.lambda*norm(rhat,obj.p)^obj.p;
%             val = -1*obj.lambda*norm(xhat,obj.p)^obj.p;		% seems to work better?
            val = -obj.lambda*abs(xhat).^obj.p;		% seems to work better?
            
        end
        
        % Size operator
        function [nx,ncol] = size(obj)
            nx = obj.nx;
            ncol = obj.ncol;
        end
        
    end
    
end

