classdef NNSoftThreshEstimIn < EstimIn
    % SoftThreshEstimIn:  Inputs a soft thresholding scalar input function.
    % Allows GAMP to be used to solve "min_x>0 1/2/var*norm(y-A*x,2)^2 + lambda*norm(x,1)". 
    
    properties
        %lambda = the gain on the ell1 term in the MAP cost. 
	%The soft threshold is set according to the expression thresh = lambda * mur;
        lambda;
        maxSumVal = true;   % Max-sum GAMP (true) or sum-product GAMP (false)?
    end
    
    methods
        % Constructor
        function obj = NNSoftThreshEstimIn(lambda, maxSumVal)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.lambda = lambda;
                if nargin >= 2 && ~isempty(maxSumVal) && isscalar(maxSumVal)
                    obj.maxSumVal = logical(maxSumVal);
                end
            end
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
	    %Set these to reasonable constants
            mean0 = 1./obj.lambda; 
            var0  = 1./obj.lambda.^2;
            valInit = -inf;
        end
        
        % Carry out soft thresholding
        function [xhat,xvar,val] = estim(obj, rhat, rvar)
            if ~obj.maxSumVal
                 error('sum-product version not coded for NNSoftThreshEstimIn')
            else
                % Compute max-sum GAMP updates
                
                %Compute the thresh
                thresh = obj.lambda .* rvar;
                
                %Estimate the signal
                xhat = max(0,rhat-thresh);
                
                %Estimate the variance
                %xvar = rvar .* (mean(double(abs(xhat) > 0))*ones(size(xhat)));
                xvar = rvar .* (abs(xhat) > 0);
                
                %Output negative cost
                %val = -1*obj.lambda*abs(rhat);
                val = -1*obj.lambda.*abs(xhat);	% seems to work better
            end
        end
            
        % Computes p(y) for y = x + v, with x ~ p(x), v ~ N(0,yvar)
        function py = plikey(obj, rhat, rvar)
            mu = rhat;
            sig2 = rvar;
            sig = sqrt(sig2);                           % Gaussian prod std dev
            muU = mu - obj.lambda.*sig2;                % Upper integral mean
            py = obj.lambda.*exp((muU.^2 - rhat.^2)./rvar*0.5).*...
                max(erfc(-muU./sig/sqrt(2)),1e-300)*0.5;
        end
    end     
    
end

