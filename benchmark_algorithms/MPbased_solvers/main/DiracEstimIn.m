classdef DiracEstimIn < EstimIn
    % DiracEstimIn:  Dirac-delta scalar input estimation function
    %
    %Assumes p_x(x) = \delta(x - x0)
    
    properties
        x0;      % prior mean
    end
    
    methods
        % Constructor
        function obj = DiracEstimIn(x0)
            obj.x0 = x0;
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = obj.x0;
            var0  = zeros(size(obj.x0));
            valInit = 0;
        end
        
        % DiracEstimIn estimation function
        % Provides the mean and variance of a variable x 
        % from a observation rhat = x + w, w = N(0,rvar)
        function [xmean, xvar, val] = estim(obj, rhat, rvar)
            % Compute posterior mean and variance
            xmean = obj.x0.*ones(size(rhat));
            xvar = zeros(size(rvar));
            
            if (nargout >= 3)  
                val = zeros(size(rhat));
            end
        end
        
        
        % Generate random samples from p(y|z)
        function x = genRand(obj, outSize)
            x = obj.x0.*ones(outSize);
        end
    end
    
end

