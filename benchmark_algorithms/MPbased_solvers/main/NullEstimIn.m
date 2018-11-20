classdef NullEstimIn < EstimIn
    % NullEstimIn:  estimation with no prior 

    properties
        % Prior mean and variance.  Only used initially
        mean0;  % Mean
        var0;   % Variance
        isCmplx=nan; % true,false,or nan to autodecide
        maxSumVal = true; 
    end
    
    methods
        % Constructor
        function obj = NullEstimIn(mean0, var0,varargin )
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.mean0 = mean0;
                obj.var0 = var0;
            end

            % allow named arguments e.g. NullEstimIn(0,1,'isCmplx',false)
            for i = 1:2:length(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = obj.mean0;
            var0  = obj.var0;
            valInit = 0;
        end
        
        % Circular AWGN estimation function
        % Provides the mean and variance of a variable u
        % from an observation v = u + w, w = CN(0,wvar)
        %
        function [xhat, xvar, val] = estim(obj, rhat, rvar)
            if isnan(obj.isCmplx)
                cpx = any(~isreal(rhat));
            else
                cpx = obj.isCmplx;
            end

            if cpx
                xhat = rhat;  
            else
                xhat = real(rhat); % enforce real
            end
            xvar = rvar;

            if obj.maxSumVal 
                val = 0; % log ( constant )
            else
                % E( log ( p(x) / p(x|R=rhat) ) | R=rhat )
                % = int_x { -log(p_{R|x}) p_{R|x}  }  because likelihood p_x =1 and p_{R|X} = p_{X|R}
                % (below assumes Real)
                % =  -int_x { ( -log(2*pi*rvar)/2 -(rhat-x)^2/(2*rvar) ) Normal(x;rhat,rvar) } 
                % =  log(2*pi*rvar)/2 + int_y {(y^/(2*rvar) ) Normal(y;0,rvar) }
                % =  log(2*pi*rvar)/2 + 1/2
                if cpx
                    val =  log(pi*rvar).*ones(size(rhat));
                else
                    val = .5* log(2*pi*rvar).*ones(size(rhat));
                end
            end
        end
        
        % Computes the likelihood p(y) for y = x + v, v = CN(0,yvar)
        function py = plikey(obj,y,yvar)
            ny = length(y);
            py = zeros(ny,1);
        end
        
        % Computes the log-likelihood, log p(y), for y = x + v, where 
        % x = CN(obj.mean0, obj.var0) and v = CN(0, yvar)
        function logpy = loglikey(obj, y, yvar)
            ny = length(y);
            logpy = zeros(ny,1);
        end
        
    end
    
end

