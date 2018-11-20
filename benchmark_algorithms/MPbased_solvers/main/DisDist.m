classdef DisDist < handle
    %DisDist:  Discrete distribution class
    
    properties
        x;      % Values for the random variable
        px;     % Probability of each point
    end
    
    methods
        
        % Constructor
        function obj = DisDist(x,px)
            obj.x = x;
            px = px / sum(px);
            obj.px = px;
        end
        
        % meanVar:  Returns distributions mean and variance
        function [xmean,xvar] = meanVar(obj)            
            xmean = obj.x'*obj.px;     
            dx = abs(obj.x-xmean).^2;
            xvar = dx'*obj.px;
        end

         % genRand:  Generates random samples of the distribution
         function xrand = genRand(obj, nx)
             xcdf = cumsum(obj.px);
             p = rand(nx,1);
             xrand = zeros(nx,1);
             for j=1:nx
                 [mm,im] = max(p(j) < xcdf);
                 xrand(j) = obj.x(im);
             end
         end
        
        
    end
    
end

