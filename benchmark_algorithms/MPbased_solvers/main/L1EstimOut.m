classdef L1EstimOut < EstimOut
    % L1EstimOut:  Output estimator for an l1-penalty
    %
    % Corresponds to the max-sum estimator with a function of the form
    %   fout(z) = -scale* sum(abs(z))
    properties (Access=private)
        scale = 1;  % scale factor
        
        % Determine scale factor automatically
        % This feature is still under experimentation.
        autoScale = false;
        scaleMax, scaleMin;
        nitMax = 5;
    end
        
    
    methods
        % Constructor
        function obj = L1EstimOut(scale)
            obj = obj@EstimOut;
            if (nargin >= 1)
                obj.scale = scale;
            end
        end
        
        % Set autoscale properties
        function setAutoScale(obj, scaleMin, scaleMax, nitMax)
            obj.autoScale = true;
            obj.scaleMin = scaleMin;
            obj.scaleMax = scaleMax;
            if (nargin >= 4)
                obj.nitMax   = nitMax;
            end
        end
        
        % Get autoscale
        function scale = getScale(obj)
            scale = obj.scale;
        end
        
        % Quadratic estimation function.  Given zmean and zvar compute minimum  
        %   zhat = argmax [ fout(z) + (1/2*zvar0)*sum( abs(z-zhat0).^2 )
        %   zvar = zvar0/(1 - fout''(zhat)*zvar0)        
        function [zhat, zvar] = estim(obj, zhat0, zvar0)
            if (obj.autoScale)
                nit = obj.nitMax;
            else
                nit = 1;
            end

            % Loop to find optimal autoscale
            for it = 1:nit
                sat = obj.scale * zvar0;
                zabs = abs(zhat0);
                I = (zabs >= sat);
                zhat = sign(zhat0).*(zabs - sat).*I;
                zvar = I.*zvar0;            
                
                if (obj.autoScale)
                    obj.scale = max(min(1/mean(abs(zhat)), obj.scaleMax), obj.scaleMin);
                end
            end            
        end
        
        
        % Compute log likelihood
        %   ll = sum( fout(zhat) )
        function ll = logLike(obj,zhat, zvar)
            ll = sum(-(obj.scale)*(sum(abs(zhat))));
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function LScale = logScale(obj,Axhat,pvar,phat)
                   
            error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');         
            
        end
        
    end
    
end
