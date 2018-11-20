classdef MaskedEstimOut < handle
    % MaskedEstimOut: Output estimator class with missing data


    
    properties
        
        %Base output estimator object with no missing data, should be an
        %EstimOut object
        estim1; 
        
        %Logical mask. Each entry is true if that element of Y is observed,
        %and false if not. Dimensions should be MxL
        omega;
        
    end
    
    methods
        % Constructor
        function obj = MaskedEstimOut(estim1,omega)
            obj.estim1 = estim1;
            obj.omega = omega;
        end
        
        %Estimation method
        function [zhat,zvar] = estim(obj,phat,pvar)
            
            %First leverage method of base object
            [zhat,zvar] = obj.estim1.estim(phat,pvar);
            
            %Now use omega to fix entries of unobserved Y
            if numel(phat) > 1
                zhat(~obj.omega) = phat(~obj.omega);
                zvar(~obj.omega) = pvar(~obj.omega);
            else
                zhat(~obj.omega) = phat;
                zvar(~obj.omega) = pvar;
            end
            
        end
        
        % Compute log likelihood
        %   E( log p_{Y|Z}(y|z) )
        function ll = logLike(obj,zhat,zvar)
            
            %Temporarily just call the underlying method
            ll = obj.estim1.logLike(zhat,zvar);
            
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
                   
            %Temporarily just call the underlying method
            ll = obj.estim1.logScale(Axhat,pvar,phat);       
            
        end
        
    end
    
end

