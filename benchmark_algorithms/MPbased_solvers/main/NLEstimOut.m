classdef NLEstimOut < EstimOut
    % NLOutEstim:  General nonlinear + AWGN output channel
    %
    % The measurements are assumed to be y = outFn(z)+w,
    % where w = N(0,wvar) and outFn is specified as a virtual function.
    properties (Access = private)
        y;      % Measured output
        wvar;   % output variance
        outFn;  % nonlinear output function
        nzint = 100;    % number of integration points for z
    end      
    
    %methods (Abstract)
        
        % Defines the output y = outFn(z) + w
        %y = outFn(v);
        
    %end
    
    methods 
        % Constructor
        function obj = NLEstimOut(y,wvar,outFn,nzint)
            obj = obj@EstimOut;
            obj.y = y;
            ny = length(y);
            if (length(wvar)==1)
                wvar = repmat(wvar,ny,1);
            end
            obj.wvar   = wvar;            
            obj.outFn  = outFn;
            if (nargin >= 4)
                obj.nzint = nzint;
            end
        end
        
        % Main estimation method:  Computes
        %   zhat = E( Z | Y)
        %   zvar = var( Z | Y )
        %
        % where Z = N(phat, pvar)
        function [zhat,zvar] = estim(obj,phat,pvar)
            
            % Get dimensions
            ny = min(length(phat),length(obj.y));  
            
            % Compute discrete points for z      
            umax = sqrt(2*log(obj.nzint/2));
            u = linspace(-umax,umax,obj.nzint)';
            logpu = -u.^2/2;
            
            % Main estimation loop
            zhat = zeros(ny,1);
            zvar = zeros(ny,1);
            for iy=1:ny
                
                % Compute posterior
                z = phat(iy) + sqrt(pvar(iy))*u;
                
                % Compute likelihood p(y|z)
                v = obj.outFn( z );
                logpyu = -(obj.y(iy)-v).^2/(2*obj.wvar(iy));
                
                % Compute posterior on U
                logpuy = logpyu + logpu;
                logpuy = logpuy - max(logpuy);
                puy = exp(logpuy);
                puy = puy / sum(puy);
                
                % Compute mean and expectation
                umean = puy'*u;
                uvar = puy'*((u-umean).^2);
                zhat(iy) = sqrt(pvar(iy))*umean + phat(iy);
                zvar(iy) = pvar(iy)*uvar;                                                
                
            end
            
        end
        
        % Log-likelihood:  The method should return 
        %   E( log p_{Y|Z}(y|Z) )  with Z = N(zhat,zvar)
        function ll = logLike(obj,zhat,zvar) 
            
            % Get dimensions
            ny = length(obj.y);            
            
            % Compute discrete points for z
            umax = sqrt(2*log(obj.nzint/2));
            u = linspace(-umax,umax,obj.nzint)';
            pu = exp(-u.^2/2);
            pu = pu/sum(pu);
            
            % Main estimation loop
            ll = zeros(ny,1);
            for iy=1:ny
                
                % Compute posterior
                z = zhat(iy) + sqrt(zvar(iy))*u;
                
                % Compute likelihood p(y|z)
                v = obj.outFn( z );
                logpyu = -(obj.y(iy)-v).^2/(2*obj.wvar(iy));
                ll(iy) = logpyu'*pu;                                                                          
                
            end
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
                   
            error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');         
            
        end
    end
end