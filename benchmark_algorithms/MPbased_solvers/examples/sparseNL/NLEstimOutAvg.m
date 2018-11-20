classdef NLEstimOutAvg < IntEstimOutAvg
    % NLOutEstim:  Avergaing function for a nonlinear + AWGN output channel
    %
    % The measurements are assumed to be y = outFn(z)+w,
    % where w = N(0,wvar) and outFn is specified as a function pointer.
    properties (Access = private)
        wvar;   % output variance
        outFn;  % nonlinear output function
        
        % Dimensions
        nzint=100;
        
        % 1=use linear estimator, 0=use true estimator
        useLin = false;
        outDeriv = 1;                
    end
    
    methods
        % Constructor
        function obj = NLEstimOutAvg(wvar,outFn,np,ny,nz,nzint,useLin,outDeriv)
            
            % Parent constructor
            obj = obj@IntEstimOutAvg(np,ny,nz);
            
            % Set properties
            obj.wvar   = wvar;
            obj.outFn  = outFn;
            obj.nzint = nzint;            
            if (nargin >= 7)             
                obj.useLin = useLin;
                obj.outDeriv = outDeriv;
            end
                
        end
        
        % Estimation function given y, p and taup
        function [zhat,zvar] = estim(obj,y,p,taup)
            
            if (obj.useLin)
                outEst = AwgnEstimOut(y, obj.wvar, [], obj.outDeriv);
                [zhat,zvar] = outEst.estim(p,taup);                
            else
                outEst = NLEstimOut(y,obj.wvar, obj.outFn, obj.nzint);
                [zhat,zvar] = outEst.estim(p,taup);
            end
            
            
        end
        
        % Compute discrete y points on which to perform integration
        % likelihood P(y(iy)|z(iz)) = pyz(iy,iz)
        function [y,pyz] = getypts(obj,z,pz,ny)
            
            % Estimate variance of y
            zmean0 = pz'*z;
            nz = length(z);
            v = obj.outFn(z);
            vmean = pz'*v;
            vvar = pz'*(v-vmean).^2;
            yvar = vvar + obj.wvar;
            
            % Use linear spacing
            ymax = sqrt(2*log(ny/2));
            y = obj.outFn(zmean0) + ...
                linspace(-ymax,ymax,ny)'*sqrt(yvar);
            
            % Compute likelihood
            pyz = exp(-(repmat(y,1,nz)-repmat(v',ny,1)).^2/(2*obj.wvar));
        end
        
        
    end
end