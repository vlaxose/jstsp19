classdef PoissonNLOutAvg < IntEstimOutAvg
    % PoissonNLOutAvg:  Avergaing function for a nonlinear + Poisson output channel
    %
    % The measurements are assumed to be:
    %   y = poissrnd(v)
    %   v = exp( polyval(lam, u) ) u = 1./(1+exp(-z))
    properties (Access = private)
        lam;   % polynomial coefficients        
    end
    
    methods
        % Constructor
        function obj = PoissonNLOutAvg(lam,np,ny,nz)
            
            % Parent constructor
            obj = obj@IntEstimOutAvg(np,ny,nz);
            
            % Set properties
            obj.lam = lam;                
        end
        
        % Estimation function given y, p and taup
        function [zhat,zvar] = estim(obj,y,p,taup)
            % Construct estimator
            outEst = PoissonNLEstim(y);
            
            % Set true value of lambda and disable adaptation
            outEst.setParam('lam', obj.lam, 'adapt', false);
            
            % Compute estimates
            [zhat,zvar] = outEst.estim(p,taup); 
            
            if (any(isnan(zhat)) || any(isnan(zvar)) )
                keyboard;
            end
            
        end
        
        % Compute discrete y points on which to perform integration
        % likelihood P(y(iy)|z(iz)) = pyz(iy,iz)
        function [y,pyz] = getypts(obj,z,pz,ny)
            
            % Compute rate for each z point
            u = 1./(1+exp(-z));
            logv = polyval(obj.lam, u);
            v = exp( logv );
            
            % Compute likelihood p(y|z)
            y = (0:ny-1)';
            nz = length(z);
            logpyz = zeros(ny,nz);
            logpyz(1,:) = -v;
            for iy=2:ny
                logpyz(iy,:) = logpyz(iy-1,:) + logv' -log(iy-1);
            end
            
            % Normalize
            pyz = zeros(ny,nz);
            for iz = 1:nz
                logpyz(:,iz) = logpyz(:,iz) - max(logpyz(:,iz));
                pyz(:,iz) = exp(logpyz(:,iz));
                pyz(:,iz) = pyz(:,iz)/sum(pyz(:,iz))*pz(iz);                
            end
            if (any(any(isnan(pyz))))
                keyboard;
            end
            
        end
        
        
    end
end