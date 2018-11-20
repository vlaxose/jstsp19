classdef AwgnEstimOutAvg < IntEstimOutAvg
    % output estimator numerical integration class for use with gampSE 
    % adapted from NLEstimOutAvg
    properties
        wvar = 1;
    end

    methods
        function obj = AwgnEstimOutAvg(wvar,np,ny,nz)
            obj = obj@IntEstimOutAvg(np,ny,nz);
            obj.wvar = wvar;
        end

        % Estimation function: calculate posterior 
        % Expected Value (zhat) and Posterior Variance 
        % given y, p and taup
        function [zhat,zvar] = estim(obj,y,p,taup)
            var_sum = taup + obj.wvar;
            zhat = ( y.* taup + p.*obj.wvar) ./ var_sum;
            zvar = taup .*obj.wvar ./ var_sum;
        end

        % Points on y and their associated p(y|z) conditional pdf (scaled OK)
        % pyz(i,j) = P(y(i)|z(j))*const
        function [y,pyz] = getypts(obj,z,pz,ny)
            zmean = pz'*z;
            zvar = pz'*(z-zmean).^2;
            yvar = zvar + obj.wvar;

            ymax = sqrt(2*log(ny/2));
            % This choice of ymax sets the pmf endpoints to 
            % 2/ny/sqrt(2*pi) =~ .8/ny .., don't ask me why
            y = zmean + linspace(-ymax,ymax,ny)'*sqrt(yvar);

            pyz = exp(-(repmat(y,1,length(z))-repmat(z',ny,1)).^2/(2*obj.wvar)) / sqrt(2*pi*obj.wvar);
            % could skip the /sqrt(2*pi*obj.wvar) since IntEstimOutAvg normalizes 
        end
    end
end
