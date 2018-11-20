classdef EstimOutAvg2 < handle
    % EstimOutAvg:  For a given estimOut of the form
    %                 [zhat,zvar] = estimOut(phat,pvar),
    %              this routine computes the average output MSE & zvar, i.e.,
    %                 E[|zhat-z|^2] & E[zvar],
    %              over z=phat+Nor(0,pvar), for a given test set z & pvar.

    properties 
        outEst; % output estimator
        z;      % signal samples 
        Ez2;   % empirical 2nd moment of z
        w;      % samples of noise w~Nor(0,1)
    end

    methods
        % Constructor
        function obj = EstimOutAvg2(outEst,z)
            obj.outEst = outEst;
            obj.z = z; 
            obj.Ez2 = mean(abs(z).^2,1); 

            % draw w~N(0,1)
            if isreal(obj.z)
                obj.w = randn(size(z)); 
            else
                obj.w = (1/sqrt(2))*(randn(size(z))+1i*randn(size(z)));
            end
        end

        % Compute average 
        %
        % mse = E[ |z - g(z-e;pvar)|^2 ] with expectation over
        %                               z and e~Nor(0,pvarTrue)
        %
        function [mse,zvar] = mse(obj, pvar, pvarTrue)
       
            % handle inputs
            if nargin<3, pvarTrue = pvar; end

            % generate phat and error e such that 
            %   z = phat+e
            %   E[e]=0 
            %   var[e]=pvarTrue
            %   E[e*phat] = 0
            % by constructing 
            %   e = a*w + b*z ... not actually needed
            %   phat = -a*w + (1-b)*z 
            % with coefficients (a,b) such that
            %   a^2 + Ez2*b^2 = pvarTrue
            %   a^2 = b(1-b)*Ez2
            b = min(1,pvarTrue./obj.Ez2);
            a = sqrt(b.*(1-b).*obj.Ez2);
            phat = -bsxfun(@times,a,obj.w) + bsxfun(@times,(1-b),obj.z);
            % e = a*obj.w + b*obj.z;

            % compute estimates of z
            [zhat,zvar] = obj.outEst.estim(phat,pvar);
            zvar = mean(zvar(:));

            % evaluate MSE 
            mse = mean(abs(obj.z(:)-zhat(:)).^2);
        end 
    end
end
