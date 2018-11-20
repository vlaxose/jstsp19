classdef EstimInAvg < handle
    % EstimInAvg:  For a given rvar and input estimator of the form
    %                 [xhat,xvar] = estimIn(rhat,rvar),
    %              this function computes the average output MSE & xvar,
    %                 E[|xhat-x|^2] & E[xvar],
    %              over rhat = x + Nor(0,rvar), for a given test set x.

    properties 
        inEst;  % input estimator
        x;      % samples of signal x~p(x)
        w;      % samples of noise w~Nor(0,1)
    end

    methods
        % Constructor
        function obj = EstimInAvg(inEst,x)
            obj.inEst = inEst;
            obj.x = x; 

            % draw w~N(0,1)
            if isreal(obj.x)
                obj.w = randn(size(x)); 
            else
                obj.w = (1/sqrt(2))*(randn(size(x))+1i*randn(size(x)));
            end
        end

        % Compute average 
        %
        % mse = E[ |x-g(x+e;rvar)|^2 ] with expectation over
        %                              x~p(x) and e~Nor(0,rvar)
        %
        function [mse,xvar] = mse(obj, rvar, rvarTrue)
       
            % handle inputs
            if nargin<3, rvarTrue = rvar; end

            % compute estimates of x
            [xhat,xvar] = obj.inEst.estim(obj.x+sqrt(rvarTrue)*obj.w,rvar);
            xvar = mean(xvar(:));

            % evaluate MSE 
            mse = mean(abs(obj.x(:)-xhat(:)).^2);
        end 
    end
end
