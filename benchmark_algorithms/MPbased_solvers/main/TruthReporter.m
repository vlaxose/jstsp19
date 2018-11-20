classdef TruthReporter < EstimIn
    % A debugging aid that reports on the "in-flight" estimates of the 
    % input while GAMP is running, comparing against a known signal.
    properties
        est; % wrapped EstimIn object
        truth; % The true signal that we hope to recover
        it; % iteration number
    end

    methods
        function obj = TruthReporter(est,truth)
            obj = obj@EstimIn;
            obj.est = est;
            obj.truth = truth;
            obj.it = 0;
        end

        function [mean0, var0, valInit] = estimInit(obj)
            [mean0, var0, valInit] = obj.est.estimInit();
            obj.it = 0;
        end

        function [xhat, xvar, val] = estim(obj, rhat, rvar)
            % Get the estimate from the wrapped estimator and report about it
            [xhat, xvar, val] = obj.est.estim(rhat, rvar); 

            obj.it = obj.it + 1;% iteration count
            errCor = corr(rhat-obj.truth,obj.truth);  % how correlated is the rhat error? (with matched MMSE estimators this should go to zero)
            if ~isreal(errCor); errCor = abs(errCor);end
            rhat_mse = norm(rhat - obj.truth)^2 / length(obj.truth); % This should be close to rvar
            xhat_mse = norm(xhat - obj.truth)^2 / length(obj.truth); % This should be close to xvar

            fprintf('%4d:corr(rhat-x,x)=%+4.2f, rvar=%6e, rhat MSE/rvar=%6.4f, xhat MSE/xvar=%6.4f, xhat NMSE=%6.2f dB\n' ,...
                obj.it, errCor, ...
                mean(rvar),rhat_mse/mean(rvar) , ...
                xhat_mse/mean(xvar) , ...
                20*log10(norm(xhat-obj.truth,'fro')/norm(obj.truth,'fro') ) );
        end

        % TODO: Decide what to do with plikey and loglikey. They may or may not be present in obj.est
    end
end
