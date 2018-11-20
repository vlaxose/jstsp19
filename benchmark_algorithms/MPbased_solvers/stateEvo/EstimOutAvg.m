classdef EstimOutAvg < handle
    % EstimOutAvg:  Given a likeType in {AWGN,Probit,...} and associated
    %               likeParams, and given vector phat and scalar pvarTrue 
    %               and pvar, this routine internally generates 
    %                          z = phat + N(0,pvarTrue)
    %                          y ~ p(y|z;likeParams) ,
    %               establishes an output estimator of the form
    %                [zhat,zvar] = estimOut(phat,pvar;likeParams)
    %               empirically computes the MSE
    %                        MSE = E[|zhat-z|^2] 
    %               and finally reports the MSE and average zvar.

    properties 
        likeType; % output channel type
        likeParams; % output channel parameters
                    % --AWGN: likeParams = wvar
                    % --Probit: likeParams = wvar
        phat;   % samples of psuedo-measurements phat
        e;      % samples of noise e~Nor(0,1)
        w;      % samples of noise w~Nor(0,1)
    end

    methods
        % Constructor
        function obj = EstimOutAvg(likeType,likeParams,phat)
            obj.likeType = likeType;
            obj.likeParams = likeParams;
            obj.phat = phat; 

            % draw w~N(0,1)
            if isreal(phat)
                obj.e = randn(size(phat)); 
                obj.w = randn(size(phat)); 
            else
                obj.e = (1/sqrt(2))*(randn(size(phat))+1i*randn(size(phat)));
                obj.w = (1/sqrt(2))*(randn(size(phat))+1i*randn(size(phat)));
            end
        end

        % Compute MSE:
        function [mse,zvar] = mse(obj, pvar, pvarTrue)
       
            % handle inputs
            if nargin<3, pvarTrue = pvar; end

            % create noiseless measurements z 
            z = obj.phat + sqrt(pvarTrue)*obj.e;

            % create noisy measurements & estimator
            switch obj.likeType
              case 'AWGN'
                % in this case, likeParams = measurement noise variance
                wvar = obj.likeParams;
                if isreal(obj.phat)
                  y = z + sqrt(wvar)*obj.w;
                  EstimOut = AwgnEstimOut(y,wvar);
                else
                  y = z + sqrt(wvar)*obj.w;
                  EstimOut = CAwgnEstimOut(y,wvar);
                end
              case 'Probit'
                % in this case, likeParams = measurement noise variance
                wvar = obj.likeParams;
               %w = sqrt(wvar)*randn(size(z));
                y = ((z+sqrt(wvar)*obj.w)>0);
                EstimOut = ProbitEstimOut(y,0,wvar);
              otherwise
                error('likeType not supported')
            end

            % compute estimates of z
            [zhat,zvar] = EstimOut.estim(obj.phat,pvar);
            zvar = mean(zvar(:));

            % evaluate MSE 
            mse = mean(abs(z(:)-zhat(:)).^2);
        end 
    end
end
