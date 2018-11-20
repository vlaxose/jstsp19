classdef CMultAwgnEstimOut < EstimOut
    % CMultAwgnEstimOut:  Complex Multiplicative and Additive White Gaussian Noise scalar estimation function
    %
    % Corresponds to an output channel of the form
    %   y = z*CN(qhat,qvar) + CN(0, wvar)
    
    properties
        % Prior mean and variance
        y;      % Measured output
        wvar;   % Variance of additive noise
        qhat;   % Mean of multiplicative noise
        qvar;   % Variance of multiplicative noise
        scale = 1;
        % True indicates to compute output for max-sum
        maxSumVal = false;
    end
    
    methods
        % Constructor
        function obj = CMultAwgnEstimOut(y, wvar, qhat, qvar) % should add "maxSumVal" input
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.y = y;
                obj.wvar = wvar;
                obj.qhat = qhat;
                obj.qvar = qvar;
                %if (nargin >= 5)
                %    if (~isempty(maxSumVal))
                %        obj.maxSumVal = maxSumVal;
                %    end
                %end
                
                %Warn user if they set zero variance
                if any(obj.wvar == 0)
                    warning(['Tiny non-zero variances will be used for'...
                        ' computing log likelihoods. May cause problems'...
                        ' with adaptive step size if used.']) %#ok<*WNTAG>
                end
            end
        end
        
        % Size
        function [nz,ncol] = size(obj)
            [nz,ncol] = size(obj.y);
        end
        
        % CMultAwgn estimation function
        % Provides the posterior mean and variance of variable z with prior CN(phat,pvar)
        % from an observation y = f*g + w with g~CN(qhat,qvar) and w~CN(0,wvar)
        function [zhat, zvar] = estim(obj, phat, pvar)
            
            % Compute posterior mean and variance
            %yqvar = obj.wvar./abs(obj.qhat).^2 + obj.qvar.*abs(phat./obj.qhat).^2 + eps;
            yqvar = (obj.wvar + obj.qvar.*abs(phat).^2)./(abs(obj.qhat).^2) + eps;
            zvar = 1./(1./pvar + 1./yqvar);
            zhat = (phat./pvar + (obj.y./obj.qhat)./yqvar).*zvar;
        end
        
        % Compute log likelihood
        %   E( log p_{Y|Z}(y|z) ) with z ~ CN(z; phat, pvar) and p(y|z) = CN(z;ynew, wvarnew)
        %   where ynew = y./qhat and wvarnew = (qvar.*abs(Axhat).^2+wvar)./abs(qhat).^2
        %   so E( log p_{Y|Z}(y|z) ) = -( pvar + (ynew-phat)^2 )./wvarnew

        function ll = logLike(obj,phat,pvar)
            
            % Ensure variance is small positive number
            wvar1 = max(eps, obj.wvar);
            
            wvarnew = (wvar1+abs(phat).^2.*obj.qvar./abs(obj.qhat).^2);
            ynew = obj.y./obj.qhat;
            
            %Compute log-likelihood
            predErr = (abs(ynew-phat).^2 + pvar)./wvarnew - 2*log(abs(obj.qhat));
            ll = -(predErr); %return the values without summing
        end

        % Compute output cost:
        % For sum-product compute
        %   abs(Axhat-phatfix)^2/(pvar) + log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar) 
        %   and where p(y|z) = CN(z;ynew, wvarnew) 
        %   for ynew = y./qhat and wvarnew = (qvar.*abs(Axhat).^2+wvar)./abs(qhat).^2
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
                   
            % Compute output cost
            closed_form_approx = false;
            if closed_form_approx
                % Closed-form approximation (not exact due to phat dependence in wvarnew!)
                wvarnew = (obj.wvar+abs(phat).^2.*obj.qvar)./abs(obj.qhat).^2;
                ynew = obj.y./obj.qhat;
                ll = -log(pi*(pvar + wvarnew)) - abs(ynew - Axhat).^2./wvarnew - 2*log(abs(obj.qhat));
            else
                % Find the fixed-point of phat
                opt.phat0 = phat;
                opt.alg = 1; % approximate newton's method
                opt.maxIter = 500;
                opt.tol = 1e-4;
                opt.stepsize = 0.1;
                opt.regularization = obj.wvar.^2; 
                opt.debug = false;
                phatfix = estimInvert(obj,Axhat,pvar,opt);

                % Compute log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar)
                wvarnew = (obj.wvar+abs(phatfix).^2.*obj.qvar)./abs(obj.qhat).^2;
                ynew = obj.y./obj.qhat;
                %ls = -log(pi*(pvar + wvarnew)) - abs(ynew - phatfix).^2./(wvarnew+pvar) - 2*log(abs(obj.qhat));
                %ll = ls + abs(Axhat - phatfix).^2./pvar;
                ll = -log(pi*(pvar + wvarnew)) - abs(ynew - Axhat).^2./wvarnew - 2*log(abs(obj.qhat));
            end
        end


        function S = numColumns(obj)
            %Return number of columns of Y
            S = size(obj.y,2);
        end
        
        % Generate random samples from p(y|z)
        function y = genRand(obj, z)
            q = sqrt(obj.qvar/2).*randn(size(z)) + ...
                1j*sqrt(obj.wvar/2).*randn(size(z))+obj.qhat;
            y = sqrt(obj.wvar/2).*randn(size(z)) + ...
                1j*sqrt(obj.wvar/2).*randn(size(z)) + ...
                obj.scale.*z.*q;
        end
     end
    
end

