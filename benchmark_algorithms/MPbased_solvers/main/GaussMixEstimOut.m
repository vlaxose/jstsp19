classdef GaussMixEstimOut < EstimOut
    % GaussMixEstimOut:  Gaussian Mixture output estimation
    %
    % Corresponds to an output channel of the form
    %   y = z + q
    %
    % Where q has the density
    % p(q) = (1-lambda) Nor(0,nu0) + lambda Nor(0,nu1)
    
    properties
        
        % Measured data
        Y;
        
        %Variances
        nu0;
        nu1;
        
        %on probability
        lambda;
        
    end
    
    methods
        % Constructor
        function obj = GaussMixEstimOut(Y,nu0,nu1,lambda)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.Y = Y;
                obj.nu0 = nu0;
                obj.nu1 = nu1;
                obj.lambda = lambda;
            end
            
            %Warn user about inputs
            if any(~isreal(obj.Y(:))),
                error('First argument of GaussEstimOut must be real-valued.  Did you mean to use CGaussMixEstimOut instead?');
            end;
            if (any(obj.nu0(:) <= 0) || any(obj.nu1(:) <= 0) || any(~isreal(obj.nu0(:))) || any(~isreal(obj.nu1(:))))
                error('Variance arguments of GaussMixEstimOut must be real and positive');
            end;
            if (any(obj.lambda(:) <= 0) || any(obj.lambda(:) >= 1))
                error('Corruption rate lambda must be strictly in (0,1). If exactly 0 or 1, use AwgnEstimOut Instead');
            end
            
            %Automatically format so nu0 < nu1
            %Find indeces where this is not true
            ind = find(obj.nu0 > obj.nu1);            
            %Swap elements of nu0 and nu1
            tmp = obj.nu0;
            obj.nu0(ind) = obj.nu1(ind);
            obj.nu1(ind) = tmp(ind);
            %Compute the correct lambda
            obj.lambda(ind) = 1 - obj.lambda(ind);
            
        end
        
        
        
        % Estimation function
        % Computes the posterior mean and variance given the estimates phat
        % and pvar
        function [zhat, zvar, p1] = estim(obj, phat, pvar)
            
            %Compute the intrinsic LLR
            int_llr = log(obj.lambda./(1 - obj.lambda));

            %Compute constant needed for various computations
            C = (obj.nu1 - obj.nu0)./((obj.nu0./pvar+1).*(obj.nu1 + pvar));

            %Compute the extrinsic LLR
            ext_llr = 0.5*log((obj.nu0 + pvar) ./ (obj.nu1 + pvar)) ...
                + 0.5*(obj.Y - phat).^2.*C./pvar;
            
%             %Old fix
%             %Limit ext_llr
%             ext_llr = min(100,max(-100,ext_llr));

            % compute log likelihood ratio
            llr = int_llr + ext_llr;

            %Now compute p1 and p0
            p1 = 1 ./ (1 + exp(-llr));

            %Compute posterior mean of the first mixture component
            zeta0 = (obj.Y .* pvar + phat.*obj.nu0) ./ (obj.nu0 + pvar);

            %Compute first posterior moment
            zhat = zeta0 + p1.*C.*(phat - obj.Y);

            %Compute constant (1-p1)*p1
            D = 1./(exp(0.5*llr) + exp(-0.5*llr)).^2;
            %D = 1./(2 + exp(llr) + exp(-llr));

            %Compute the variance of the posterior
            zvar = D.*(obj.Y - phat).^2.*C.^2 ...
                + pvar.*(obj.nu0./(obj.nu0 + pvar) + p1.*C);
            
%             %Old "fix"   
%             %Protect form zvar getting too large or negative
%             rval = 0.99;
%             zvar = max(min(zvar,rval*pvar),0);           
        end
        
        
        
        % Compute log likelihood
        %   E( log p_{Y|Z}(y|z) )
        function ll = logLike(obj,zhat,zvar)
            
            %Here we implement a lower bound to the expected log
            %likelihood. We exploit the identity
            %log(a+b) = log(a) + log(1 + b/a) and then approximate
            %log(1 + b/a) by log(b/a) where we limit the domain of
            %integration for this term to the region where b/a > 1.
            
            %Define cdf for Gaussian
            std_normal_cdf = @(x) 1/2*(1 + erf(x/sqrt(2)));
            
            %Define density of Gaussian
            std_normal_pdf = @(x) 1/sqrt(2*pi)*exp(-x.^2/2);
            
            
            %First, compute the nu1 term analytically
            ll = log(obj.lambda./sqrt(2.*pi.*obj.nu1)) -...
                1./(2.*obj.nu1).*(zvar + (zhat-obj.Y).^2);
            
            %Determine bounds of integration. We will integrate from
            %y - alpha to y + alpha
            alpha = sqrt(max(0,2.*log((1-obj.lambda).*sqrt(obj.nu1)./ ...
                (obj.lambda)./sqrt(obj.nu0)).* ...
                obj.nu0.*obj.nu1./(obj.nu1 - obj.nu0)));
            
            %Compute several required quantities
            pp1 = std_normal_pdf((obj.Y - alpha - zhat)./sqrt(zvar));
            pp2 = std_normal_pdf((obj.Y + alpha - zhat)./sqrt(zvar));
            c2 = std_normal_cdf((obj.Y - alpha - zhat)./sqrt(zvar));
            c1 = std_normal_cdf((obj.Y + alpha - zhat)./sqrt(zvar));
            beta = c1 - c2;
            
            %Compute mean and variance of the truncated normal distribution
            mval = zhat + sqrt(zvar).*(pp1 - pp2)./beta;
            vval = zvar.*(1 ...
                + ((obj.Y - alpha - zhat)./sqrt(zvar).*pp1 -...
                (obj.Y + alpha - zhat)./sqrt(zvar).*pp2)./beta...
                - ( (pp1 - pp2) ./ beta ).^2);
            
            %Compute the the nu0 term
            val = log((1 - obj.lambda).*sqrt(obj.nu1)./ ...
                obj.lambda./sqrt(obj.nu0)) +...
                (obj.nu0 - obj.nu1)./2./obj.nu0./obj.nu1.* ...
                (vval + (mval - obj.Y).^2);
            val = beta .* val;
            
            %Zero out entries where beta = 0
            val(beta == 0) = 0;
            
            %Combine the results
            ll = ll + val;
            
        end
       
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)

            % Find the fixed-point of phat
            opt.alg = 1; 
            opt.maxIter = 50; 
            opt.stepsize = 0.25; 
            opt.regularization = min(obj.nu0,obj.nu1)^2;
            opt.phat0 = phat; 
            opt.tol = 1e-4; 
            opt.debug = false; 
            phatfix = estimInvert(obj,Axhat,pvar,opt);

            %Compute int_z p_{Y|Z}(y|z) N(z;phatfix, pvar),
            % Old way of computing log Scale pre 10/01/14
%             scaleFac = (1-obj.lambda).*exp(-0.5./(obj.nu0+pvar)...
%                 .*abs(phatfix-obj.Y).^2)./sqrt(2*pi*(obj.nu0+pvar)) + ...
%                 obj.lambda.*exp(-0.5./(obj.nu1+pvar)...
%                 .*abs(phatfix-obj.Y).^2)./sqrt(2*pi*(obj.nu1+pvar));
% 
%             % Combine to form output cost
%             ll = log(scaleFac) + 0.5*(Axhat - phatfix).^2./pvar;
            
            %Compute int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
            %Done robustly, Uses log(a +b) = log(a) + log(1 +b/a) for
            %positive a,b
            sumvar0 = pvar + obj.nu0;
            sumvar1 = pvar + obj.nu1;
            ratio = (1-obj.lambda)./obj.lambda.*sqrt(sumvar1./sumvar0)...
                .*exp(0.5*(obj.nu0-obj.nu1).*(obj.Y-phatfix).^2./sumvar0./sumvar1);
            ls = log(obj.lambda) - 0.5*log(2*pi*(sumvar1)) ...
                - 0.5*((obj.Y-phatfix).^2./sumvar1) + log(1 + ratio);

            % Combine to form output cost
            ll = ls + 0.5*(Axhat - phatfix).^2./pvar;
        end
        
        function S = numColumns(obj)
            %Return number of columns of Y
            S = size(obj.Y,2);
        end
        
        % Generate random samples from p(y|z)
        function y = genRand(obj, z)
            smallNoise = sqrt(obj.nu0).*randn(size(z));
            largeNoise = sqrt(obj.nu1).*randn(size(z));
            largeInds = (rand(size(z)) < obj.lambda);
            Noise = smallNoise;
            Noise(largeInds) = largeNoise(largeInds);
            y = z + Noise;
        end
    end
    
end

