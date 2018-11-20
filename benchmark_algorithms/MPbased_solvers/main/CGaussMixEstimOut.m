classdef CGaussMixEstimOut < EstimOut
    % CGaussMixEstimOut:  Complex Gaussian Mixture output estimation
    %
    % Corresponds to an output channel of the form
    %   y = z + q
    %
    % where q has the density
    % p(q) = (1-lambda) CNorm(0,nu0) + lambda CNorm(0,nu1),
    % and CNorm(0,b) is a circularly symmetric complex Normal distribution
    
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
        function obj = CGaussMixEstimOut(Y,nu0,nu1,lambda)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.Y = Y;
                obj.nu0 = nu0;
                obj.nu1 = nu1;
                obj.lambda = lambda;

                %Warn user that logLike has not been implemented yet
                warning(['The logLike method has been implemented for this' ...
                    ' class, but not thoroughly tested.  May return incorrect' ...
                    ' cost when using adaptive step size.']) %#ok<WNTAG>
            end
        end
        
        
        
        % Estimation function
        % Computes the posterior mean and variance given the estimates phat
        % and pvar
        function [zhat, zvar] = estim(obj, phat, pvar)
            
            
            %Compute the intrinsic LLR
            int_llr = log(obj.lambda./(1 - obj.lambda));
            
            %Compute the extrinsic LLR
            ext_llr = log( (obj.nu0 + pvar) ./ (obj.nu1 + pvar)) ...
                + abs(obj.Y - phat).^2 .*...
                ( 1 ./ (obj.nu0 + pvar) - 1./ (obj.nu1 + pvar));
            
            %Limit ext_llr
            ext_llr = min(100,max(-100,ext_llr));
            
            %Now compute p1 and p0
            p1 = 1 ./ (1 + exp(-int_llr - ext_llr) + eps);
           
            
            %Compute p0
            p0 = 1 - p1;
            
            %We can now obtain the mean
            zeta0 = (obj.Y .* pvar + phat.*obj.nu0) ./ ...
                (obj.nu0 + pvar);
            
            zeta1 = (obj.Y .* pvar + phat.*obj.nu1) ./ ...
                (obj.nu1 + pvar);
            
            zhat = p0.*zeta0 + p1.*zeta1;
            
            %Compute variance
            zvar = p0.*(obj.nu0.*pvar ./ (obj.nu0 + pvar) + abs(zeta0).^2)...
                + p1.*(obj.nu1.*pvar ./ (obj.nu1 + pvar) + abs(zeta1).^2)...
                - abs(zhat).^2;
            
            
            %Protect form zvar getting too large or negative
            rval = 0.99;
            zvar = max(min(zvar,rval*pvar),0);                
            
            
        end                
        
        % Compute expected log-likelihood:
        %   E_Z( log p_{Y|Z}(y|z) ) with Z ~ CN(zhat, zvar)
        function ll = logLike(obj,zhat,zvar)
            
            %Here we implement a lower bound to the expected log
            %likelihood. We exploit the identity
            %log(a+b) = log(a) + log(1 + b/a) and then approximate
            %log(1 + b/a) by log(b/a) where we limit the domain of
            %integration for this term to the region where b/a > 1.
            
            %Define cdf for Gaussian
            normal_cdf = @(x) 1/2*(1 + erf(x/sqrt(2)));
            
            %Define density of Gaussian
            normal_pdf = @(x) 1/sqrt(2*pi)*exp(-x.^2/2);
            
            
            %First, compute the nu1 term analytically
            ll = log(obj.lambda./(pi.*obj.nu1)) -...
                (zvar + abs(zhat-obj.Y).^2)./obj.nu1;
            
            %Determine bounds of integration. We will integrate from
            %y - alpha to y + alpha, on real and imaginary axis
            alpha = sqrt(max(0,0.5*log((1-obj.lambda).*obj.nu1./ ...
                (obj.lambda)./obj.nu0).* ...
                obj.nu0.*obj.nu1./(obj.nu1 - obj.nu0)));
            alpha = alpha + 1i*alpha;
            
            %Compute several required quantities for real part
            pp1 = normal_pdf((real(obj.Y - alpha - zhat))./sqrt(zvar));
            pp2 = normal_pdf((real(obj.Y + alpha - zhat))./sqrt(zvar));
            c2 = normal_cdf((real(obj.Y - alpha - zhat))./sqrt(zvar));
            c1 = normal_cdf((real(obj.Y + alpha - zhat))./sqrt(zvar));
            RealBeta = c1 - c2;
            
            %Compute mean and variance of the truncated normal distribution
            %for real
            mval = real(zhat) + sqrt(zvar).*(pp1 - pp2)./RealBeta;
            vval = zvar.*(1 ...
                + (real(obj.Y - alpha - zhat)./sqrt(zvar).*pp1 -...
                real(obj.Y + alpha - zhat)./sqrt(zvar).*pp2)./RealBeta...
                - ( (pp1 - pp2) ./ RealBeta ).^2);
            
            %Compute cost of second component associated with real value         
            RealVal = RealBeta.*(obj.nu0 - obj.nu1)./2./obj.nu0./obj.nu1.* ...
                (vval + (mval - real(obj.Y)).^2);
            RealVal(RealBeta == 0) = 0;
            
            %Compute several required quantities fr imaginary part
            pp1 = normal_pdf((imag(obj.Y - alpha - zhat))./sqrt(zvar));
            pp2 = normal_pdf((imag(obj.Y + alpha - zhat))./sqrt(zvar));
            c2 = normal_cdf((imag(obj.Y - alpha - zhat))./sqrt(zvar));
            c1 = normal_cdf((imag(obj.Y + alpha - zhat))./sqrt(zvar));
            ImagBeta = c1 - c2;
            
            %Compute mean and variance of the truncated normal distribution
            % Imaginary part
            mval = imag(zhat) + sqrt(zvar).*(pp1 - pp2)./ImagBeta;
            vval = zvar.*(1 ...
                + (imag(obj.Y - alpha - zhat)./sqrt(zvar).*pp1 -...
                imag(obj.Y + alpha - zhat)./sqrt(zvar).*pp2)./ImagBeta...
                - ( (pp1 - pp2) ./ ImagBeta ).^2);
            
            %Compute cost of second component associated with imag value         
            ImagVal = ImagBeta.*(obj.nu0 - obj.nu1)./2./obj.nu0./obj.nu1.* ...
                (vval + (mval - imag(obj.Y)).^2);
            ImagVal(ImagBeta == 0) = 0;
            
            %Compute the total cost including additional term
            val = RealBeta.*ImagBeta.*log((1 - obj.lambda).*sqrt(obj.nu1)./ ...
                obj.lambda./sqrt(obj.nu0)) +...
                RealVal + ImagVal;
            
            %Combine the results
            ll = ll + val;
             
        end
        
        % Compute output cost:
        % For sum-product compute
        %   abs(Axhat-phatfix)^2/(pvar) + log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar) 
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

            %Compute int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar),
            scaleFac = (1-obj.lambda).*exp(-abs(phatfix-obj.Y).^2./(obj.nu0+pvar))...
                ./(pi*(obj.nu0+pvar)) + ...
                obj.lambda.*exp(-abs(phatfix-obj.Y).^2./(obj.nu1+pvar))...
                ./(pi*(obj.nu1+pvar));

            % Combine to form output cost
            ll = log(scaleFac) + abs(Axhat - phatfix).^2./pvar;       
            
        end
        
        % Number of columns
        function S = numColumns(obj)
            %Return number of columns of Y
            S = size(obj.Y,2);
        end
        
        % Size operator
        function [nz,ncol] = size(obj)
            [nz,ncol] = size(obj.Y);
        end
        
        % Generate random samples from p(y|z)
        function y = genRand(obj, z)
            smallNoise = sqrt(obj.nu0/2).*(randn(size(z)) + 1j*randn(size(z)));
            largeNoise = sqrt(obj.nu1/2).*(randn(size(z)) + 1j*randn(size(z)));
            largeInds = (rand(size(z)) < obj.lambda);
            Noise = smallNoise;
            Noise(largeInds) = largeNoise(largeInds);
            y = z + Noise;
        end
        
    end
    
end