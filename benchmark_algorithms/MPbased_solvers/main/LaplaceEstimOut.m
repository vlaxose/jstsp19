classdef LaplaceEstimOut < EstimOut
    % LaplaceEstimOut:  Additive White Laplacian scalar output estimation function
    %
    % Corresponds to an output channel of the form
    %   y = z + L(0,lambda)
    %   where L(x;0,lambda) = lambda/2*exp(-1*lambda*abs(x))
    
    properties
        % Prior mean and variance
        y;      % Measured output
        lambda;   % Rate parameter
        
        % True indicates to compute output for max-sum
        maxSumVal = false;
    end
    
    methods
        % Constructor
        function obj = LaplaceEstimOut(y, lambda, maxSumVal)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.y = y;
                obj.lambda = lambda;
                if (nargin >= 3)
                    if (~isempty(maxSumVal))
                        obj.maxSumVal = maxSumVal;
                    end
                end
                %Warn user about inputs
                if any(~isreal(obj.y)),
                    error('First argument of LaplaceEstimOut must be real-valued.');
                end;
                if any(any((obj.lambda<0)))||any(any(~isreal(obj.lambda))),
                    error('Second argument of LaplaceEstimOut must be non-negative');
                end;
                if any(obj.lambda==0)
                    warning(['Tiny non-zero rates will be used for'...
                        ' computing log likelihoods. May cause problems'...                  
                        ' with adaptive step size if used.']) %#ok<*WNTAG>
                    obj.lambda = max(obj.lambda,1e-20);
                end
            end
        end
        
        % Size
        function [nz,ncol] = size(obj)
            [nz,ncol] = size(obj.y);
        end
        
        % AWL Estimation function
        % Provides the posterior mean and variance of variable z
        % from an observation y = z + w, z = N(zmean0,zvar0), w = L(0,lambda)
        function [zmean, zvar] = estim(obj, zmean0, zvar0)
            
            if obj.maxSumVal
                %Compute the threshold
                thresh = obj.lambda.*zvar0;
                %shift the mean parameter by y
                ztil = zmean0 - obj.y;

                %Compute MAP estimate
                zmean = obj.y + max(0,abs(ztil)-thresh) .* sign(ztil);
                
                %Compute variance parameter
                zvar = zvar0.*(abs(zmean - obj.y) > 0);
                
            else
                % To avoid numerical problems (0/0) when evaluating
                % ratios of Gaussian CDFs, impose a firm cap on the
                % maximum value of entries of pvar
                zvar0 = min(zvar0, 700);

                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean and variance depend
                sig = sqrt(zvar0);                       	% Gaussian prod std dev
                muL = zmean0 - obj.y + obj.lambda.*zvar0;   % Lower integral mean
                muU = zmean0 - obj.y - obj.lambda.*zvar0;   % Upper integral mean
                muL_over_sig = muL ./ sig;
                muU_over_sig = muU ./ sig;
                cdfL = normcdf(-muL_over_sig);              % Lower cdf
                cdfU = normcdf(muU_over_sig);               % Upper cdf
                cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
                SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*zvar0) ) .* ...
                    cdfRatio;
                NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants


                % For the "trouble" constants (those with muL's and muU's
                % that are too large to give accurate numerical answers),
                % we will effectively peg the special constant to be Inf or
                % 0 based on whether muL dominates muU or vice-versa
                SpecialConstant(NaN_Idx & (-muL >= muU)) = Inf;
                SpecialConstant(NaN_Idx & (-muL < muU)) = 0;

                % Compute the ratio normpdf(a)/normcdf(a) for
                % appropriate upper- and lower-integral constants, a
                RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
                RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));

                % Now compute the first posterior moment...
                zmean = obj.y +  (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                    (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                    (muU + sig.*RatioU);

                % ...and second central posterior moment
                varL = zvar0 .* (1 - RatioL.*(RatioL - muL_over_sig));
                varU = zvar0 .* (1 - RatioU.*(RatioU + muU_over_sig));
                meanL = muL - sig.*RatioL;
                meanU = muU + sig.*RatioU;
                SecondMoment = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                    (varL + meanL.^2) + (1 ./ (1 + SpecialConstant)) .* ...
                    (varU + meanU.^2);

                %Add correction for non-zero mean
                SecondMoment = SecondMoment + ...
                    2.*obj.y.*zmean - abs(obj.y).^2;

                %Finish up
                zvar = SecondMoment - zmean.^2;
            
            end
            
            
        end
        
        % Compute log likelihood
        % For sum-product compute
        %   E_z( log p_{Y|Z}(y|z) ) with Z ~ N(zmean0, zvar)
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = zhat
        function ll = logLike(obj,zmean0,zvar)
            
            
            %Handle max sum case
            if obj.maxSumVal
                ll =  abs(obj.y - zmean0);
            else %otherwise, address sum-product
                % To avoid numerical problems (0/0) when evaluating
                % ratios of Gaussian CDFs, impose a firm cap on the
                % maximum value of entries of rvar
                zvar0 = min(zvar, 700);
                
                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean and variance depend
                sig = sqrt(zvar0);                       	% Gaussian prod std dev
                mu = zmean0 - obj.y;                        %  integral mean
                mu_over_sig = mu ./ sig;
                cdfL = normcdf(-mu_over_sig);              % Lower cdf
                cdfU = normcdf(mu_over_sig);               % Upper cdf
                SpecialConstant = cdfL ./ cdfU; %          Simpler for this case
                
                % Compute the ratio normpdf(a)/normcdf(a) for
                % appropriate upper- and lower-integral constants, a
                RatioL = 2/sqrt(2*pi) ./ erfcx(mu_over_sig / sqrt(2));
                RatioU = 2/sqrt(2*pi) ./ erfcx(-mu_over_sig / sqrt(2));
                
                % Compute mean
                ll = (-1 ./ (1 + SpecialConstant.^(-1))) .* ...
                    (mu - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                    (mu + sig.*RatioU);
                                
            end
      
            %Finish
            ll = -1*obj.lambda.*ll + log(0.5*obj.lambda);
            
            
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
            
            %error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');      
                   
             if ~(obj.maxSumVal)
                % Find the fixed-point of phat
                opt.phat0 = Axhat; % works better than phat
                opt.alg = 1; % approximate newton's method
                % For large rates needs more iterations to make progress
                opt.maxIter = max(50,floor(100*log10(obj.lambda))); 
                opt.tol = 1e-3; 
                opt.stepsize = 1; 
                opt.regularization = 1./(obj.lambda)^2;  % works well up to obj.lambda = 1e4
                opt.debug = false;
                phatfix = estimInvert(obj,Axhat,pvar,opt);

                % Begin by computing various constants on which the
                % posterior mean and variance depend
                sig = sqrt(pvar);              % Gaussian prod std dev
                mu = phatfix - obj.y;           % Shift
                muL = mu + obj.lambda.*pvar;   % Lower integral mean
                muU = mu - obj.lambda.*pvar;   % Upper integral mean
                muL_over_sig = muL ./ sig;
                muU_over_sig = muU ./ sig;
                
                %Calculate lower and Upper non-shared integration factors
                BL = erfcx(muL_over_sig/sqrt(2));
                BU = erfcx(-muU_over_sig/sqrt(2));
                % The individual terms can still be infinite. For these
                % cases use the approximation erfc(x) = 1/sqrt(pi)/x for
                % large x
                I = find(isinf(BL));
                BL(I) = 1./(sqrt(pi/2)*abs(muL_over_sig(I)));
                I = find(isinf(BU));
                BU(I) = 1./(sqrt(pi/2)*abs(muU_over_sig(I)));

                % Calculate the log scale factor
                ls = log(obj.lambda/4) - mu.^2./(2*pvar) + log(BL + BU);

                % Combine to form output cost
                ll = ls + 0.5*(Axhat - phatfix).^2./pvar;
            else 
                ll =  abs(obj.y - Axhat);
            end     
            
        end
        
        function S = numColumns(obj)
            %Return number of columns of Y
            S = size(obj.y,2);
        end
        
        % Generate random samples from p(y|z)
        function y = genRand(obj, z)
            
            %Use CDF
            U = rand(size(z)) - 0.5;
            y = z - sign(U).*log(1 - 2*abs(U))./obj.lambda;
        end
        
        %Estimate rate parameter using EM algorithm
        %This could should use phat,pvar computed by a run of GAMP
        function lam = updateRate(obj,zmean0,zvar0)
            
            % To avoid numerical problems (0/0) when evaluating
            % ratios of Gaussian CDFs, impose a firm cap on the
            % maximum value of entries of rvar
            zvar0 = min(zvar0, 700);
            
            % *********************************************************
            % Begin by computing various constants on which the
            % posterior mean and variance depend
            sig = sqrt(zvar0);                       	% Gaussian prod std dev
            muL = zmean0 - obj.y + obj.lambda.*zvar0;   % Lower integral mean
            muU = zmean0 - obj.y - obj.lambda.*zvar0;   % Upper integral mean
            muL_over_sig = muL ./ sig;
            muU_over_sig = muU ./ sig;
            cdfL = normcdf(-muL_over_sig);              % Lower cdf
            cdfU = normcdf(muU_over_sig);               % Upper cdf
            cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
            SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*zvar0) ) .* ...
                cdfRatio;
            NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants
            
            
            % For the "trouble" constants (those with muL's and muU's
            % that are too large to give accurate numerical answers),
            % we will effectively peg the special constant to be Inf or
            % 0 based on whether muL dominates muU or vice-versa
            SpecialConstant(NaN_Idx & (-muL >= muU)) = Inf;
            SpecialConstant(NaN_Idx & (-muL < muU)) = 0;
            
            % Compute the ratio normpdf(a)/normcdf(a) for
            % appropriate upper- and lower-integral constants, a
            RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
            RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));
            
            % Compute mean
            absZminusYMean = (-1 ./ (1 + SpecialConstant.^(-1))) .* ...
                (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                (muU + sig.*RatioU);
            
            %Compute lam
            lam = 1 / mean(absZminusYMean(:));
            
        end
        
    end
    
end

