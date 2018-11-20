% CLASS: ElasticNetEstimIn
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimIn
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The ElasticNetEstimIn class defines a prior distribution on the signal,
%   x, that, in the MAP (max-sum) case, allows one to perform elastic net
%   penalized regression, as described in Zou and Hastie's paper,
%   "Regularization and Variable Selection via the Elastic Net."  Under a
%   Gaussian likelihood, the elastic net regression problem can be
%   expressed as
%   	min_x norm(y - Ax, 2) + lambda1*norm(x, 1) + lambda2*norm(x, 2),
%   where lambda1 and lambda2 are non-negative parameters.  In a Bayesian
%   context, the elastic net represents a prior on each coefficient, x_n,
%   of the form
%        p(x_n) \propto N(x_n; 0, (2*lambda2)^-1)*Lap(x_n; lambda1),
%   where N(x_n; 0, a) is a normal distribution with variance a, and
%   Lap(x_n; b) is a Laplace distribution with variance 2/b^2.
%   If maxSumVal = true (the default), then updates are performed according
%   to the max-sum GAMP update rules.  Otherwise, sum-product updates are
%   used.
%
% PROPERTIES (State variables)
%   lambda1     Non-negative ell-1 norm regularization penalty term
%               [Default: 1]
%   lambda2     Non-negative ell-2 norm regularization penalty term
%               [Default: 1]
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: true]
%
% METHODS (Subroutines/functions)
%   ElasticNetEstimIn()
%       - Default constructor.  Assigns lambda1, lambda2, and maxSumVal to 
%         default values.
%   ElasticNetEstimIn(lambda1)
%       - Optional constructor.  Sets lambda2 and maxSumVal to defaults.
%   ElasticNetEstimIn(lambda1, lambda2)
%       - Optional constructor.  Sets maxSumVal to default.
%   ElasticNetEstimIn(lambda1, lambda2, maxSumVal)
%       - Optional constructor.  Sets all custom values.
%   estim(obj, rhat, rvar)
%       - When maxSumVal = true, estim returns the two required max-sum
%         GAMP quantities: g_in, and rvar*g'_in (see Table 1, "Generalized 
%         Approximate Message Passing for Estimation with Random Linear 
%         Mixing").  When maxSumVal = false, sum-product GAMP updates are
%         used
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 10/09/12
% Change summary: 
%       - Created (10/01/12; JAZ)
%       - Added sum-product GAMP support (10/09/12; JAZ)
% Version 0.1
%

classdef ElasticNetEstimIn < EstimIn
    
    properties
        lambda1 = 1;        % ell-1 penalty value
        lambda2 = 1;        % ell-2 penalty value
        maxSumVal = true;   % Sum-product (false) or max-sum (true) GAMP?
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = ElasticNetEstimIn(lambda1, lambda2, maxsumval)
            obj = obj@EstimIn;
            if nargin >= 1 && ~isempty(lambda1)
                obj.lambda1 = lambda1;
            end
            if nargin >= 2 && ~isempty(lambda2)
                obj.lambda2 = lambda2;
            end
            if nargin >= 3 && ~isempty(maxsumval)
                obj.maxSumVal = maxsumval;
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.lambda1(obj, lambda1)
            if any(lambda1(:) < 0)
                error('lambda1 must be strictly non-negative')
            else
                obj.lambda1 = double(lambda1);
            end
        end
        
        function obj = set.lambda2(obj, lambda2)
            if any(lambda2(:) < 0)
                error('lambda2 must be strictly non-negative')
            else
                obj.lambda2 = double(lambda2);
            end
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            if isscalar(maxsumval) && islogical(maxsumval)
                obj.maxSumVal = maxsumval;
            else
                error('ElasticNetEstimIn: maxSumVal must be a logical scalar')
            end
        end
        
        
        % *****************************************************************
        %                        ESTIMINIT METHOD
        % *****************************************************************
        % Initial mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = zeros(size(obj.lambda1));
            var0  = 5e-4;
            valInit = -Inf;
        end
        
        
        % *****************************************************************
        %                          ESTIM METHOD
        % *****************************************************************
        % If obj.maxSumVal = true, this method will compute the required
        % max-sum GAMP update expressions for an elastic net prior, o.w.,
        % it will compute sum-product GAMP expressions
        function [xhat, xvar, val] = estim(obj, rhat, rvar)
            switch obj.maxSumVal
                case false      % Sum-product GAMP
                    % Begin by computing various constants on which the
                    % posterior mean and variance depend
                    mu = rhat ./ (1 + 2*rvar.*obj.lambda2);     % Gaussian prod mean
                    sig2 = rvar ./ (1 + 2*rvar.*obj.lambda2);   % Gaussian prod var
                    sig = sqrt(sig2);                           % Gaussian prod std dev
                    muL = mu + obj.lambda1.*sig2;               % Lower integral mean
                    muU = mu - obj.lambda1.*sig2;               % Upper integral mean
                    muL_over_sig = muL ./ sig;
                    muU_over_sig = muU ./ sig;
                    cdfL = normcdf(-muL_over_sig);              % Lower cdf
                    cdfU = normcdf(muU_over_sig);               % Upper cdf
                    cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
                    SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*rvar) ) .* ...
                        cdfRatio;
                    NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants
                    
                    % For the "trouble" constants (those with muL's and muU's
                    % that are too large to give accurate numerical answers),
                    % we will effectively peg the special constant to be Inf or
                    % 0 based on whether muL dominates muU or vice-versa
                    SpecialConstant(NaN_Idx & (-muL > muU)) = Inf;
                    SpecialConstant(NaN_Idx & (-muL < muU)) = 0;
                                        
                    % Compute the ratio normpdf(a)/normcdf(a) for
                    % appropriate upper- and lower-integral constants, a
                    RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
                    RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));
                    
                    % Now compute the first posterior moment...
                    xhat = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                        (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                        (muU + sig.*RatioU);
                    
                    % ...and second central posterior moment
                    varL = sig2 .* (1 - RatioL.*(RatioL - muL_over_sig));
                    varU = sig2 .* (1 - RatioU.*(RatioU + muU_over_sig));
                    meanL = muL - sig.*RatioL;
                    meanU = muU + sig.*RatioU;
                    SecondMoment = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                        (varL + meanL.^2) + (1 ./ (1 + SpecialConstant)) .* ...
                        (varU + meanU.^2);
                    xvar = SecondMoment - xhat.^2;
                    
                    % Lastly, compute negative KL divergence:
                    % \int_x p(x|r) log(p(x)/p(x|r)) dx
                    NormConL = obj.lambda1/2 .* ...             % Mass of lower integral
                        exp( (muL.^2 - mu.^2) ./ (2*sig2) ) .* cdfL;
                    NormConU = obj.lambda1/2 .* ...             % Mass of upper integral
                        exp( (muU.^2 - mu.^2) ./ (2*sig2) ) .* cdfU;
                    NormCon = NormConL + NormConU;      % Posterior normaliz. constant recip.
                    if obj.lambda2 > 0
                        val = (1/2)*log(2*pi*rvar) + ...
                            log( normpdf(0, rhat, sqrt(rvar + ...
                            (1./(2*obj.lambda2)))) .* (NormConL + ...
                            NormConU) ) + (1./(2*rvar)).*(xvar + xhat.^2) - ...
                            rhat.*xhat./rvar + (rhat.^2)./(2*rvar);
                    else
                        % Requires special treatment when lambda2 = 0
                        val = (1/2)*log(2*pi*rvar) + ...
                            log( NormConL + NormConU ) + ...
                            (1./(2*rvar)).*(xvar + xhat.^2) - ...
                            rhat.*xhat./rvar + (rhat.^2)./(2*rvar);
                    end
                    
                case true
                    % Return soft-thresholding max-sum GAMP updates
                    C1 = (1 + 2*obj.lambda2 .* rvar);
                    thresh = (obj.lambda1 .* rvar) ./ C1;
                    
                    % Un-soft-thresholded xhat
                    xhat = rhat ./ C1;
                    
                    % Soft-thresholded xhat
                    xhat = sign(xhat) .* max(0, abs(xhat) - thresh);
                    
                    % xvar = rvar*g'_in
                    xvar = (rvar ./ C1) .* (xhat ~= 0);
                    
                    % Lastly, compute log-prior evaluated at xhat
                    val = -(obj.lambda1 .* abs(xhat)) - ...
                        (obj.lambda2 .* xhat.^2);
            end
        end
        
        
        % *****************************************************************
        %                          GENRAND METHOD
        % *****************************************************************
        % Generate random samples
        function x = genRand(obj, outSize)
            % We will generate x_n according to the distribution
            % p(x_n) \propto N(x_n; 0, (2*lambda2)^(-1)) Lap(x_n; lambda1)
            % using the method of inverse transform sampling, wherein we
            % generate p_n ~ U(0,1) and evaluate x_n = Finv(p_n), where
            % Finv is the inverse CDF of the distribution p(x_n), implying
            % that x_n ~ p(x).  Numerical precision problems occur when 
            % lambda2 << 1
            
            l1 = obj.lambda1;   % Shorthand
            l2 = obj.lambda2;   % Shorthand
            if ~isscalar(l1) || ~isscalar(l2)
                error('Please use scalar choices of lambda1 and lambda2')
            end
            
            % Compute normalizing constant 
            a = l1*exp(l1^2 / 4 /l2)*normcdf(-l1/sqrt(2*l2));
            
            % Generate vector of uniform RVs
            if isscalar(outSize)
                p = rand(outSize, 1);
            else
                p = rand(outSize);
            end
            
            % Apply inverse CDF to p
            x = NaN(size(p));
            x(p <= 1/2) = norminv(2* normcdf(-l1/sqrt(2*l2)) * p(p <= 1/2), ...
                l1/2/l2, 1/sqrt(2*l2));
            x(p > 1/2) = norminv(2*normcdf(-l1/sqrt(2*l2))*(p(p > 1/2) - 1/2) + ...
                normcdf(l1/sqrt(2*l2)), -l1/2/l2, 1/sqrt(2*l2));
            
            if any(isnan(x(:)))
                error('Something went wrong generating ElasticNet samples')
            end
        end
        
        
        % *****************************************************************
        %                           PLIKEY METHOD
        % *****************************************************************
        % Computes p(y) for y = x + v, with x ~ p(x), v ~ N(0,yvar)
        function py = plikey(obj, y, yvar)
            % Begin by computing various constants on which the
            % posterior mean and variance depend
            mu = y ./ (1 + 2*yvar.*obj.lambda2);     % Gaussian prod mean
            sig2 = yvar ./ (1 + 2*yvar.*obj.lambda2);   % Gaussian prod var
            sig = sqrt(sig2);                           % Gaussian prod std dev
            muL = mu + obj.lambda1.*sig2;               % Lower integral mean
            muU = mu - obj.lambda1.*sig2;               % Upper integral mean
            muL_over_sig = muL ./ sig;
            muU_over_sig = muU ./ sig;
            pdfL = normpdf(muL_over_sig);               % Lower pdf
            cdfL = normcdf(-muL_over_sig);              % Lower cdf
            pdfU = normpdf(muU_over_sig);               % Upper pdf
            cdfU = normcdf(muU_over_sig);               % Upper cdf
            NormConL = obj.lambda1/2 .* ...             % Mass of lower integral
                exp( (muL.^2 - mu.^2) ./ (2*sig2) ) .* cdfL;
            NormConU = obj.lambda1/2 .* ...             % Mass of upper integral
                exp( (muU.^2 - mu.^2) ./ (2*sig2) ) .* cdfU;
            NormCon = NormConL + NormConU;      % Posterior normaliz. constant recip.
            
            % Also compute normalization constant of prior, p(x)
            sig2 = 1./(2*obj.lambda2);                  % Gaussian var
            sig = sqrt(sig2);                           % Gaussian prod std dev
            muL = obj.lambda1.*sig2;                    % Lower integral mean
            muU = obj.lambda1.*sig2;                    % Upper integral mean
            muL_over_sig = muL ./ sig;
            muU_over_sig = muU ./ sig;
            cdfL = normcdf(-muL_over_sig);              % Lower cdf
            cdfU = normcdf(muU_over_sig);               % Upper cdf
            PriorConL = obj.lambda1/2 .* ...             % Mass of lower integral
                exp( (muL.^2 - mu.^2) ./ (2*sig2) ) .* cdfL;
            PriorConU = obj.lambda1/2 .* ...             % Mass of upper integral
                exp( (muU.^2 - mu.^2) ./ (2*sig2) ) .* cdfU;
            
            py = normpdf(0, y, sqrt(yvar + 1./(2*obj.lambda2))) .* NormCon ./ ...
                (PriorConL + PriorConU);
        end
    end
end