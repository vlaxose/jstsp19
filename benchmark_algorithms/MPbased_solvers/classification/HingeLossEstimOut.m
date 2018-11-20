% CLASS: HingeLossEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The HingeLossEstimOut class defines a penalized output channel governed
%   by the "hinge loss" function:
%                          HL(z,y) = max(0, 1 - y*z),
%   where y takes the value -1 or +1.  The hinge loss function can be
%   intepreted as a (negative) log-likelihood, i.e.,
%                   p(y|z) \propto 1/exp(max(0, 1 - y*z)),
%   although note that this distribution is in fact improper (cannot
%   integrate to unity).
%
% PROPERTIES (State variables)
%   Y           An M-by-T array of binary ({-1,+1}) class labels for the
%               training data, where M is the number of training data
%               points, and T is the number of classifiers being learned
%               (typically T = 1)
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: true]
%
% METHODS (Subroutines/functions)
%   HingeLossEstimOut(Y)
%       - Default constructor.  Assigns maxSumVal to default value.
%   HingeLossEstimOut(Y, maxSumVal)
%       - Optional constructor.  Sets Y and maxSumVal.
%   HingeLossEstimOut(Y, maxSumVal, scale)
%       - Optional constructor.  Sets Y, maxSumVal, and scale parameter.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the hinge loss model and maxSumVal = false (see 
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 11/01/12
% Change summary: 
%       - Created (10/23/12; JAZ)
% Version 0.2
%

classdef HingeLossEstimOut < EstimOut
    
    properties
        Y;              % M-by-T vector of binary class labels
        maxSumVal = false;   % Sum-product (false) or max-sum (true) GAMP?
        scale = 1;  % generalization of p(y|z) to p(y|scale*z)
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = HingeLossEstimOut(Y, maxsumval, scale)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.Y = Y;      % Set Y
                if nargin >= 2 && ~isempty(maxsumval)
                    % maxSumVal property is an argument
                    obj.maxSumVal = maxsumval;
                end
                if nargin >= 3 && ~isempty(scale)
                    obj.scale = scale;
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.Y(obj, Y)
            if all((Y(:) == 0) | (Y(:) == 1)) 
                % Convert automatically from {0,1} binary labeling to
                % {-1,1} labeling
                Y(Y == 0) = -1;
                obj.Y = Y;
            elseif islogical(Y)
                % Convert automatically from {0,1} logical labeling to
                % {-1,1} labeling
                Y = double(Y);
                Y(Y == 0) = -1;
                obj.Y = Y;
            elseif all((Y(:) == -1) | (Y(:) == 1))
                obj.Y = Y;
            else
                error('Entries of Y must either be -1 or +1')
            end
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            if isscalar(maxsumval) && islogical(maxsumval)
                obj.maxSumVal = maxsumval;
            else
                error('HingeLossEstimOut: maxSumVal must be a logical scalar')
            end
        end
        
        
        % *****************************************************************
        %                          ESTIM METHOD
        % *****************************************************************
        % This function will compute the posterior mean and variance of a
        % random vector Z whose prior distribution is N(Phat, Pvar), given
        % observations Y obtained through the separable channel model:
        % p(Y(m,t) = 1 | Z(m,t)) = Phi((Z(m,t) - Mean(m,t))/sqrt(Var(m,t)))
        % if obj.maxSumVal = false, otherwise it will return Zhat = argmax
        % log p(y|z) - 1/2/Pvar (z - Phat)^2 and the second derivative of
        % log p(y|z) evaluated at Zhat in Zvar, if obj.maxSumVal = true
        function [Zhat, Zvar] = estim(obj, Phat, Pvar)

            % Scale inputs
            Phat = Phat*obj.scale; 
            Pvar = Pvar*(obj.scale^2);

            switch obj.maxSumVal
                case false
                    % Return sum-product expressions to GAMP in Zhat and
                    % Zvar
                    
                    % Start by computing various commonly used quantities
                    Pstd = sqrt(Pvar);      % Gaussian std dev
                    Mask = obj.Y == 1;      % Logical mask
                    NormCon = NaN(size(obj.Y));     % Placeholder for normalizing constant
                    ExpArg = NaN(size(obj.Y));
                    Alpha = NaN(size(obj.Y));
                    Beta = NaN(size(obj.Y));
                    Alpha(Mask) = (1 - Phat(Mask)) ./ Pstd(Mask);
                    Beta(Mask) = (1 - Phat(Mask) - Pvar(Mask)) ./ Pstd(Mask);
                    Alpha(~Mask) = (-1 - Phat(~Mask) + Pvar(~Mask)) ./ Pstd(~Mask);
                    Beta(~Mask) = -(Phat(~Mask) + 1) ./ Pstd(~Mask);
                    CDFalpha = normcdf(-Alpha);     % Recurring qty, Phi(-alpha)
                    CDFbeta = normcdf(Beta);        % Recurring qty, Phi(beta)
                    ExpArg(Mask) = exp(Phat(Mask) + Pstd(Mask)/2 - 1);
                    ExpArg(~Mask) = exp(-Phat(~Mask) + Pstd(~Mask)/2 - 1);
                    ExpArg = min(ExpArg, realmax);
                    
                    % Now compute the normalizing constant for p(z|y=1) and
                    % p(z|y=-1)
                    NormCon(Mask) = ExpArg(Mask) .* CDFbeta(Mask) + ...
                        CDFalpha(Mask);
                    NormCon(~Mask) = CDFbeta(~Mask) + ExpArg(~Mask) .* ...
                        CDFalpha(~Mask);
                    
                    % Compute the ratio normpdf(#)/normcdf(#) for select
                    % choices of #
                    RatioAlpha = NaN(size(obj.Y)); RatioBeta = NaN(size(obj.Y));
                    RatioAlpha(Mask) = 2/sqrt(2*pi) ./ erfcx(Alpha(Mask) / sqrt(2));
                    RatioBeta(Mask) = 2/sqrt(2*pi) ./ erfcx(-Beta(Mask) / sqrt(2));
                    RatioAlpha(~Mask) = 2/sqrt(2*pi) ./ erfcx(Alpha(~Mask) / sqrt(2));
                    RatioBeta(~Mask) = 2/sqrt(2*pi) ./ erfcx(-Beta(~Mask) / sqrt(2));
                    
                    % Compute means and variances of certain truncated
                    % normal random variables
                    MeanAlpha = NaN(size(obj.Y)); MeanBeta = NaN(size(obj.Y));
                    MeanAlpha(Mask) = Phat(Mask) + Pstd(Mask) .* RatioAlpha(Mask);
                    MeanBeta(Mask) = Phat(Mask) + Pvar(Mask) - ...
                        Pstd(Mask) .* RatioBeta(Mask);
                    MeanAlpha(~Mask) = Phat(~Mask) - Pvar(~Mask) + ...
                        Pstd(~Mask) .* RatioAlpha(~Mask);
                    MeanBeta(~Mask) = Phat(~Mask) - Pstd(~Mask) .* RatioBeta(~Mask);
                    VarAlpha = Pvar .* (1 - RatioAlpha .* (RatioAlpha - Alpha));
                    VarBeta = Pvar .* (1 - RatioBeta .* (RatioBeta + Beta));
                
                    % Now compute the first posterior moment...
                    Zhat = NaN(size(obj.Y));
                    Zhat(Mask) = (1./NormCon(Mask)) .* (ExpArg(Mask) .* ...
                        CDFbeta(Mask) .* MeanBeta(Mask) + ...
                        CDFalpha(Mask) .* MeanAlpha(Mask));
                    Zhat(~Mask) = (1./NormCon(~Mask)) .* (CDFbeta(~Mask) .* ...
                        MeanBeta(~Mask) + ExpArg(~Mask) .* CDFalpha(~Mask) .* ...
                        MeanAlpha(~Mask));
                    
                    % ...and second central posterior moment
                    SecondMoment = NaN(size(obj.Y));
                    SecondMoment(Mask) = (1./NormCon(Mask)) .* ...
                        ( ExpArg(Mask) .* CDFbeta(Mask) .* (VarBeta(Mask) + ...
                        MeanBeta(Mask).^2) + CDFalpha(Mask) .* ...
                        (VarAlpha(Mask) + MeanAlpha(Mask).^2) );
                    SecondMoment(~Mask) = (1./NormCon(~Mask)) .* ( CDFbeta(~Mask) .* ...
                        (VarBeta(~Mask) + MeanBeta(~Mask).^2) + ExpArg(~Mask) .* ...
                        CDFalpha(~Mask) .* (VarAlpha(~Mask) + MeanAlpha(~Mask).^2) );
                    Zvar = SecondMoment - Zhat.^2;
                    
                    % In cases where output variance is almost equal to
                    % input variance, we must slightly elevate the output
                    % variance in order to maintain GAMP numerical
                    % stability
                    inds = find(1 - Zvar./Pvar < 1e-9);
                    Zvar(inds) = (1 - 1e-9).*Pvar(inds);
                   
                case true
                    % Return max-sum expressions to GAMP in Zhat and Zvar
                    
                    % Manually locate the value of z that sets the cost
                    % function derivative to zero using fsolve
                    opts = optimset('Jacobian', 'on', 'MaxIter', 50, ...
                        'Display', 'off');
                    F = @(z) zero_deriv(obj, z, Phat, Pvar);
                    Zhat = fsolve(F, Phat, opts);                    
                    
                    % Output in Zvar a function of the 2nd derivative that,
                    % once manipulated by gampEst, yields the desired
                    % max-sum expression for -g'_{out}.  Since 2nd
                    % derivative equals -1/Pvar for all Zhat, and
                    % manipulation is Pvar ./ (1 - Pvar * 2ndDeriv), just
                    % return Pvar/2
                    Zvar = Pvar / 2;
            end

            % De-scale outputs
            Zhat = Zhat*(1/obj.scale);
            Zvar = Zvar*(1/obj.scale^2);
        end
        
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % This function will compute  the expected log-likelihood, 
        % E_z|y[log p(y|z) | y] when performing sum-product GAMP
        % (obj.maxSumVal = false).  If performing max-sum GAMP 
        % (obj.maxSumVal = true), logLike returns log p(y|z) evaluated at 
        % z = Zhat
        function ll = logLike(obj, Zhat, Zvar)
            % Scale inputs
            Zhat = Zhat*obj.scale; 
            Zvar = Zvar*(obj.scale^2);

            switch obj.maxSumVal
                case false
                    % Compute certain constants
                    Mask = (obj.Y == 1);    % Logical mask
                    ll = NaN(size(obj.Y));
                    C = NaN(size(obj.Y));
                    C(Mask) = (1 - Zhat(Mask)) ./ sqrt(Zvar(Mask));
                    C(~Mask) = (Zhat(~Mask) + 1) ./ sqrt(Zvar(~Mask));
                    
                    % Compute normcdf(C) and normpdf(C)/normcdf(C)
                    CDF = normcdf(C);
                    Ratio = NaN(size(obj.Y));
                    Ratio(Mask) = 2/sqrt(2*pi) ./ erfcx(-C(Mask) / sqrt(2));
                    Ratio(~Mask) = 2/sqrt(2*pi) ./ erfcx(C(~Mask) / sqrt(2));
                    
                    % Compute mean of truncated normal RVs
                    TruncMean = NaN(size(obj.Y));
                    TruncMean(Mask) = Zhat(Mask) - sqrt(Zvar(Mask)) .* ...
                        Ratio(Mask);
                    TruncMean(~Mask) = Zhat(~Mask) - sqrt(Zvar(~Mask)) .* ...
                        Ratio(~Mask);
                    
                    % Now evaluate log-likelihood (up to a Zhat- and
                    % Zvar-independent additive constant)
                    ll = -CDF + obj.Y .* CDF .* TruncMean;
                case true
                    ll = -max(0, 1 - obj.Y .* Zhat);
            end
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
                   
            % Scale inputs
            Axhat = Axhat*obj.scale; 
            phat = phat*obj.scale; 
            pvar = pvar*(obj.scale^2);

            if ~(obj.maxSumVal)
                % Find the fixed-point of phat
                opt.phat0 = Axhat; % works better than phat
                opt.alg = 1; % approximate newton's method
                opt.maxIter = 20; 
                opt.tol = 1e-5; 
                opt.stepsize = 1; 
                opt.regularization = 1e-8;
                opt.debug = false;
                phatfix = estimInvert(obj,Axhat,pvar,opt); 
                
                % Start by computing various commonly used quantities
                Pstd = sqrt(pvar);      % Gaussian std dev
                Mask = obj.Y == 1;      % Logical mask
                NormCon = NaN(size(obj.Y));     % Placeholder for normalizing constant
                ExpArg = NaN(size(obj.Y));
                Alpha = NaN(size(obj.Y));
                Beta = NaN(size(obj.Y));
                Alpha(Mask) = (1 - phatfix(Mask)) ./ Pstd(Mask);
                Beta(Mask) = (1 - phatfix(Mask) - pvar(Mask)) ./ Pstd(Mask);
                Alpha(~Mask) = (-1 - phatfix(~Mask) + pvar(~Mask)) ./ Pstd(~Mask);
                Beta(~Mask) = -(phatfix(~Mask) + 1) ./ Pstd(~Mask);
                CDFalpha = normcdf(-Alpha);     % Recurring qty, Phi(-alpha)
                CDFbeta = normcdf(Beta);        % Recurring qty, Phi(beta)
                ExpArg(Mask) = exp(phatfix(Mask) + Pstd(Mask)/2 - 1);
                ExpArg(~Mask) = exp(-phatfix(~Mask) + Pstd(~Mask)/2 - 1);
                ExpArg = min(ExpArg, realmax);

                % Now compute the normalizing constant for p(z|y=1) and
                % p(z|y=-1)
                NormCon(Mask) = ExpArg(Mask) .* CDFbeta(Mask) + ...
                    CDFalpha(Mask);
                NormCon(~Mask) = CDFbeta(~Mask) + ExpArg(~Mask) .* ...
                    CDFalpha(~Mask);
                
                % Combine log scale factor with extra quantity to form output cost
                ll = log(NormCon) + 0.5*(Axhat - phatfix).^2./pvar;
                         
            else 
                ll = -max(0, 1 - obj.Y .* Zhat);
            end
            
        end
        
        
        % *****************************************************************
        %                       NUMCOLUMNS METHOD
        % *****************************************************************
        function S = numColumns(obj)
            % Return number of columns of Y
            S = size(obj.Y, 2);
        end
    end
    
    methods (Access = private)
        % *****************************************************************
        %                         FSOLVE METHOD
        % *****************************************************************
        % This method is used by MATLAB's fsolve function in the max-sum
        % case to locate the value of z that sets the prox-operator
        % derivative to zero
        function [Fval, Jacobian] = zero_deriv(obj, z, phat, pvar)
            % Compute value of derivative at z
            Fval = (-1 ./ pvar) .* (z - phat);
            Mask = (obj.Y == 1) & (z < 1);
            Fval(Mask) = Fval(Mask) + 1;
            Mask = (obj.Y == -1) & (z > -1);
            Fval(Mask) = Fval(Mask) - 1;
            
            % Optionally compute Jacobian of F at z
            if nargout >= 2
                M = numel(phat);
                Jvec = -1 ./ pvar;
%                 Jacobian = diag(Jvec);      % Okay for small problems
                Jacobian = sparse(1:M, 1:M, Jvec);
            end
        end
    end
end
