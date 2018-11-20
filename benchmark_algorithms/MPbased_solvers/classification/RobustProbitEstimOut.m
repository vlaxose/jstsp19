% CLASS: RobustProbitEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The RobustProbitEstimOut class defines a scalar observation channel, 
%   p(y|z), that constitutes an outlier-robust probit binary classification
%   model in which it is assumed that, with probability p_flip, a training
%   class label, y \in {0,1}, has had its value flipped, i.e.,
%      p(y = 1 | z) = (1 - p_flip) * Phi((z - Mean)/sqrt(Var)) + ...
%                     p_flip * Phi(-(z - Mean)/sqrt(Var)),
%                   = p_flip + (1 - 2*p_flip) * Phi((z - Mean)/sqrt(Var)),
%   where Phi((x-b)/sqrt(c)) is the cumulative density function (CDF) of a
%   Gaussian random variable with mean b, variance c, and argument x.
%   In other words, this class takes the standard probit model (see
%   ProbitEstimOut) and extends it to the cases in which some of the
%   training samples have been mislabeled.
%
%   In order to construct an object of the RobustProbitEstimOut class, one
%   must first construct an object of the ProbitEstimOut class, which will
%   contain the class labels and probit channel mean(s) and variance(s).
%   This ProbitEstimOut object is then used as an argument in the
%   RobustProbitEstimOut constructor, along with the mislabeling
%   probability, p_flip.
%
% PROPERTIES (State variables)
%   ProbitObj   An object of the ProbitEstimOut class, containing class
%               labels and the probit channel mean(s) and variance(s)
%   p_flip      The probability that a training sample has been mislabeled
%               (flipped)
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: false]
%
% METHODS (Subroutines/functions)
%   RobustProbitEstimOut(ProbitObj, p_flip)
%       - Default constructor.
%   ProbitEstimOut(ProbitObj, p_flip, maxSumVal)
%       - Optional constructor.  Sets ProbitObj, p_flip, and maxSumVal.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the robust probit model and maxSumVal = false (see 
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 02/05/13
% Change summary: 
%       - Created (02/05/13; JAZ)
% Version 0.2
%

classdef RobustProbitEstimOut < EstimOut
    
    properties
        ProbitObj;          % ProbitEstimOut object
        p_flip;             % Mislabeling probability
        maxSumVal = false;  % Sum-product (false) or max-sum (true) GAMP?
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = RobustProbitEstimOut(ProbitObj, p_flip, maxsumval)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.ProbitObj = ProbitObj;      % Set ProbitObj
                assert(nargin >= 2 && ~isempty(p_flip), ...
                    'p_flip is a required argument');
                obj.p_flip = p_flip;
                if nargin >= 3 && ~isempty(maxsumval)
                    % maxSumVal property is an argument
                    obj.maxSumVal = maxsumval;
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.ProbitObj(obj, ProbitObj)
            assert(isa(ProbitObj, 'ProbitEstimOut'), ...
                'ProbitObj must be a valid ProbitEstimOut object');
            obj.ProbitObj = ProbitObj;
        end
        
        function obj = set.p_flip(obj, p_flip)
            assert(isscalar(p_flip) && (p_flip >= 0) && (p_flip <= 1), ...
                'p_flip must be a scalar in the interval [0,1]');
            obj.p_flip = p_flip;
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            assert(isscalar(maxsumval) && islogical(maxsumval), ...
                'RobustProbitEstimOut: maxSumVal must be a logical scalar');
            obj.maxSumVal = maxsumval;
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
            switch obj.maxSumVal
                case false
                    % Extract some needed quantities
                    pflip = obj.p_flip;
                    Y = obj.ProbitObj.Y;
                    Y(Y == 0) = -1;         % {0,1} -> {-1,1}
                    mean = obj.ProbitObj.Mean;
                    var = obj.ProbitObj.Var;
                    
                    % Compute normalizing constant, C
                    c_bar = (Phat - mean) ./ (sqrt(var + Pvar));
                    sCDF = (1 - 2*pflip) * normcdf(Y .* c_bar);
                    C = pflip + sCDF;
                    PartC = ((pflip ./ sCDF) + 1).^(-1);    % sCDF / C
                    assert(~any(isnan(C(:))) && ~any(isnan(PartC(:))), ...
                        'NaNs encountered in computing normalizing constant');
                    
                    % Get the standard probit channel posterior mean and
                    % variance from the ProbitEstimOut object
                    [Zhat_std, Zvar_std] = estim(obj.ProbitObj, Phat, Pvar);
                    
                    % Use the above quantities to compute the posterior
                    % mean and variance under the robust probit model
                    Zhat = pflip*(Phat ./ C) +  PartC .* Zhat_std;	% E[z|y]
                    SecMom = pflip*(Pvar + abs(Phat).^2) ./ C + ...
                        PartC .* (Zvar_std + abs(Zhat_std).^2);  	% E[z^2|y]
                    Zvar = SecMom - abs(Zhat).^2;                   % var{z|y}
                    
                case true
                    error('MAP support unavailable');
%                     % Return max-sum expressions to GAMP in Zhat and Zvar
%                     PMonesY = 2*(obj.Y - 1/2);      % +/- 1 for Y(m,t)'s
%                     
%                     % Determine the expansion point about which to perform
%                     % the Taylor series approximation
%                     EP = (sign(PMonesY) == sign(Phat)) .* Phat;
%                     
% %                     % First compute a second-order Taylor series
% %                     % approximation of log p(y|z) - 1/2/Pvar (z - Phat)^2,
% %                     % about the point EP, and set as Zhat the maximizer 
% %                     % of this approximation
% %                     C = PMonesY .* (EP - obj.Mean) ./ sqrt(obj.Var);
% %                     % Now compute the ratio normpdf(C)/normcdf(C)
% %                     ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
% %                     % Compute 1st deriv of maximization functional
% %                     Deriv1 = (PMonesY ./ sqrt(obj.Var)) .* ratio - ...
% %                         (1./Pvar) .* (EP - Phat);
% %                     % Compute 2nd deriv of maximization functional
% %                     Deriv2 = -(1./obj.Var) .* ratio .* (C + ratio) - ...
% %                         (1 ./ Pvar);
% %                     % Set maximizer of Taylor approximation as Zhat
% %                     Zhat = Phat - Deriv1 ./ Deriv2;
% 
%                     % Manually locate the value of z that sets the cost
%                     % function derivative to zero using fsolve
%                     opts = optimset('Jacobian', 'on', 'MaxIter', 25, ...
%                         'Display', 'off');
%                     F = @(z) zero_deriv(obj, z, Phat, Pvar);
%                     Zhat = fsolve(F, EP, opts);
%                     
%                     % Now compute second derivative of log p(y|z) evaluated
%                     % at Zhat (Note: Not an approximation)
%                     % Start by computing frequently appearing constant
%                     C = PMonesY .* (Zhat - obj.Mean) ./ sqrt(obj.Var);
%                     % Now compute the ratio normpdf(C)/normcdf(C)
%                     ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
%                     % Compute 2nd deriv of log p(y|z)
%                     Deriv = -(1./obj.Var) .* ratio .* (C + ratio);
% %                     Deriv = max(1e-6, Deriv);
%                     
%                     % Output in Zvar a function of the 2nd derivative that,
%                     % once manipulated by gampEst, yields the desired
%                     % max-sum expression for -g'_{out}
%                     Zvar = Pvar ./ (1 - Pvar.*Deriv);
            end
        end
        
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % This function will compute *an approximation* to the expected
        % log-likelihood, E_z[log p(y|z)] when performing sum-product GAMP
        % (obj.maxSumVal = false).  The approximation is based on Jensen's 
        % inequality, i.e., computing log E_z[p(y|z)] instead.  If
        % performing max-sum GAMP (obj.maxSumVal = true), logLike returns
        % log p(y|z) evaluated at z = Zhat
        function ll = logLike(obj, Zhat, Zvar)
            PMonesY = obj.ProbitObj.Y;
            PMonesY(PMonesY == 0) = -1;     % {0,1} -> {-1,1}
            switch obj.maxSumVal
                case false
                    % Start by computing the critical constant, C, on which
                    % the remainder of the computations depend.  Modulate 
                    % this constant by -1 for cases where Y(m,t) = 0.
                    C = PMonesY .* ((Zhat - obj.ProbitObj.Mean) ./ ...
                        sqrt(Zvar + obj.ProbitObj.Var));
                    CDF = normcdf(C);
                    ll = log(obj.p_flip + (1 - 2*obj.p_flip)*CDF);
%                     ll = log(CDF);
%                     ll(ll == -inf) = -1e4;
                case true
                    ll = log(obj.p_flip + (1 - 2*obj.p_flip) * ...
                        normcdf(PMonesY .* (Zhat - obj.Mean)/sqrt(obj.Var)));
            end
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
                   
            error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');         
            
        end
        
        
        % *****************************************************************
        %                       NUMCOLUMNS METHOD
        % *****************************************************************
        function S = numColumns(obj)
            % Return number of columns of Y
            S = size(obj.ProbitObj.Y, 2);
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
            PMonesY = 2*(obj.Y - 1/2);
            C = PMonesY .* (z - obj.Mean) ./ sqrt(obj.Var);
            % Now compute the ratio normpdf(C)/normcdf(C)
            ratio = (2/sqrt(2*pi)) * (erfcx(-C / sqrt(2)).^(-1));
            % Value of derivative
            Fval = PMonesY.*ratio./sqrt(obj.Var) - (z - phat)./pvar;
            
            % Optionally compute Jacobian of F at z
            if nargout >= 2
                M = numel(phat);
                Jvec = -(1./obj.Var) .* ratio .* (C + ratio) - (1 ./ pvar);
%                 Jacobian = diag(Jvec);      % Okay for small problems
                Jacobian = sparse(1:M, 1:M, Jvec);
            end
        end
    end
end