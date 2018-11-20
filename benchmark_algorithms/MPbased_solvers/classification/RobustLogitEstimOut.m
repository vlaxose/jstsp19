% CLASS: RobustLogitEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The RobustLogitEstimOut class defines a scalar observation channel, 
%   p(y|z), that constitutes an outlier-robust logit binary classification
%   model in which it is assumed that, with probability p_flip, a training
%   class label, y \in {0,1}, has had its value flipped, i.e.,
%      p(y | z) = (1 - p_flip) * sigma(y; z) + p_flip * sigma(-y; z),
%                   = p_flip + (1 - 2*p_flip) * sigma(y; z),
%   where
%               sigma(y; z) = (1 + exp(-scale*y*z))^(-1)
%   is the logistic likelihood, and scale is a positive scalar parameter
%   that controls the shape of the logistic sigmoid function.  In other 
%   words, this class takes the standard logit model (see LogitEstimOut) 
%   and extends it to the cases in which some of the training samples are
%   either outliers, or mislabeled.
%
%   In order to construct an object of the RobustLogitEstimOut class, one
%   must first construct an object of the LogitEstimOut class, which will
%   contain the class labels and logit scale parameter (see above).
%   This LogitEstimOut object is then used as an argument in the
%   RobustLogitEstimOut constructor, along with the mislabeling
%   probability, p_flip.
%
% PROPERTIES (State variables)
%   LogitObj    An object of the LogitEstimOut class, containing class
%               labels and the logit scale factor
%   p_flip      The probability that a training sample has been mislabeled
%               (flipped)
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: false]
%
% METHODS (Subroutines/functions)
%   RobustLogitEstimOut(LogitObj, p_flip)
%       - Default constructor.
%   LogitEstimOut(LogitObj, p_flip, maxSumVal)
%       - Optional constructor.  Sets LogitObj, p_flip, and maxSumVal.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the robust logit model and maxSumVal = false (see 
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 07/03/13
% Change summary: 
%       - Created (07/03/13; JAZ)
% Version 0.2
%

classdef RobustLogitEstimOut < EstimOut
    
    properties
        LogitObj;           % LogitEstimOut object
        p_flip;             % Mislabeling probability
        maxSumVal = false;  % Sum-product (false) or max-sum (true) GAMP?
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = RobustLogitEstimOut(LogitObj, p_flip, maxsumval)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.LogitObj = LogitObj;      % Set LogitObj
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
        function obj = set.LogitObj(obj, LogitObj)
            assert(isa(LogitObj, 'LogitEstimOut'), ...
                'LogitObj must be a valid LogitEstimOut object');
            obj.LogitObj = LogitObj;
        end
        
        function obj = set.p_flip(obj, p_flip)
            assert(isscalar(p_flip) && (p_flip >= 0) && (p_flip <= 1), ...
                'p_flip must be a scalar in the interval [0,1]');
            obj.p_flip = p_flip;
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            assert(isscalar(maxsumval) && islogical(maxsumval), ...
                'RobustLogitEstimOut: maxSumVal must be a logical scalar');
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
                    Y = obj.LogitObj.y;
                    Y(Y == 0) = -1;         % {0,1} -> {-1,1}
                    a = obj.LogitObj.scale;
                    
                    % Compute an upper bound of the normalizing constant,
                    % C*_y, based on a variational lower bound of p*(y|z)
                    % (the logistic activation function), and Jensen's
                    % inequality ( log E[p*(y|z)] <= E[log(p*(y|z))] )
                    Xi = sqrt(Pvar + abs(Phat).^2);
                    Cstar = exp(a*Y.*Phat - (a/2)*(Phat + Xi) - ...
                        log(1 + exp(-a*Xi)));
                    C = pflip ./ (pflip + (1 - 2*pflip)*Cstar);
                    assert(~any(isnan(C(:))) && ~any(isnan(Cstar(:))), ...
                        'NaNs encountered in computing normalizing constant');
                    
                    % Get the standard Logit channel posterior mean and
                    % variance from the LogitEstimOut object
                    [Zhat_std, Zvar_std] = estim(obj.LogitObj, Phat, Pvar);
                    
                    % Use the above quantities to compute the posterior
                    % mean and variance under the robust Logit model
                    Zhat = (C .* Phat) +  (1 - C) .* Zhat_std;	% E[z|y]
                    SecMom = C .* (Pvar + abs(Phat).^2) + ...
                        (1 - C) .* (Zvar_std + abs(Zhat_std).^2);  	% E[z^2|y]
                    Zvar = SecMom - abs(Zhat).^2;                   % var{z|y}
                    
                case true
                    error('MAP support unavailable');

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
            PMonesY = obj.LogitObj.y;
            PMonesY(PMonesY == 0) = -1;     % {0,1} -> {-1,1}
            switch obj.maxSumVal
                case false
                    pflip = obj.p_flip;
                    Y = obj.LogitObj.y;
                    Y(Y == 0) = -1;         % {0,1} -> {-1,1}
                    a = obj.LogitObj.scale;
                    
                    % Compute an upper bound of the normalizing constant,
                    % C*_y, based on a variational lower bound of p*(y|z)
                    % (the logistic activation function), and Jensen's
                    % inequality ( log E[p*(y|z)] <= E[log(p*(y|z))] )
                    Xi = sqrt(Zvar + abs(Zhat).^2);
                    Cstar = exp(a*Y.*Zhat - (a/2)*(Zhat + Xi) - ...
                        log(1 + exp(-a*Xi)));   % E_z[p*(y|z)]
                    
                    % Evaluate log p(y|z)
                    ll = log(pflip + (1 - 2*pflip) .* Cstar);
                case true
                    ll = log(obj.p_flip + (1 - 2*obj.p_flip) * ...
                        (1 + exp(-obj.LogitObj.scale * PMonesY .* Zhat)).^(-1));
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
            S = size(obj.LogitObj.y, 2);
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
                Jacobian = sparse(1:M, 1:M, Jvec);
            end
        end
    end
end