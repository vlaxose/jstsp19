% CLASS: TDistEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The TDistEstimOut class defines a scalar observation channel, p(y|z),
%   that constitutes a binary classification model, i.e., y is an element 
%   of the set {0,1}, z is a real number, and
%             	 p(y = 1 | z) = F_2(z/sigma),
%   where F_2(.) denotes the cumulative distribution function (CDF) of a
%   Student's t distribution with 2 degrees of freedom, and sigma is a
%   positive scalar that defines the sharpness of the sigmoid function
%   defined by the Student's t CDF.  This is the so-called "robit
%   regression" model described in "Robit Regression: A Simple Robust 
%   Alternative to Logistic and Probit Regression," Chuanhai Liu, 2006.
%
% PROPERTIES (State variables)
%   Y           An M-by-T array of binary ({0,1}) class labels for the
%               training data, where M is the number of training data
%               points, and T is the number of classifiers being learned
%               (typically T = 1)
%   sigma       A positive scalar quantity that governs the sharpness of
%               the Student's t CDF sigmoid shape (smaller sigma equates to
%               sharper/steeper sigmoid) [Default: 1e-1]
%
% METHODS (Subroutines/functions)
%   TDistEstimOut(Y)
%       - Default constructor.  Assigns sigma to default value.
%   TDistEstimOut(Y, sigma)
%       - Optional constructor.  Sets both Y and sigma.
%   estim(obj, zhat, zvar)
%       - Returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z), under the robit
%         regression model, and with p(z) = N(zhat, zvar).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 09/04/12
% Change summary: 
%       - Created (09/04/12; JAZ)
% Version 0.1
%

classdef TDistEstimOut < EstimOut
    
    properties
        Y;              % M-by-T vector of binary class labels
        sigma = 1e-1;   % Scalar sigmoid shape parameter [Default: 1e-1]
        maxSumVal = true;   % Default to max-sum (MAP) GAMP
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = TDistEstimOut(Y, sigma)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.Y = Y;      % Set Y
                if nargin >= 2 && ~isempty(sigma)
                    % Mean property is an argument
                    obj.sigma = sigma;
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.Y(obj, Y)
            if ~all((Y(:) == 0) | (Y(:) == 1))
                error('Elements of Y must be binary (0,1)')
            else
                obj.Y = Y;
            end
        end
        
        function obj = set.sigma(obj, sigma)
            if isscalar(sigma) && (sigma > 0)
                obj.sigma = sigma;
            else
                error('sigma must be a positive scalar')
            end
        end
        
        function obj = set.maxSumVal(obj, maxSumVal)
            if isscalar(maxSumVal)
                obj.maxSumVal = logical(maxSumVal);
            end
            if ~obj.maxSumVal
                error('TDistEstimOut does not support sum-product (MMSE) GAMP')
            end
        end
        
        
        % *****************************************************************
        %                          ESTIM METHOD
        % *****************************************************************
        % This function will return (approximately)
        %       Zhat = argmax log p(y|z) - 1/2/Pvar (z - Phat)^2 
        % and the second derivative of log p(y|z) evaluated at Zhat in Zvar
        function [Zhat, Zvar] = estim(obj, Phat, Pvar)
            % Return max-sum expressions to GAMP in Zhat and Zvar
            PMonesY = 2*(obj.Y - 1/2);      % +/- 1 for Y(m,t)'s
            SignInd = (sign(Phat) == PMonesY);
            sig = obj.sigma;
            
            % To compute maximizer of F=log p(y|z) - 1/2/Pvar (z - Phat)^2,
            % perform a second-order Taylor series approximation and
            % maximize it instead.  For those entries of Phat whose sign
            % matches that of PMonesY (above), take the Taylor series
            % approximation about Phat, otherwise take the approximation
            % about zero
            EP = zeros(size(obj.Y));
            EP(SignInd) = Phat(SignInd);    % Expansion point of Taylor series
            Comm = (2 + EP.^2);     % Common quantity that appears often
            
            % Calculate value of first derivative of F about either Phat or
            % zero (the expansion point, EP)
            Deriv1 = (2/sig) * (PMonesY.*(Comm.^(3/2)) + EP.*Comm) - ...
                (1./Pvar) .* (EP - Phat);
            
            % Calculate value of second derivative of F about expansion
            % point EP
            Deriv2 = (-2/sig^2) * (3*PMonesY.*EP.*sqrt(Comm) + ...
                3*EP.^2 + 2) ./ ((Comm.^(3/2) + PMonesY.*EP.*Comm).^2) - ...
                (1./Pvar);
            
            % Compute Zhat, the maximizer of the Taylor series
            % approximation of F
            Zhat = EP - (Deriv1 ./ Deriv2);
            
            % Now compute second derivative of log p(y|z) evaluated at Zhat
            Comm = (2 + Zhat.^2);     % Common quantity that appears often
            Zvar = (-2/sig^2) * (3*PMonesY.*Zhat.*sqrt(Comm) + ...
                3*Zhat.^2 + 2) ./ ((Comm.^(3/2) + PMonesY.*Zhat.*Comm).^2);
            
            % Output in Zvar a function of the 2nd derivative that,
            % once manipulated by gampEst, yields the desired
            % max-sum expression for -g'_{out}
            Zvar = Pvar ./ (1 - Pvar.*Zvar);
        end
        
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % This function will compute log p(y|z) evaluated at z = Zhat
        function ll = logLike(obj, Zhat, Zvar)
            % Allocate storage
            ll = NaN(size(obj.Y));
            y = logical(obj.Y);
            
            % Compute log p(y = 1 | z)
            ll(y) = log( max(tcdf(Zhat(y) / obj.sigma, 2), eps) );
            
            % Compute log p(y = 0 | z)
            ll(~y) = log( max(1 - tcdf(Zhat(~y) / obj.sigma, 2), eps) );
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
            S = size(obj.Y, 2);
        end
    end
end