% CLASS: ProbitStateEvoEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: IntEstimOutAvg
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The ProbitStateEvoEstimOut class defines a scalar observation channel, 
%   p(y|z), that constitutes a probit binary classification model, i.e., 
%   y is an element of the set {0,1}, z is a real number, and
%             	 p(y = 1 | z) = Phi((z - Mean)/sqrt(Var)),
%   where Phi((x-b)/sqrt(c)) is the cumulative density function (CDF) of a
%   Gaussian random variable with mean b, variance c, and argument x.
%   Typically, mean = 0 and var = 1, thus p(y = 1 | z) = Phi(z).
%
%   This class is used by the gampSE function for the purpose of
%   numerically evaluating state evolution-related integrals.
%
% PROPERTIES (State variables)
%   Mean        [Optional] A scalar probit function mean (see DESCRIPTION) 
%               [Default: 0]
%   Var         [Optional] A scalar probit function variance (see 
%               DESCRIPTION) [Default: 1e-2]
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: false]
%
% METHODS (Subroutines/functions)
%   ProbitStateEvoEstimOut(Np, Ny, Nz)
%       - Default constructor.  Specifies number of discrete integration
%         points used for the random variables P, Y, and Z when calculating
%         state evolution-related expectations.  Assigns Mean and Var to 
%         default values.
%   ProbitEstimOut(Np, Ny, Nz, Mean, Var, maxSumVal)
%       - Optional constructor.  In addition to setting Np, Ny, and Nz, 
%         also sets any/all non-empty assignment of Mean, Var, and 
%         maxSumVal.
%   [zhat, zvar] = estim(obj, y, p, taup)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the probit model and maxSumVal = false (see 
%         DESCRIPTION), and p(z) = Normal(p,taup).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%   [y, pyzp] = getypts(obj, z, pz, Ny)
%       - Given a vector of values of the random variable z, and associated
%         prior probabilities, pz, this method will return a length-Ny
%         vector of values of y, and an Ny-by-Nz matrix of posterior
%         probabilities pyzp, where pyzp(i,j) = p(y(i) | z(j))
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/12/12
% Change summary: 
%       - Created from ProbitEstimOut (12/12/12; JAZ)
% Version 0.2
%

classdef ProbitStateEvoEstimOut < IntEstimOutAvg
    
    properties
        Mean = 0;       % Probit function mean [dflt: 0]
        Var = 1;    	% Probit function variance [dflt: 1]
        maxSumVal = false;   % Sum-product (false) or max-sum (true) GAMP?
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = ProbitStateEvoEstimOut(Np, Ny, Nz, Mean, Var, ...
                maxSumVal)
            if nargin < 3 || ~isscalar(Np) || ~isscalar(Ny) || ~isscalar(Nz)
                error('Constructor requires at least three arguments')
            end
            obj = obj@IntEstimOutAvg(Np, Ny, Nz);
            if nargin >= 2 && ~isempty(Mean) && isscalar(Mean)
                % Mean property is an argument
                obj.Mean = Mean;
            end
            if nargin >= 3 && ~isempty(Var) && isscalar(Var)
                % Var property is an argument
                obj.Var = Var;
            end
            if nargin >= 4 && ~isempty(maxSumVal)
                % maxSumVal property is an argument
                obj.maxSumVal = maxSumVal;
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.Mean(obj, Mean)
                obj.Mean = double(Mean);
        end
        
        function obj = set.Var(obj, Var)
            if any(Var(:) <= 0)
                error('Var must be strictly positive')
            else
                obj.Var = double(Var);
            end
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            if isscalar(maxsumval) && islogical(maxsumval)
                obj.maxSumVal = maxsumval;
            else
                error('ProbitEstimOut: maxSumVal must be a logical scalar')
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
        function [zhat, zvar] = estim(obj, y, p, taup)
            % Construct an object of the ProbitEstimOut class, and call its
            % estim method
            EstimOutObj = ProbitEstimOut(y, obj.Mean, obj.Var, ...
                obj.maxSumVal);
            [zhat, zvar] = EstimOutObj.estim(p, taup);
            EstimOutObj.delete();
        end
        
        
        % *****************************************************************
        %                         GETYPTS METHOD
        % *****************************************************************
        % This function accepts a length-Nz vector, z, containing values of
        % the random variable Z, and associated prior probabilities, pz,
        % and returns a length-Ny vector, y, of values of the random
        % variable Y, and an Ny-by-Nz matrix of posterior probabilities,
        % pyzp, where pyzp(i,j) = p(y(i) | z(j))
        function [y, pyzp] = getypts(obj, z, ~, Ny)
            % Since we only have two possible values of the random
            % variable Y, evenly allocated both choices in the length-Ny
            % output vector, y
            Nones = round(Ny / 2);  % # of class-1 outputs
            Nzeroes = Ny - Nones;   % # of class-0 outputs
            y = [ones(Nones,1); -ones(Nzeroes,1)];  % +/-1 representation
            
            % Calculate p(y|z) matrix
            pyzp = normcdf(y * (z - obj.Mean).', 0, sqrt(obj.Var));
            
            y = (y > 0);    % {-1,1} -> {0,1}
        end
    end
end