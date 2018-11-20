% CLASS: SemiSuperProbitEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The SemiSuperProbitEstimOut class defines a scalar observation channel, 
%   p(y|z), that constitutes a probit binary classification model when a 
%   portion of the training data is unlabeled.  To incorporate unlabeled
%   data, this class applies a variation of the null category noise model
%   (NCNM) proposed by Lawrence and Jordan, "Semi-supervised Learning via
%   Gaussian Processes," 2005.  Specifically, we consider an output channel
%   of the form:
%                   / Phi(-(z/v + w/2)),                y = -1
%        p(y | z) = | Phi(z/v + w/2) - Phi(z/v - w/2),  y = 0
%                   \ Phi(z/v - w/2),                   y = 1,
%   where Phi(.) is the cumulative density function (CDF) of a standard
%   Normal random variable (i.e., N(0,1)).  y is an element of the set
%   {-1,*,1}, where the two labeled binary classes are {-1,1}, and the null
%   category class is {0}.  The two labeled classes have sigmoidal
%   likelihoods of opposite orientations that are separated by a "null
%   category region" of width w.  It is assumed that *no* unlabeled
%   training examples can come from the null category region, thus the
%   region acts to exclude unlabeled points.  In addition, a parameter
%   Gamma is used to denote the probability of a missing label (a good rule
%   of thumb is to set Gamma equal to the proportion of the training set
%   that is unlabeled).
%
%   Note that, for convenience, this class uses a ternary coding scheme
%   for the variable Y (see below).  If Y(m,t) = +/-1, it is assumed that
%   the (m,t)^th training example was observed, and has a binary label from
%   {-1,1}.  If Y(m,t) = 0, then the (m,t)^th training example is
%   considered unlabeled.  (Do not confuse this ternary coding scheme with
%   the output channel likelihood described above; no unlabeled points can
%   come from the null category class).
%
% PROPERTIES (State variables)
%   Y           An M-by-T array of ternary ({-1,0,1}) codes for the
%               training data, where M is the number of training data
%               points, and T is the number of classifiers being learned
%               (typically T = 1).  Use {-1,1} to denote the labels of
%               observed training examples, and {0} to denote unlabeled
%               examples (see DESCRIPTION).
%   Width       The width of the null category region, w, (see
%               DESCRIPTION).  The wider the null category region is, the
%               farther apart it is assumed that unlabeled points lie from
%               the decision boundary.
%   Gamma       The probability that a label is missing.
%   Var         [Optional] The probit function variance, v, (see 
%               DESCRIPTION).  Smaller values of v correspond to more
%               step-like sigmoidal shapes.  [Default: 1e-2]
%   maxSumVal 	Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: false]
%
% METHODS (Subroutines/functions)
%   SemiSuperProbitEstimOut(Y, Width, Gamma)
%       - Default constructor.  Assigns Var and maxSumVal to default values.
%   ProbitEstimOut(Y, Width, Gamma, Var, maxSumVal)
%       - Full constructor.  Sets all properties.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the probit model and maxSumVal = false (see 
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 03/21/13
% Change summary: 
%       - Created (03/21/13; JAZ)
% Version 0.2
%

classdef SemiSuperProbitEstimOut < EstimOut
    
    properties (Dependent)
        Y;              % M-by-T vector of binary class labels
        Width;          % M-by-T vector of probit function means [dflt: 0]
        Var;            % M-by-T vector of probit function variances [dflt: 1e-2]
        maxSumVal;      % Sum-product (false) or max-sum (true) GAMP?
    end
    
    properties
        Gamma;          % Missing label probability
    end
    
    properties (Access = private)
        Y_priv;                     % Private Y
        Width_priv;                 % Private Width
        Var_priv = 1e-2;            % Private Var
        maxSumVal_priv = false;  	% Private maxSumVal
        recalc_L = true;            % Re-construct the LabeledEstimOut class
        recalc_U = true;            % Re-construct the UnlabeledEstimOut class
    end
    
    properties (SetAccess = private)
        LabeledEstimOut;    % A ProbitEstimOut class for the labeled data
        UnlabeledEstimOut;  % A ProbitEstimOut class for the unlabeled data
        LabeledIdx;         % Boolean of same size as Y, w/ 1 = labeled
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = SemiSuperProbitEstimOut(Y, Width, Gamma, Var, ...
                maxsumval)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                assert(nargin >= 3 && ~isempty(Y) && ~isempty(Width) && ...
                    ~isempty(Gamma), 'Insufficient arguments');
                obj.Y = Y;      % Set Y
                obj.Width = Width;
                obj.Gamma = Gamma;
                if nargin >= 4 && ~isempty(Var)
                    % Var property is an argument
                    obj.Var = Var;
                end
                if nargin >= 5 && ~isempty(maxsumval)
                    % maxSumVal property is an argument
                    obj.maxSumVal = maxsumval;
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        
        function obj = set.Width(obj, w)
            assert(isnumeric(w) && isscalar(w) && w >= 0, ...
                'Width must be a non-negative scalar')
            obj.Width_priv = double(w);
            obj.recalc_L = true;        % Set flag
            obj.recalc_U = true;        % Set flag
        end
        
        function obj = set.Gamma(obj, gam)
            assert(isnumeric(gam) && isscalar(gam) && gam >= 0 && gam <= 1, ...
                'Gamma must be a scalar in [0,1]')
            obj.Gamma = double(gam);
        end
        
        function obj = set.Y(obj, Y)
            if ~all((Y(:) == -1) | (Y(:) == 0) | (Y(:) == 1))
                error('Invalid elements in Y')
            end
            obj.Y_priv = Y;
            obj.LabeledIdx = (Y ~= 0);      % Refresh list of labeled points
            obj.recalc_L = true;            % Set flag
            obj.recalc_U = true;            % Set flag
        end
        
        function obj = set.Var(obj, Var)
            if ~isnumeric(Var) || ~isscalar(Var) || Var <= 0
                error('Var must be strictly positive')
            else
                obj.Var_priv = double(Var);
                obj.recalc_L = true;            % Set flag
                obj.recalc_U = true;            % Set flag
            end
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            if isscalar(maxsumval) && islogical(maxsumval) && ~maxsumval
                obj.maxSumVal_priv = maxsumval;
            elseif isscalar(maxsumval) && islogical(maxsumval) && maxsumval
                error('Max-sum updates not supported yet')
            else
                error('maxSumVal must be a logical scalar')
            end
        end
        
        
        % *****************************************************************
        %                           GET METHODS
        % *****************************************************************
        
        function w = get.Width(obj)
            w = obj.Width_priv;
        end
        
        function Y = get.Y(obj)
            Y = obj.Y_priv;
        end
        
        function v = get.Var(obj)
            v = obj.Var_priv;
        end
        
        function msv = get.maxSumVal(obj)
            msv = obj.maxSumVal_priv;
        end
        
        function LEO = get.LabeledEstimOut(obj)
            if obj.recalc_L
                % Re-construct the EstimOut object
                LabeledY = double(obj.Y(obj.LabeledIdx) > 0);
                Means = (sqrt(obj.Var) * obj.Width / 2) * ...
                    obj.Y(obj.LabeledIdx);  % Shift sigmoids for +/-1 classes
                obj.LabeledEstimOut = ProbitEstimOut(LabeledY, Means, ...
                    obj.Var, obj.maxSumVal);
                obj.recalc_L = false;   % Clear flag
            end
            LEO = obj.LabeledEstimOut;
        end
        
        function UEO = get.UnlabeledEstimOut(obj)
            if obj.recalc_U
                % Re-construct the EstimOut object
                NumUL = sum(~obj.LabeledIdx(:));    % # of unlabeled points
                UnlabeledY = [ones(NumUL,1); zeros(NumUL,1)];
                Means = [(sqrt(obj.Var) * obj.Width / 2) * ones(NumUL,1);
                    -(sqrt(obj.Var) * obj.Width / 2) * ones(NumUL,1)];
                obj.UnlabeledEstimOut = ProbitEstimOut(UnlabeledY, Means, ...
                    obj.Var, obj.maxSumVal);
                obj.recalc_U = false;   % Clear flag
            end
            UEO = obj.UnlabeledEstimOut;
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
                    % Return sum-product expressions to GAMP in Zhat and
                    % Zvar.  Treat the estimation of labeled and unlabeled
                    % points separately.  Begin with the labeled points...
                    [ZhatL, ZvarL] = obj.LabeledEstimOut.estim(...
                        Phat(obj.LabeledIdx), Pvar(obj.LabeledIdx));
                    
                    % Now the unlabeled points...
                    NumUL = sum(~obj.LabeledIdx(:));
                    wMod = sqrt(obj.Var) * obj.Width / 2;
                    Con1 = (Phat(~obj.LabeledIdx) - wMod) ./ ...
                        sqrt(obj.Var + Pvar(~obj.LabeledIdx));
                    Con0 = (Phat(~obj.LabeledIdx) + wMod) ./ ...
                        sqrt(obj.Var + Pvar(~obj.LabeledIdx));
                    CDF1 = normcdf(Con1);   
                    CDF0 = normcdf(-Con0);   
                    NormCon1 = (1 + CDF0./CDF1).^(-1);  % Upper normalizing constant
                    NormCon0 = (1 + CDF1./CDF0).^(-1);  % Lower normalizing constant
                    
                    % Get the conditional expectation for the unlabeled
                    % points under both hypotheses (y = 1 & y = -1)
                    [ZhatU2, ZvarU2] = obj.UnlabeledEstimOut.estim(...
                        [Phat(~obj.LabeledIdx); Phat(~obj.LabeledIdx)], ...
                        [Pvar(~obj.LabeledIdx); Pvar(~obj.LabeledIdx)]);
                    
                    % Compute the posterior mean of the unlabeled points
                    ZhatU = NormCon1 .* ZhatU2(1:NumUL) + ...
                        NormCon0 .* ZhatU2(NumUL+1:end);
                    ZhatU(isnan(ZhatU)) = Phat(isnan(ZhatU));
                    
                    % Compute the second moment for the unlabeled points
                    % under both hypotheses
                    SecMom1 = ZvarU2(1:NumUL) + abs(ZhatU2(1:NumUL)).^2;
                    SecMom0 = ZvarU2(NumUL+1:end) + abs(ZhatU2(NumUL+1:end)).^2;
                    
                    % Compute weighted average of both second moments to
                    % get the posterior second moment of the unlabeleds
                    SecMom = NormCon1 .* SecMom1 + NormCon0 .* SecMom0;
                    
                    % Now the posterior variance of the unlabeled points
                    ZvarU = SecMom - abs(ZhatU).^2;
                    
                    % Combine the labeled and unlabeled updates now
                    [Zhat, Zvar] = deal(NaN(size(Phat)));
                    Zhat(obj.LabeledIdx) = ZhatL;
                    Zhat(~obj.LabeledIdx) = ZhatU;
                    Zvar(obj.LabeledIdx) = ZvarL;
                    Zvar(~obj.LabeledIdx) = ZvarU;
                case true
                    % Not presently implemented
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
            switch obj.maxSumVal
                case false
                    % Evaluate the expected log-likelihood only for
                    % labeled data
                    wMod = sqrt(obj.Var) * obj.Width / 2;
                    LabeledY = obj.Y(obj.LabeledIdx);
                    Mean = wMod * LabeledY;
                    
                    % Start by computing the critical constant, C, on which
                    % the remainder of the computations depend.  Modulate 
                    % this constant by -1 for cases where Y(m,t) = 0.
                    C = LabeledY .* ((Zhat(obj.LabeledIdx) - Mean) ./ ...
                        sqrt(Zvar(obj.LabeledIdx) + obj.Var));
                    CDF = normcdf(C);
                    ll = sum(log(CDF(:)));
                case true
                    ll = log(normcdf(PMonesY .* ...
                        (Zhat - obj.Mean)/sqrt(obj.Var), 0, 1));
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
            S = size(obj.Y, 2);
        end
    end
end