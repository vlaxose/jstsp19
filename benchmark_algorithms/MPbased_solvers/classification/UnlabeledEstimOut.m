% CLASS: UnlabeledEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The UnlabeledEstimOut class is useful in semi-supervised learning tasks
%   where one wishes to make use of both labeled and unlabeled data.  Under
%   the GAMP model that relates unknowns to observations (x -> z = Ax -> y)
%   this corresponds to the case where certain entries of the vector z have
%   no associated entries in the observation/label vector y (i.e., there
%   are rows of the data matrix, A, whose corresponding rows in y are
%   absent).
%
%   The UnlabeledEstimOut class acts as a "wrapper" class around another
%   EstimOut object that defines the relationship between z and y **at 
%   those locations where labels are present**.  This EstimOut object,
%   together with a binary mask that defines which entries of z are
%   labeled, is used to construct an UnlabeledEstimOut object.
%
% PROPERTIES (State variables)
%   EstimOutObj     An EstimOut object that describes the scalar
%                   relationship between the labeled entries of z and their
%                   labeled values in y.  Note that the estim method of
%                   this EstimOut object will be called only for the
%                   vectorized subset of the z array whose entries are 
%                   labeled
%   Mask            A binary array of the same dimensions as the "complete"
%                   z array, i.e., the array consisting of all labeled and
%                   unlabeled entries of z.  If Mask(i,j) = 1, then z(i,j)
%                   is assumed to be labeled, and unlabeled otherwise.
%   maxSumVal       This property specifies whether to perform sum-product
%                   (false) or max-sum (true) GAMP computations [Default:
%                   false]
%
% METHODS (Subroutines/functions)
%   UnlabeledEstimOut(EstimOutObj, Mask)
%       - Object constructor, consisting of a valid EstimOut object and a
%         binary array specifying labeled/unlabeled values of the
%         "complete" z array (see PROPERTIES)
%   UnlabeledEstimOut(EstimOutObj, Mask, maxSumVal)
%       - Optional full constructor.  Assigns maxSumVal as well.
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
% Last change: 11/19/12
% Change summary: 
%       - Created (11/19/12; JAZ)
% Version 0.2
%

classdef UnlabeledEstimOut < EstimOut
    
    properties
        EstimOutObj;        % EstimOut object for labeled entries of z
        Mask;               % Binary array specifying labeled/unlabeled
                            % entries of the "complete" z array
        maxSumVal = false;  % Sum-product (false) or max-sum (true) GAMP?
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = UnlabeledEstimOut(EstimOutObj, Mask, maxSumVal)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                if nargin < 2 || isempty(EstimOutObj) || isempty(Mask)
                    error('Constructor requires two arguments')
                else
                    obj.EstimOutObj = EstimOutObj;
                    obj.Mask = Mask;
                end
                if nargin >= 3 && ~isempty(maxSumVal)
                    % maxSumVal property is an argument
                    obj.maxSumVal = maxSumVal;
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.EstimOutObj(obj, EstimOutObj)
            if isa(EstimOutObj, 'EstimOut')
                obj.EstimOutObj = EstimOutObj;
            else
                error('EstimOutObj must be a valid EstimOut object')
            end
        end
        
        function obj = set.Mask(obj, Mask)
            if ~all((Mask(:) == 0) | (Mask(:) == 1))
                error('Elements of Mask must be binary {0,1}')
            else
                obj.Mask = Mask;
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
        % For labeled entries of z (i.e., those entries for which
        % Mask(i,j) = 1), this method will call the estim method of
        % obj.EstimOutObj on a vectorized subset of inputs (Phat & Pvar),
        % while unlabeled entries will return their matching entries in
        % Phat, and, if maxSumVal = false, Pvar, or, if maxSumVal = true,
        % the value zero (0).
        function [zhat, zvar] = estim(obj, phat, pvar)
            [zhat, zvar] = deal(NaN(size(phat)));   % Initialize
            
            % First deal with labeled entries...
            [zhat(obj.Mask), zvar(obj.Mask)] = ...
                obj.EstimOutObj.estim(phat(obj.Mask), pvar(obj.Mask));
            
            % ...and now unlabeled entries
            if ~obj.maxSumVal
                % Sum-product updates
                zhat(~obj.Mask) = phat(~obj.Mask);
                zvar(~obj.Mask) = pvar(~obj.Mask);
            else
                % Max-sum updates
                zhat(~obj.Mask) = phat(~obj.Mask);
                zvar(~obj.Mask) = 0;
            end
        end
        
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % If performing sum-product GAMP, this method will return the
        % sum of expected log-likelihoods **for all labeled entries of z**,
        % i.e., ll = \sum_i E[log p(y(i) | z(i))], where i is the index to 
        % a labeled entry of the "complete" z array.  If performing max-sum
        % GAMP (obj.maxSumVal = true), logLike returns the sum of
        % posterior log-likelihoods evaluated at z(i) = zhat(i)
        function ll = logLike(obj, zhat, zvar)
            ll = sum( obj.EstimOutObj.logLike(zhat(obj.Mask), ...
                zvar(obj.Mask)) );
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
            % Return number of columns of z
            S = numColumns(obj.EstimOutObj);
        end
    end
    
end