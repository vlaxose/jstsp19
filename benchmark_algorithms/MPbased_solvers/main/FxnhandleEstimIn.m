classdef FxnhandleEstimIn < EstimIn
    % FxnhandleEstimIn:  Creates EstimIn object given a fxn handle to 
    %   a black-box denoiser (e.g., BM3D).  In particular, it uses the 
    %   D-AMP approach, which computes a Monte-Carlo estimate of the 
    %   denoiser's divergence (i.e., its avgerage gradient).
    %   (See http://dsp.rice.edu/software/DAMP-toolbox)
    %
    % Usage: EstimIn = FxnhandleEstimIn(fxnDenoise)
    %   where fxnDenoise is handle to a function of the form
    %     xhat = fxnDenoise(rhat,rvar)
    %   such that
    %     rhat is noisy version of the truth,
    %     rvar is the variance of the assumed (zero-mean) noise
    %     xhat is a de-noised version of rhat
    %   Here, rhat can be a vector or a matrix. In the latter case, 
    %   the divergence is estimated separately on each column.
    %   Likewise, rvar can be a scalar, a row vector (with same # of
    %   rows as rhat), or a matrix (with same dimensions as rhat).

    properties
        fxnDenoise; % fxn handle to point estimator 
        changeFactor = 1e-1; % controls amount to perturb input for divergence
        avg = 1; % # monte-carlo averages to use for computing divergence
        divMin = 0; % minimum allowed divergence
                    % note GEC can croak if div<0
        divMax = 1-1e-5; % maximum allowed divergence
                         % note non-convex costs can yield div>1
                         % but GEC can croak if div=>1
        maxTry = 5; % max # probes to get divergence in [0,1]
    end

    methods
        % Constructor
        function obj = FxnhandleEstimIn(fxnDenoise,varargin)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.fxnDenoise = fxnDenoise;

                if nargin >= 2
                    for i = 1:2:length(varargin)
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end

            end
        end

        % Estimation function
        function [xhat,xvar,val] = estim(obj, rhat, rvar)

            % get size
            [N,L] = size(rhat);

            % call the point estimator
            xhat = obj.fxnDenoise(rhat,rvar);

            % compute approximate divergence
            %epsilon = obj.changeFactor*0.1*max(abs(rhat),[],1) + eps;
            %epsilon = obj.changeFactor*0.1*mean(abs(rhat),1) + eps;
            epsilon = obj.changeFactor*min( sqrt(mean(rvar,1)), mean(abs(rhat),1) ) + eps;
            ll = [1:L]; % column indices at which to (re)compute divergence
            LL = length(ll); % number of columns to compute
            tries = 0; % counter for # tries
            while (tries<obj.maxTry)&&(LL>0)
              div_ = nan(obj.avg,LL);
              for i=1:obj.avg % for each monte-carlo average
                eta = sign(randn(N,LL)); % random direction
                rhat_perturbed = rhat(:,ll) + bsxfun(@times,epsilon(ll),eta);
                if size(rvar,2)>1, rvar_=rvar(:,ll); else rvar_=rvar(:,1); end;
                xhat_perturbed = obj.fxnDenoise(rhat_perturbed,rvar_);
                div_(i,:) = mean( eta.*(xhat_perturbed-xhat(:,ll)) ,1)./epsilon(ll);
              end;
              div(ll) = mean(div_,1); % average divergence for each column
              ll = find((div<obj.divMin)|(div>obj.divMax)); % exceed limits 
              LL = length(ll);
              tries = tries + 1;
            end % while
            div = max(min(div,obj.divMax),obj.divMin); % enforce limits

            % compute posterior variance
            xvar = rvar.*div;

            % gampEst wants xvar to have same # rows as xhat
            if size(xvar,1)==1, xvar=ones(N,1)*xvar; end;

            % Not clear how to set this, so we'll set it at zero
            val = zeros(N,L);
        end

        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(~)
            mean0 = 0; %For now, set at arbitrary constants
            var0  = 1;
            valInit = -Inf;
        end

    end

end
