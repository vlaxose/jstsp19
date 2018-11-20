classdef BGZeroMeanEstimIn < EstimIn
    % BGZeroMeanEstimIn:  Implements zero mean Bernoulli-Gaussian input
    % estimator.
    % Note that this can also be achieved by combining AwgnEstimIn with
    % SparseScaEstim. However, this version avoids some intermediate
    % calculations and offers a computational improvement for this special
    % case.
    
    properties
        % Prior mean and variance
        var0;   % Variance
        p1;     % Sparsity
        
        % True indicates to compute output for max-sum. Currently only MMSE
        % is implemented
        maxSumVal = false;
    end
    
    methods
        % Constructor
        function obj = BGZeroMeanEstimIn(var0,p1, maxSumVal)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.var0 = var0;
                obj.p1 = p1;
                if (nargin >= 3)
                    if (~isempty(maxSumVal))
                        obj.maxSumVal = maxSumVal;
                    end
                end
                
                % warn user about inputs
                if any((var0(:)<0))||any(~isreal(var0(:))),
                    error('Second argument of AwgnEstimIn must be non-negative');
                end;
            end
        end
        
        % Prior mean and variance
        function [xhat0, xvar0, valInit] = estimInit(obj)
            xhat0 = zeros(size(obj.var0));
            xvar0  = obj.var0 .* obj.p1;
            valInit = 0;
        end
        
        % Zero mean BG estimation function
        % Provides the mean and variance of a variable u
        % from a observation real(v) = u + w, w = N(0,wvar)
        %
        function [umean, uvar, val,py1] = estim(obj, v, wvar)
            
            % Get prior
            uvar0 = obj.var0;
            lam = obj.p1;
            
            %First, compute nu
            nu = wvar.*uvar0 ./ (uvar0 + wvar);
            gamma = (nu .* v) ./ wvar;
            
            %Now alpha
            alpha = 1 + (1 - lam)./lam .* sqrt(uvar0 ./ nu) .*...
                exp(-0.5*gamma.^2 ./ nu);
            
            %Mean
            umean = gamma ./ alpha;
            
            %Variance
            uvar = gamma.^2.*(alpha - 1) ./ alpha.^2 + nu ./ alpha;
            
            if (nargout >= 3)
                if ~(obj.maxSumVal)
                    % Compute the negative KL divergence
                    %   klDivNeg = \sum_i \int p(u|v)*\log( p(u) / p(u|v) )du
                    
                    %First, compute KL for the non-sparse Gaussian
                    val = 0.5* (log(nu./uvar0) + (1-nu./uvar0) ...
                        - (gamma).^2./uvar0 );
                    
                    %Compute the pis
                    py1 = 1 ./ alpha;
                    py0 = 1 - py1;
                    
                    %Finish
                    val = py1.*val + py1.*log(max(1e-8,obj.p1)./max(py1,1e-8)) ...
                        + py0.*log(max(1e-8,(1-obj.p1))./max(py0,1e-8));
                else
                    % Evaluate the (log) prior
                    error('This estimator is not implemented for maxsum')
                end
            end
            
        end
        
        % Generate random samples
        function x = genRand(obj, outSize)
            if isscalar(outSize)
                x = sqrt(obj.var0).*randn(outSize,1).*(rand(outSize,1) <= obj.p1);
            else
                x = sqrt(obj.var0).*randn(outSize).*(rand(outSize) <= obj.p1);
            end
        end
        
        
        
    end
    
end

