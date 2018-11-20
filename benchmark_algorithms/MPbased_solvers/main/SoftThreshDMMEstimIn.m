classdef SoftThreshDMMEstimIn < EstimIn
    % SoftThreshDMMEstimIn:  
    % Performs Donoho/Maleki/Montanari-style soft thresholding
    % when using GAMP to solve 
    %    "min_x 1/2/var*norm(y-A*x,2)^2 + lambda*norm(x,1)". 
    %
    % Warning: the val output is currently set to zero
    
    properties
        alpha = 1.5; % threshold is set at alpha * sqrt(mean(mur))
        debias = false; % debias the outputs? 
    end
    
    methods
        % Constructor
        function obj = SoftThreshDMMEstimIn(alpha,varargin)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax

                obj.alpha = alpha;

                if nargin >= 2
                    for i = 1:2:length(varargin)
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end

            end
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(~)
            mean0 = 0; %For now, set these to arbitrary constants
            var0  = 1e-2;
            valInit = -Inf;
        end
        
        % Carry out soft thresholding
        function [xhat,xvar,val] = estim(obj,rhat,rvar)
            
            %Compute the threshold
            thresh = bsxfun(@times, obj.alpha, sqrt(mean(rvar,1)));
            
            %Estimate the signal
            xhat = bsxfun(@times, ...
                          max(0,bsxfun(@minus,abs(rhat),thresh)), ...
                          sign(rhat));
            
            %Estimate the variance
            active = (xhat~=0);
            xvar = bsxfun(@times, rvar, mean(double(active),1));
           
            %Debias if needed
            if obj.debias
               rhatOn_minus_thresh = bsxfun(@minus, rhat(active), thresh);
               scale = 1 + thresh.*sum(rhatOn_minus_thresh,1)./ ...
                                   sum((rhatOn_minus_thresh).^2,1);
               xhat = bsxfun(@times,scale,xhat);
               xvar = bsxfun(@times,scale,xvar);
            end

            % gampEst wants xvar to have same # rows as xhat
            if size(xvar,1)==1, xvar=ones(size(xhat,1),1)*xvar; end;
 
            %For now, let's set the val output to zero. Not clear
            %how to properly set this without knowledge of "lambda"
            val = zeros(size(xhat));
            
        end
        
    end
    
end

