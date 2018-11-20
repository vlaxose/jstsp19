classdef DiracEstimOut < EstimOut
    % DiracEstimOut:  Dirac-delta scalar output estimation function
    %
    % Corresponds to an output channel of the form
    %   y = scale*z 
    % 
    % z and y can be real or complex
    
    properties
        % Prior mean and variance
        y;      % Measured output
        scale = 1;  % scale factor
        cplxFlag = false;
        
        % True indicates to compute output for max-sum
        maxSumVal = false;
    end
    
    methods
        % Constructor
        function obj = DiracEstimOut(y, maxSumVal, scale, cplxFlag)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.y = y;
                %Return error if complex measurements
                if (nargin >= 2)
                    if (~isempty(maxSumVal))
                        obj.maxSumVal = maxSumVal;
                    end
                end
                if (nargin >= 3)
                    obj.scale = scale;            
                end
                if (nargin >=4)
                    obj.cplxFlag = cplxFlag;            
                end
            end
        end
        
        % Return size
        function [nz,ncol] = size(obj)
            [nz,ncol] = size(obj.y);
        end
        
        % Dirac estimator function
        % Provides the posterior mean and variance of variable z
        % from an observation y = scale*z, z = CN(zmean0,zvar0) 
        function [zmean, zvar] = estim(obj, zmean0, zvar0)
            % Compute posterior mean and variance
            zmean = obj.y/obj.scale;
            zvar = zeros(size(zmean0));
        end
        
        % Compute log likelihood
        % For sum-product GAMP, compute
        %   E( log p_{Y|Z}(y|z) ) with z = CN(zhat, zvar)
        % For max-sum GAMP compute
        %   log p_{Y|Z}(y|z) @ z = zhat
        % Return zero for Log likelihood 
        function ll = logLike(obj,zhat,zvar)
            %For both sum-product and max-sum GAMP return zeros as the
            %log-likelihood cost.  log(0) does not exist, as does log of
            %the Dirac function.
            ll = zeros(size(obj.y));
        end
        
        % Compute output cost for real case:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        %
        % For complex case replace N with CN and remove 1/2's
        function ll = logScale(obj,Axhat,pvar,phat)
                   
%             error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');  
            
            %Get scale
            s = obj.scale;
            
            if ~(obj.maxSumVal)
                
                if ~obj.cplxFlag
                    %Compute the log scale + 0.5*(Axhat - phat)^2/pvar and
                    %simplify to get the total output cost
                    ll = -0.5*log(2*pi*pvar) - log(s);
                else
                    %Compute the log scale + abs(Axhat - phat)^2/pvar and
                    %simplify to get the total output cost
                    ll = -log(pi*pvar) - log(abs(s));
                end
            else
                %For both max-sum GAMP return zeros as the
                %log-likelihood cost.  log(0) does not exist,
                ll = zeros(size(obj.y));
            end
            
        end
        
        %Return number of columns of y
        function S = numColumns(obj)
            S = size(obj.y,2);
        end
        
        % Generate random (in this case deterministic) samples from p(y|z)
        function y = genRand(obj, z)
            y = obj.scale.*z;
        end
    end
    
end

