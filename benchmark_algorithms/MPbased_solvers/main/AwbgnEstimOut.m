classdef AwbgnEstimOut < EstimOut
    % AwbgnEstimOut:  Implements the additive white Bernoulli Gaussian noise
    % with the likelihood
    %
    % p_{Y|Z}(y|z) = (1-lambda) delta(z-y) + lambda N(z; y; wvar)
    %
    % Coded By Jeremy Vila
    % 9-05-14
    
    properties
        y;      % Measured output
        wvar = 1;   % Variance on the "noisy" component
        lambda = 1-eps; % Average number of noisy elements to total elements
        %Default case is (nearly) AWGN with variance 1.
    end
    
    methods
        % Constructor
        function obj = AwbgnEstimOut(y,wvar,lambda)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.y = y;
                obj.wvar = wvar;
                obj.lambda = lambda;
                
                %Warn user about inputs
                if any(~isreal(obj.y)),
                    error('First argument of AwbgnEstimOut must be real-valued.');
                end;
                if any(any((obj.wvar<0)))||any(any(~isreal(obj.wvar))),
                    error('Second argument of AwbgnEstimOut must be non-negative');
                end;
                if (lambda <= 0 || lambda >= 1)
                    error('Thirs argument must fall strictly between 0 and 1');
                end
                if any(obj.wvar==0)
                    warning(['Tiny non-zero variances will be used for'...
                        ' computing log likelihoods. May cause problems'...
                        ' with adaptive step size if used.']) %#ok<*WNTAG>
                end
                
            end
        end
        
        % Prior mean and variance
        function [wvar, lambda, valInit] = estimInit(obj)
            wvar  = obj.wvar;
            lambda = obj.lambda;
            valInit = 0;
        end
        
        % AWBGN estimation function
        % Provides the posterior mean and variance of z
        % from an observation y = z + w
        % where z = N(zmean0,zvar0) and w is zero mean BG(lambda, wvar);
        function [zhat, zvar] = estim(obj, phat, pvar)
            
            %Compute extrinsic log likelihood that measurement is noiseless
            loglike0 = -0.5*( log(2*pi) + log(pvar) + ...
                (phat - obj.y).^2./pvar );
            
            %Compute extrinsic log likelihood that measurement is noiseless
            loglike1 = -0.5*(log(2*pi) + log(pvar + obj.wvar) + ...
                (phat - obj.y).^2./(pvar + obj.wvar));
            
            % Convert log-domain quantities into posterior activity
            % probabilities (i.e., py1 = Pr{Z \neq y | y}, py0 = Pr{Z = y | y})
            exparg = loglike0 - loglike1 + log(1 - obj.lambda) - log(obj.lambda);
            py1 = (1 + exp(exparg)).^(-1);
            py0 = 1 - py1;
            
            %Compute the posterior variance given Z is noisy
            nu = obj.wvar.*pvar ./ (pvar + obj.wvar);
            %Compute the posterior mean given Z is noisy
            gamma = (obj.wvar.*phat + obj.y.*pvar)./ (pvar + obj.wvar);
            
            %Now compute the posterior mean and variance of z
            zhat = py1.*gamma + py0.*obj.y;
            zvar = py1.*(abs(gamma).^2 + nu) + py0.*(abs(obj.y).^2)...
                - abs(zhat).^2;
            
        end
        
        % Compute output cost:
        % For sum-product compute
        %   E_Z( log p_{Y|Z}(y|z) ) with Z ~ N(zhat, zvar)
        %
        % Unfortunately, this quantity cannot be computed.  Instead
        % compute a lower bound by calculating the logLike 
        % assuming p_{Y|Z}(y|z) is AWGN with variance wvar and scaling by lambda;
        function ll = logLike(obj,zhat,zvar)
            
            %Warn user that logLike has not been implemented yet
            warning(['The logLike method has been implemented for this' ...
                    ' class, but not thoroughly tested.  May return incorrect' ...
                    ' cost when using adaptive step size.']) %#ok<WNTAG>

            % Ensure variance is small positive number
            wvar1 = max(1e-20, obj.wvar);
            
            % Compute log-likelihood
            ll = -0.5*(log(2*pi) + log(obj.wvar) + ...
                (obj.y-zhat).^2 + zvar)./wvar1 + log(obj.lambda);
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        function ll = logScale(obj,Axhat,pvar,phat)

            % Find the fixed-point of phat
            opt.phat0 = phat; 
            opt.alg = 1; % approximate newton's method
            opt.maxIter = 150; 
            opt.tol = 1e-4; 
            opt.stepsize = 0.4; 
            opt.regularization = 0;
            opt.debug = false;
            phatfix = estimInvert(obj,Axhat,pvar,opt);

            % Compute log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
            % Exploit log(a+b) = log(a) + log(1 + b/a) for positive a and b          
            %log(a) term
            ls = log(obj.lambda)-0.5*(log(2*pi) + log(pvar + obj.wvar) ...
                + (obj.y-phatfix).^2./(pvar + obj.wvar));
            
            %Add to to log(1 + b/a) term          
            ls = ls + log(1 + (1-obj.lambda)./obj.lambda.*sqrt(1 + obj.wvar./pvar)...
                .*exp(-(obj.y-phatfix).^2./(2.*pvar.*(pvar./obj.wvar + 1))));
            
%             %Old version of taking log(a + b)
%             ls = log(exp(-(obj.y-phatfix).^2./(2*pvar))...
%                 .*(1-obj.lambda)./sqrt(2*pi*pvar) + ...
%                 exp(-(obj.y-phatfix).^2./(2*(pvar+obj.wvar)))...
%                 .*obj.lambda./sqrt(2*pi*(pvar+obj.wvar)));  
            

            % Combine to form output cost
            ll = ls + 0.5*(Axhat - phatfix).^2./pvar;

        end
        
        function S = numColumns(obj)
            %Return number of columns of Y
            S = size(obj.y,2);
        end
        
        % Generate random samples from p(y|z)
        function y = genRand(obj, z)
            y1 = sqrt(obj.wvar).*randn(size(z)) + z;
            p = rand(size(z)) < obj.lambda;
            y = y1.*p + (1-p).*obj.y;
        end
    end
    
end

