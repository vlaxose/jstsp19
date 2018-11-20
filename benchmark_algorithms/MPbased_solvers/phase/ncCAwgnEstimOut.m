classdef ncCAwgnEstimOut < EstimOut
    % ncCAwgnEstimOut:  noncoherent CAWGN scalar output estimation function
    %
    % Corresponds to an output channel of the form
    %   mean0 = exp(j*theta)*(z + CN(0,var0)) for theta = U[0,2pi)
    
    properties
        % Prior mean and variance
        mean0;  % Mean
        var0;   % Variance
        
        % True indicates to compute output for max-sum
        maxSumVal = false;

        % True indicates to automatically estimate var0
        autoTune = false;
    end
    
    methods
        % Constructor
        function obj = ncCAwgnEstimOut(mean0, var0, maxSumVal, autoTune)
            obj = obj@EstimOut;
            obj.mean0 = mean0;
            if (nargin < 2)
                obj.var0 = 0.01*abs(mean0).^2;
            else
                obj.var0 = var0;
            end
            if (nargin >= 3)
                obj.maxSumVal = maxSumVal;
            end
            if (nargin >= 4)
                obj.autoTune = autoTune;
            end
            
            %Warn user if they set zero variance
            if any(obj.var0 == 0)
                warning(['Tiny non-zero variances will be used for'...
                    ' computing log likelihoods. May cause problems'...
                    ' with adaptive step size if used.']) %#ok<*WNTAG>
            end

        end
        
        % ncCAWGN estimation function
        % Provides the posterior mean and variance of a variable z
        % from prior z = CN(phat,pvar) and observation |z+w| = mean0, w = CN(0,var0)
        function [zhat, zvar] = estim(obj, phat, pvar)
            
            y_abs = abs(obj.mean0);
            phat_abs = abs(phat);
            var0 = obj.var0;
           
            if (obj.maxSumVal)
                % Compute posterior mode and sensitivity of u 
                zhat = sign(phat).*(y_abs+phat_abs.*var0./pvar)./(1+var0./pvar);
                zvar = (0.5*y_abs./phat_abs+var0)./(1+var0./pvar); 
            else
                % Compute posterior mean and variance of u
                B = 2*y_abs.*phat_abs./(var0+pvar);
                I1overI0 = min( B./(sqrt(B.^2+4)), ...
                    B./(0.5+sqrt(B.^2+0.25)) );%upper bounds (11)&(16) from Amos
                y_sca = y_abs./(1+var0./pvar);
                phat_sca = phat_abs./(1+pvar./var0);
                zhat = (phat_sca + y_sca.*I1overI0).*sign(phat);
                zvar = y_sca.^2 + phat_sca.^2 ...
                    + (1+B.*I1overI0)./(1./var0+1./pvar) ...
                    - abs(zhat).^2;

                if obj.autoTune
                end

            end
        end
       
        % Bethe tuning of var0
        function var0 = tune(obj, phatfix, pvar)

            y_abs = abs(obj.mean0);
            phat_abs = abs(phatfix);
            var0 = obj.var0;
           
            step = 1.0; %[dflt=1.0]
            maxIt = 1; %[dflt=1]
            approx = true; %[dflt=true]
            for i=1:maxIt,
                if approx,
                    dI0 = 0.5./(var0+pvar); % good approximation for small var0+pvar 
                else
                    dv = 0.01*var0; 
                    dI0 = (log(besseli(0,2*y_abs.*phat_abs./(var0+dv+pvar),1)) ...
                                   -log(besseli(0,2*y_abs.*phat_abs./(var0+pvar),1)))./dv;
                end
                tmp = 1./(1+pvar./var0);
                var0full = ones(size(y_abs,1),1)*sum( ((y_abs-phat_abs).*tmp).^2 ...
                            + (var0.^2).*dI0 ,1) ./ sum( tmp ,1);
                var0 = (1-step)*obj.var0 + step*var0full;
            end
            obj.var0 = var0;

        end


        % Compute log likelihood
        %   E( log p_{Y|Z}(y|z) )
        function ll = logLike(obj,Axhat,pvar)
           
            %Use noise variance, or tiny value if variance is zero
            var1 = obj.var0;                    % noise var
            var1(var1 == 0) = 1e-8;
         
            y_abs = abs(obj.mean0);             % |y|
            Axhat_abs = abs(Axhat);             % |Axhat|
            B = 2*y_abs.*Axhat_abs./var1;
            if (obj.maxSumVal)
                ll = - ((y_abs-Axhat_abs).^2 )./var1 ...
                    + log(besseli(0,B,1));
            else
                %Compute upper bound on (offset) log likelihood
                ll = log(y_abs./var1) ...
                    - ((y_abs-Axhat_abs).^2 + pvar)./var1 ...
                    + log(besseli(0,B,1));
            end
        end
      
        % Compute output cost:
        % For sum-product compute
        %   abs(Axhat-phatfix)^2/(pvar) + log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat)
                   
            y_abs = abs(obj.mean0);

            % Compute output cost
            if ~(obj.maxSumVal)

                % Find the fixed-point of phat
                opt.phat0 = Axhat; % SEEMS THAT Axhat WORKS BETTER than phat
                opt.alg = 1; % approximate newton's method
                opt.maxIter = 25; % [dflt=25] 
                opt.tol = 1e-6; % [dflt=1e-6]
                opt.stepsize = 0.95; % [dftl=.95]
                opt.regularization = 0;  
                opt.debug = false;
                phatfix = estimInvert(obj,Axhat,pvar,opt);

                % update noise variance (FIX THIS AFTER gampEst SUPPORTS OUTPUT TUNING!)
                if obj.autoTune, obj.tune(phatfix,pvar); end

                % Compute log int_z p_{Y|Z}(y|z) CN(z;phatfix, pvar)
                phatfix_abs = abs(phatfix);
                if 0
                  ls1 = log(2*y_abs./(obj.var0 + pvar)) ...
                          - (y_abs.^2 + phatfix_abs.^2) ./ (obj.var0 + pvar);
                  ratio = 2*y_abs.*phatfix_abs./(obj.var0 + pvar);
                  ls2 = log(besseli(0,ratio));
                  big = find(ratio>100);
                  ls2(big) = ratio(big) - 0.5*log(2*pi*ratio(big)); % for numerical reasons
                  ls = ls1+ls2;
                else
                  ls = log(2*y_abs./(obj.var0 + pvar)) ...
                        - ((y_abs-phatfix_abs).^2) ./ (obj.var0 + pvar) ...
                        + log(besseli(0,2*y_abs.*phatfix_abs./(obj.var0 + pvar),1));
                end

                % Combine to form output cost
                ll = ls + abs(Axhat - phatfix).^2./pvar;

            else

                % Ensure variance is small positive number
                var1 = max(1e-20, obj.var0);
            
                % Output cost is simply the log likelihood
                Axhat_abs = abs(Axhat);
                %ll1 = log(2*y_abs./var1) ...
                %    - (y_abs.^2+Axhat_abs.^2)./var1;
                %ratio = 2*y_abs.*Axhat_abs./var1;
                %ll2 = log(besseli(0,ratio));
                %big = find(ratio>100);
                %ll2(big) = ratio(big) - 0.5*log(2*pi*ratio(big)); % for numerical reasons
                %ll = ll1 + ll2; 
                ll = log(2*y_abs./var1) ...
                        - ((y_abs-Axhat_abs).^2) ./ var1 ...
                        + log(besseli(0,2*y_abs.*Axhat_abs./var1,1));

            end 
            
        end

    end
    
end

