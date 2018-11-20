classdef UnifEstimIn < EstimIn
    %UnifEstimIn:  Uniform scalar input estimation function (MAP or MMSE )
    
    properties 
        %endpoints for which p(x) = const
        a    % minimum  
        b;   % maximum
        maxSumVal = true;
    end
    
    methods
        % Constructor
        function obj = UnifEstimIn(a,b,maxSumVal)
            obj = obj@EstimIn;
            if nargin >= 2,
                obj.a = min(a,b);
                obj.b = max(a,b);
            end
            if nargin >= 3
                obj.maxSumVal=maxSumVal;
            end
        end

        % endpoints of uniform prior
        function [umean0, uvar0, valInit] = estimInit(obj)
            if obj.a == -inf || obj.b == inf
                umean0 = max(min(0,obj.b),obj.a);
                uvar0 = 1;
            else
                umean0 = (obj.a +obj.b)/2;
                uvar0 = (obj.b-obj.a)/12;
            end
            valInit = 0;
        end

        % Uniform estimation function
        function [umean, uvar, val] = estim(obj, Rhat, Rvar)
            
            Rhat = real(Rhat);
            if obj.maxSumVal
                % Compute MAP mean and variance parameter
                umean = Rhat;
                umean(Rhat <= obj.a) = obj.a;
                umean(Rhat >= obj.b) = obj.b;
                uvar = Rvar;
                uvar(umean == obj.a | umean == obj.b) = 0;
                
                if (nargout >= 3)                            
                    val = 1./(obj.b-obj.a).*ones(size(Rhat));
                end
            else

                % Compute MMSE mean and variance
                alpha = (obj.a-Rhat)./sqrt(Rvar);
                beta = (obj.b-Rhat)./sqrt(Rvar);
                a_phi = normpdf( alpha);
                b_phi = normpdf( beta);
                Z = normcdf( beta)-normcdf( alpha);
                umean = Rhat + (a_phi-b_phi) ./ Z .* sqrt(Rvar);
                umean = max(obj.a,min(obj.b,umean));

                uvar = Rvar .* ( 1 + ...
                            (alpha.*a_phi - beta.*b_phi) ./ Z - ...
                            ( (a_phi - b_phi) ./ Z ).^2 ...
                            );

                % Numerical errors happen when the intersection of the prior and the Gaussian pdf becomes very small.
                % ( Exceedingly low probability == gonna happen )
                % To mitigate the problems when the divisor of umean approaches zero, 
                % use the uninformed prior estimates if a<rhat<b, otherwise the closest limit
                invalid = Z < 1e-13;% the nearest of a,b is 7+ sigmas from rhat
                umean(invalid & (Rhat<obj.a)) = (obj.a +obj.b)/2;
                umean(invalid & (Rhat<=obj.a)) = obj.a;
                umean(invalid & (Rhat>=obj.b)) = obj.b;
                diff = umean - Rhat;
                if (nargout >= 3)
                    Dkl1 = log( obj.b - obj.a ) + .5*log(2*pi*Rvar);
                    Dkl2 = log( Z ) + .5*( uvar + (umean - Rhat).^2)./Rvar;
                    val = Dkl1+Dkl2;
                    if length(Rvar)>1
                        val(invalid) = Dkl1(invalid) -2; % the -2 offset is for tail integral Z=1e-13 
                    else
                        val(invalid) = Dkl1 -2;
                    end
                end
                uvar(invalid) = 0; % fell off the cliff
            end
        end
        
        % Generate random samples
        function x = genRand(obj, outSize)
            if isscalar(outSize)
                x = obj.a + (obj.b-obj.a).*rand(outSize,1);
            else
                x = obj.a + (obj.b-obj.a).*rand(outSize);
            end
        end
        
        % Computes the likelihood p(y) for real(y) = x + v, v = N(0,yvar)
        function py = plikey(obj,Y,Yvar)
            alpha = (obj.a - Y)./sqrt(Yvar);
            beta = (obj.b - Y)./sqrt(Yvar);
            py = (erf(beta./sqrt(2))/2 - erf(alpha./sqrt(2))/2)./(obj.b - obj.a);
        end
        
    end
    
end

