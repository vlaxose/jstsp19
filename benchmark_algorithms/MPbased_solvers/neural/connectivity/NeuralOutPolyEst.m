classdef NeuralOutPolyEst < EstimOut
    % NeuralOutEst:  Output estimation function for neuron
    %
    % Based on non-linear Poisson model:
    %
    %   cnt = poissrnd(rate);
    %   rate = log(1+exp(v))
    %   v = polyval(pcoeff, z + N(0,noisevar))
    % where z is the linear filter output
    properties
        cnt;        % Observed Poisson count
        pcoeff;     % Polynomial coeffs
        noiseVar;   % Noise variance
        
        % Use polynomial interpolation up to polyMax
        % A zero value indicates that there is no max
        polyMax = 0;    
        
        % Minimum rate
        rateMin = 1;
    end
    
    methods
        % Constructor
        function obj = NeuralOutPolyEst(cnt, noiseVar, pcoeff, polyMax, rateMin)
            obj = obj@EstimOut;
            obj.cnt = cnt;
            obj.pcoeff = pcoeff;
            obj.noiseVar = noiseVar;
            obj.polyMax = polyMax;
            if (nargin >= 4)
                obj.rateMin = rateMin;
            end
            
        end
        
        
        % AWGN estimation function
        % The method returns the posterior conditional mean
        % and variance of a random vector z given the Gaussian prior
        % on each component:
        %
        %   z(i) = N(zmean0(i), zvar0(i)), z(i) > 0
        %
        % and the Poisson observation
        %
        %   cnt(i) = poissrnd( r(i) )
        %   r(i) = 1/(1+exp(-z(i));
        function [zmean, zvar] = estim(obj, zmean0, zvar0)
            w = linspace(-4,4,100)';
            logpw = -w.^2/2;
            nz = length(zmean0);
            zmean = zeros(nz,1);
            zvar = zeros(nz,1);
            
            for iz = 1:nz
                % Compute rate
                z = zmean0(iz) + sqrt(zvar0(iz))*w;
                rate = obj.rateFn(z);
                
                % Compute posterior probability
                logpyz = obj.cnt(iz)*log(rate) - rate + logpw;
                logpyz = logpyz - max(logpyz);
                pzy = exp(logpyz);
                pzy = pzy / sum(pzy);
                
                % Compute posterior mean and variance
                zmean(iz) = pzy'*z;
                zvar(iz) = pzy'*(z-zmean(iz)).^2;
                
            end
            
        end
        
        % Compute log likelihood
        %   E( log p_{Y|Z}}(y|z) )
        function ll = logLike(obj,zhat,zvar)
            
            % Compute pdf of a unit Gaussian
            w = linspace(-4,4,100)';
            pw = exp(-w.^2/2);
            pw = pw / sum(pw);
            nz = length(zhat);
            
            if (nargin < 3)
                rate = obj.rateFn(zhat);
                ll = sum( obj.cnt.*log(rate) - rate );
            else
                                                
                % Compute log-likelihood
                ll = 0;
                for iz = 1:nz
                    % Compute rate
                    z = zhat(iz) + sqrt(zvar(iz))*w;
                    rate = obj.rateFn(z);

                    % Compute posterior probability
                    ll = ll + pw'*(obj.cnt(iz)*log(rate) - rate);
                    if (isnan(ll))
                        disp('NeuralOutPolyEst: NaN');
                        keyboard;
                    end
                end
            end
        end
        
        % Compute rate from linear output
        function rate = rateFn(obj, z)  
            if (obj.polyMax > 0)
                z = min(z, obj.polyMax);    
            end
            v = polyval(obj.pcoeff, z);
            rate = v;
            I = find(v < 10);
            rate(I) = max( log(1+exp(v(I))), obj.rateMin );
        end
        
    end
end