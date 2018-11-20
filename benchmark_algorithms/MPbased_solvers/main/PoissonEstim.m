classdef PoissonEstim < handle
    % PoissonEstim:  Poisson estimation function
    properties
        cnt;         % Observed Poisson count
        scale = [];  % Scale factor in the Poisson intensity
    end        
    
    methods
        % Constructor
        function obj = PoissonEstim(cnt, scale)
            obj.cnt     = cnt;
            obj.scale   = scale;            
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
        %   cnt(i) = poissrnd( scale(i)*z(i) )
        function [zmean, zvar] = estim(obj, zmean0, zvar0)
     
            % Get dimensions and initialize vectors
            nz = length(zmean0);
            zmean = zeros(nz,1);
            zvar = zeros(nz,1);
     
            % Find MAP estimate
            % The log prob is 
            %   log p(z|cnt) = cnt*log(scale*z) - scale*z - (z-zmean0)^2/(2*zvar0)
            % So, the MAP estimate is:
            %
            %   zvar0*cnt/z - zvar0*scale - (z-zmean0) = 0
            %   z^2 +z*(-zmean0+zvar0*scale) - zvar0*cnt = 0
            b = -zmean0 + zvar0.*obj.scale;
            c = -zvar0.*obj.cnt;
            zmap = (-b + sqrt(b.^2 - 4*c) )/2;
            
            % Compute range over which 
            zmin = max(1e-3, zmap-4*sqrt(zvar0));
            zmax = zmap+4*sqrt(zvar0);
               
            for iz = 1:nz
                
                % Compute log posterior on a discretized space
                z = linspace(zmin(iz), zmax(iz), 1000)';
                logpz = obj.cnt(iz)*log(obj.scale(iz)*z) - obj.scale(iz)*z ...
                    - (z-zmean0(iz)).^2/(2*zvar0(iz));
                logpz = logpz - max(logpz);
                pz = exp(logpz);
                pz = pz / sum(pz);
                
                % Compute mean and variance
                zmean(iz) = pz'*z;
                zvar(iz) = pz'*((z-zmean(iz)).^2);                                
            end
            
            if (any(isnan(zvar)))
                error('Undefined input value');
            end
        end
        
         % Compute log likelihood
        %   E( log p_{Z|Y}(z|y) )
        function ll = logLike(obj,zhat,zvar) 
            
            % Compute range of z values to test
            nz = length(zhat);
            zmin = max(1e-3, zhat-4*sqrt(zvar));
            zmax = zhat+4*sqrt(zvar);            
            ll = 0;
               
            for iz = 1:nz
                
                % Compute log posterior on a discretized space
                z = linspace(zmin(iz), zmax(iz), 1000)';
                logLikei = obj.cnt(iz)*log(obj.scale(iz)*z) - obj.scale(iz)*z;

                pz = exp(-(z-zhat(iz)).^2/(2*zvar(iz)));
                ll = ll + pz'*logLikei / sum(pz);
                
            end
        end
    end
end