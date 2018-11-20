classdef NeuralOutEst < EstimOut
    % NeuralOutEst:  Output estimation function for neuron
    %
    % Based on non-linear Poisson model:
    %
    %   cnt = poissrnd(rate); 
    %   rate = log(1+exp(z))
    % where z is the linear filter output
    properties
        cnt;    % Observed Poisson count        
        
        % Points to test r on for numerical integration
        rpt;    % Discretization points for the rate
        zpt;    % Discretization points for the linear input z 
        
        % Pre-computed probability of y given z.
        % p(iz,iy) = P(Y=iy-1|Z = zpt(iz))
        logpyz;    
        
        % Noise variance
        noiseVar;
    end        
    
    methods
        % Constructor
        function obj = NeuralOutEst(cnt, zpt, noiseVar, ymax)
            obj = obj@EstimOut;
            obj.cnt = cnt;                                   
            obj.zpt = zpt;
            obj.noiseVar = noiseVar;
            
            % Rate r for each z
            obj.rpt = log(1+exp(obj.zpt));

            % Noise variance
            nw = 100;
            w = linspace(-3,3,nw)'*sqrt(noiseVar);            
            pw = exp(-w.^2/(2*noiseVar));
            pw = pw/sum(pw);
            
            % Compute logyfact = log(factorial(y))
            logyfact = zeros(1,ymax+1);
            for iy=1:ymax
                logyfact(iy+1)=logyfact(iy)+log(iy);
            end

            % Compute p(y|z)
            nz = length(zpt);
            obj.logpyz = zeros(nz,ymax+1);
            y = (0:ymax);
            for iz = 1:nz
                zw = obj.zpt(iz) + w;
                rate = log(1+exp(zw));
                logpyzi = log(rate)*y - repmat(rate,1,ymax+1) ...
                    - repmat(logyfact,nw,1);
                pyzi = pw'*exp(logpyzi);
                pyzi = pyzi/sum(pyzi);
                obj.logpyz(iz,:) = log(pyzi);                
            end            
        end
        
        % Simulate output and store value in cnt
        function cnt = simOut(obj,z)
            nz = length(z);
            v = randn(nz,1)*sqrt(obj.noiseVar);
            rate = z + v;
            cnt = log(1 + exp(rate));
            obj.cnt = cnt;
            
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
     
            % Get dimensions and initialize vectors
            nz = length(zmean0);
            zmean = zeros(nz,1);
            zvar = zeros(nz,1);                
               
            for iz = 1:nz
                
                % Compute log posterior on a discretized space
                logpz = obj.logpyz(:,obj.cnt(iz)+1) ...
                    -(obj.zpt-zmean0(iz)).^2/(2*zvar0(iz));
                logpz = logpz - max(logpz);
                pz = exp(logpz);
                pz = pz / sum(pz);
                
                % Compute mean and variance
                zmean(iz) = pz'*obj.zpt;
                zvar(iz) = pz'*((obj.zpt-zmean(iz)).^2);                                
            end
        end
        
        % Compute log likelihood
        %   E( log p_{Y|Z}}(y|z) )
        function ll = logLike(obj,zhat,zvar) 
            
            % Compute range of z values to test
            nz = length(zhat);
            ll = 0;              
            if (nargin < 3)
                % With no zvariance
                for iz = 1:nz
                    % Find closest point
                    [mm,im] = min(abs(zhat(iz)-obj.zpt));
                    ll = ll + obj.logpyz(im,obj.cnt(iz)+1);
                end
                
            else
                % With Gaussian variance
                for iz = 1:nz

                    % Compute log posterior on a discretized space
                    pz = exp(-(obj.zpt-zhat(iz)).^2/(2*zvar(iz)));
                    ll = ll + pz'*obj.logpyz(:,obj.cnt(iz)+1) / sum(pz);

                end
            end
        end

    end
end