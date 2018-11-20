classdef IntEstimOutAvg < handle
    % IntEstimOutAvg:  Output SE averaging via numerical integration
    %
    % This is a base class where you must define two methods
    %   estim:  Output estimation
    %   getypts:  Finds points over which to integrate over y
    properties (Access = private)        
        % Number of points for numerical integration
        np = 100;
        nz = 100;
        ny = 100;        
    end
    
    methods (Abstract)
        % Estimation function given y, p and taup
        [zhat,zvar] = estim(obj,y,p,taup)
        
        % Points on which to integrate over y given z
        % pyzp(iy,iz) = P(y(iy)|z(ip))*const
        [y,pyzp] = getypts(obj,z,pz,ny)
        
        
    end
    
    methods
        % Constructor
        function obj = IntEstimOutAvg(np,ny,nz)          
            if (nargin >= 1)
                obj.np = np;
                obj.nz = nz;
                obj.ny = ny;                
            end
        end
        
        % Compute SE outputs
        function [taur,xir,alphar] = seOut(obj,zpcov, taup)
            
            % Get points to perform numerical integration
            [p,y,zmeanyp,pyp] = obj.getProb(zpcov);
            
            % Run estimator to compute zhat and zvar for each (y,p) point
            zhat = zeros(obj.ny,obj.np);
            zvar = zeros(obj.ny,obj.np);
            for ip = 1:obj.np
                [zhati,zvari] = obj.estim(y(:,ip),repmat(p(ip),obj.ny,1),...
                    repmat(taup,obj.ny,1));
                zhat(:,ip) = zhati;
                zvar(:,ip) = zvari;
            end
            
            % Compute gout and -dgout/dp
            gout = (zhat-repmat(p',obj.ny,1))/taup;
            goutdp = 1/taup*(1-zvar/taup);
            
            % Compute 
            %   taur = -1/E(dgout/dp)
            %   xir = (taur)^2*E(gout^2)
            taur = 1/sum(sum(pyp.*goutdp));
            xir = (taur^2)*sum(sum(pyp.*(gout.^2)));
            
            % Coampute 
            %   alphar = taur*E(dgout/dz)
            %
            % The derivative can be computed from Steing's lemma:           
            %   E(z*gout) = E(z^2)E(dgout/dz) + E(zp)*E(dgout/dp)
            goutz = sum(sum(pyp.*zmeanyp.*gout));
            alphar = (taur*goutz+zpcov(1,2))/zpcov(1,1);
                       
        end
        
        % Compute integration points and probabilities
        % Given covariance zpcov = cov(z,p), the method
        % finds points y and p along with probabilities
        %
        %   pyp(iy,ip) = Prob( y(iy,ip), p(ip))
        %   zmean(iy,ip) = E(z| y(iy,ip), p(ip))       %
        function [p,y,zmean,pyp] = getProb(obj,zpcov)
            
            % Get integration points for p
            pvar = zpcov(2,2);
            pmax = sqrt(2*log(obj.np/2));
            p = linspace(-pmax,pmax,obj.np)'*sqrt(pvar);
            if (pvar < 1e-6)
                pp = ones(obj.np,1);
            else
                pp =  exp(-p.^2/(2*pvar));
            end
            pp = pp / sum(pp);
            
            % Get mean and variance of z conditional on p
            if (zpcov(2,2) < 1e-6)
                zmean0 = zeros(obj.nz,1);
                zvar0 = zpcov(1,1);
            else
                zmean0 = zpcov(1,2)/zpcov(2,2)*p;
                zvar0 = zpcov(1,1)-abs(zpcov(1,2))^2/zpcov(2,2);
            end
            
            % For each point in p compute the integration points over z
            umax = sqrt(2*log(obj.nz/2));
            u = linspace(-umax,umax,obj.nz)';
            pz = exp(-u.^2/2);
            pz = pz/sum(pz);
            
            % Initialize vectors
            y = zeros(obj.ny,obj.np);
            pyp = zeros(obj.ny,obj.np);
            zmean = zeros(obj.ny,obj.np);
            for ip = 1:obj.np
                
                % Integration points over z
                z = zmean0(ip) + sqrt(zvar0)*u;
                
                % Integration points over y and the likelihood
                [yi,pyzp] = obj.getypts(z,pz,obj.ny);
                y(:,ip) = yi;
                
                % Compute p(y|z,p(ip))
                pypi = pyzp*pz;
                pypi = pypi / sum(pypi);
                
                % Compute p(y|p)
                pyp(:,ip) = pypi*pp(ip);
                
                % Compute E(z|y,p)
                den = pyzp*pz;
                den = den + 1e-6*max(den);
                zmean(:,ip) = pyzp*(z.*pz) ./ den;
                
            end
            
        end
        
    end
end