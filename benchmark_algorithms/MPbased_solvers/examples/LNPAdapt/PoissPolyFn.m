classdef PoissPolyFn < handle
    % PoissPolyFn:  Optimization functions for Poisson polynomial fit
    %
    % Given a vectors y, zmean, and zvar, the class defines methods for
    % computing the objective function
    %
    %   fobj(lam) = sum ( log( p(y(i)|lam) ) )
    %
    % where
    %
    %   p(y|lam) = E( P(y| v(lam,z) )
    %
    %   y ~ Poisson of rate v(lam,z) = exp( polyval(lam,u))
    %   u = 1/(1+exp(-z))
    %   z ~ N(zmean,zvar)        
    properties (Access = private)
        y;              % measured Poisson counts
        zmean, zvar;    % mean and variance of z
        npoly;          % num coeffs in polynomial fit
        
        % Parameters for numerical integration over z
        % Integration is performed over nzint points from +/- zintmax std devs
        zintmax = 2;
        nzint = 10;
        
        % Use logarithmic objective function 
        logObj = false;
       
        % Pre-computed values for z for numerical integration as well as
        % the associated values for u, y and the const terms in the
        % probabilities
        pz;
        z;
        u;
        ymat;
        pconst;
               
        % Pre-computed regression matrix
        U;
        
    end
    
    methods
        
        % Constructor
        function obj = PoissPolyFn(y,zmean,zvar,npoly,nzint)
            
            % Store the measurements
            obj.y = y;
            obj.npoly = npoly;
            ny = length(y);
            if (length(zmean) == 1)
                zmean = repmat(zmean,ny,1);
            end
            if (length(zvar) == 1)
                zvar = repmat(zvar,ny,1);
            end
            obj.zmean = zmean;
            obj.zvar = zvar;
            if (nargin >= 5)
                obj.nzint = nzint;
                obj.zintmax = sqrt(2*log(nzint/2));
            end
            
            % Generate points for numerical integration and compute the probability
            % of each point
            w = linspace(-obj.zintmax,obj.zintmax,obj.nzint)';
            pz1 = exp(-w.^2/2);
            pz1 = pz1/sum(pz1);
            z1 = repmat(zmean,1,obj.nzint) + sqrt(zvar)*w';
            u1 = 1./(1+exp(-z1));
            obj.pz = pz1;
            obj.z = z1;
            obj.u = u1;
            
            % Create regression matrix
            nztot = ny*obj.nzint;
            uvec = u1(:);
            U1 = ones(nztot,npoly);
            for ip = npoly-1:-1:1
                U1(:,ip) = U1(:,ip+1).*uvec;
            end
            obj.U = U1;
            
            % Pre-computed matrices for objective function
            obj.ymat = repmat(y,1,obj.nzint);
            obj.pconst = -y  + y.*log(max(1,y));
            obj.pconst = repmat(obj.pconst,1,obj.nzint);
            
        end
        
        % Accessor variables
        function setLogObj(obj, val)
            obj.logObj = val;
        end
                
        % Fn eval and gradient
        function [fobj, fgrad, logpyMean] = optFn(obj,lam)
            
            % Compute the rate at each point
            logv = polyval(lam, obj.u);
            v = exp(logv);
            
            % Compute p(y|lam) and objective function
            logpy = -v + obj.ymat.*logv - obj.pconst;
            if (obj.logObj)
                fobj = mean( logpy*obj.pz );
            else
                pymat = exp(logpy);
                py = pymat*obj.pz;
                logpyMean = log(py);
                fobj = mean( logpyMean );
            end
            
            % If requested, compute gradient
            if (nargout >= 2)
                ny = length(obj.y);                
                if (obj.logObj)
                    dpydr = repmat(obj.pz',ny,1).*(obj.ymat-v);
                else
                    g = (1./py)*obj.pz';
                    dpydr = g.*pymat.*(obj.ymat -v);
                end
                
                fgrad = (1/ny)*sum(repmat(dpydr(:),1,obj.npoly).*obj.U)';
            end
            
        end
        
        
        % Compute initial value of lambda based on fitting the CDF of Y.
        % This should only be used when obj.zmean is zero, otherwise there
        % are better ways to fit the data.
        function lamInit = initFit(obj)
            ysort = sort(obj.y);
            nz = length(obj.y);
            ycdf = (1:nz)'/(nz+1);
            zmean0 = mean(obj.zmean);
            zvar0 = mean( (obj.zmean-zmean0).^2 ) + mean(obj.zvar);
            lamInit = polyFitCdf(zmean0, zvar0, obj.npoly, ycdf, ysort);

        end
    end
    
end

