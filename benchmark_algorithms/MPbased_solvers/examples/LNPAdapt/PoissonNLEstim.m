classdef PoissonNLEstim < EstimOut
    % PoissonEstim:  Poisson estimation function with nonlinear rate
    % function.
    %
    % The output is assumed to be of the form
    %   cnt = poissrnd(v),
    %   v = exp( polyval(lam, u))
    %   u = 1./(1+exp(-z))
    properties (Access=private)
        cnt;         % Observed Poisson count
        lam = [];    % Polynomial coefficients.  
                     % Empty indicates that an initial estimate is required
        npoly = 3;      % number of polynomial coefficients
        nzint = 100;    % number of integration points for estim fn       

        % Oracle values for debugging.  These do not need to be supplied
        lamt = [];   % True lambda
        zt = [];     % True z        
        
        % Adaptation options
        adapt = false;  % enable adaptation
        adaptIter = 1;  % adaptation iteration cnt
        verbose = 0;    % 0=no print, 1=print per adapt, 2=print per iter
        nit = 20;           % num gradient ascent iterations per iteration
        nzintAdapt = 10;    % num of integration points over z for fn eval
        stopTol = 1e-4;     % stopping tolerance
        
        % Oracle assisted adaptation (for debugging only)
        % 0 = no oracle
        % 1 = use true variance in adaptation
        % 2 = use true value for z in adaptation
        oracleAdaptLev = 0;       
        
        % Parameter string
        paramStr = {'lam', 'adapt', 'verbose', 'nit', 'nzint', ...
            'npoly', 'nzintAdapt', 'oracleAdaptLev', 'stopTol' };
        
    end
    
    methods
        % Constructor
        function obj = PoissonNLEstim(cnt, zt, lamt)
            obj         = obj@EstimOut;
            obj.cnt     = cnt;            
            if (nargin >= 2)
                obj.lamt = lamt;    % True lambda
                obj.zt = zt;        % True z
            end
        end
        
        % Set parameters
        % The format of the function is
        %   obj.setParam('paramName1', val1, 'paramName2', val2, ...)
        % The parameter names should be members of the class obj.paramStr.
        function setParam(obj, varargin)
            
            nparam = length(varargin)/2;
            for iparam = 1:nparam
                % Get field name and value
                field = varargin{2*iparam-1};
                val = varargin{2*iparam};
                
                % Check if field is valid
                isvalid = false;
                for ip=1:length(obj.paramStr)
                    if (strcmp(field, obj.paramStr{ip}))
                        isvalid = true;
                        break;
                    end
                end
                if (~isvalid)
                    error(['Invalid parameter ', field]);
                end
                
                % Run command
                cmd = sprintf('obj.%s = val;', field);
                eval(cmd);            
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
        %   cnt(i) = poissrnd( f(z(i)) )
        function [zmean, zvar] = estim(obj, zmean0, zvar0)
            
            % If lambda is not known, compute an initial fit
            if (isempty(obj.lam))
                obj.initFit(zmean0,zvar0)
            end
            
            % Adapt the parameters
            if (obj.adapt)
                if (obj.oracleAdaptLev == 0) 
                    % Adaptation based on empirical values for z
                    obj.adaptLam(zmean0,zvar0);
                elseif (obj.oracleAdaptLev == 1) 
                    % True variance
                    zvart = mean((obj.zt-zmean0).^2);                    
                    obj.adaptLam(zmean0,zvart);
                else 
                    % True value for z
                    obj.adaptLam(obj.zt, 0);
                end
            end
            
            % Check dimensions
            ny = length(obj.cnt);
            if (length(zmean0) == 1)
                zmean0 = repmat(zmean0, ny, 1);
            end
            if (length(zvar0) == 1)
                zvar0 = repmat(zvar0, ny, 1);
            end
            
            % Get dimensions and initialize vectors
            nz = length(zmean0);
            zmean = zeros(nz,1);
            zvar = zeros(nz,1);
            
            % Integration points           
            wmax = sqrt(2*log(obj.nzint/2));
            w = linspace(-wmax,wmax,obj.nzint)';
            logpz0 = -w.^2/2;
            
            for iz = 1:nz
                
                % Compute rate for the points
                z = zmean0(iz) + sqrt(zvar0(iz))*w;
                u = 1./(1+exp(-z));
                logv = polyval(obj.lam, u);
                v = exp(logv);
                
                % Compute log posterior on a discretized space
                logpz = obj.cnt(iz)*logv - v + logpz0;
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
            
            % Check dimensions
            ny = length(obj.cnt);
            if (length(zhat) == 1)
                zhat = repmat(zhat, ny, 1);
            end
            if (length(zvar) == 1)
                zvar = repmat(zvar, ny, 1);
            end
                        
            % Integration points
            obj.nzint = 100; % number of integration points
            wmax = sqrt(2*log(obj.nzint/2));
            w = linspace(-wmax,wmax,obj.nzint)';
            pz0 = exp(-w.^2/2);
            pz0 = pz0 / sum(pz0);
            
            % Compute range of z values to test
            nz = length(zhat);
            ll = 0;
            
            for iz = 1:nz
                
                % Compute rate at each of the points
                z = zhat(iz) + sqrt(zvar(iz))*w;
                u = 1./(1+exp(-z));
                logv = polyval(obj.lam, u);
                v = exp(logv);
                
                % Accumulate log likelihood
                logLikei = obj.cnt(iz)*logv - v;
                ll = ll + pz0'*logLikei;
                
            end
        end
        
        % Compute an initial fit for the parameters
        function initFit(obj,zmean0,zvar0)
            
            % Create polynomial fit function
            fn = PoissPolyFn(obj.cnt, zmean0, zvar0, ...
                obj.npoly, obj.nzintAdapt);
            
            % Initial fit based on CDF.  
            obj.lam = fn.initFit();
        end
        
        % Adapation of lambda
        % Sets a new value of lambda based on gradient ascent on the 
        % likelihood
        function adaptLam(obj,zmean0,zvar0)                      
                                   
            % Optimization parameters
            step = 1e-2;
            alpha = 0.5;
            beta = 2;
                        
            % Get initial lambda
            lam0 = obj.lam;
            
            % Create optimization function
            fn = PoissPolyFn(obj.cnt, zmean0, zvar0, obj.npoly, ...
                obj.nzintAdapt);  
                     
            % Get initial fn and gradient
            [fobj0,fgrad0] = fn.optFn(lam0);
            dx = fgrad0;
            fobjHist = zeros(obj.nit,1);
            
            done = false;
            it = 1;
            while ~done
                
                % Get test point
                dlam = step*dx;
                lam1 = lam0 + dlam;
                
                % Evalutate test point
                [fobj1,fgrad1] = fn.optFn(lam1);
                
                % See if point passes
                pass = (fobj1 >= fobj0 + alpha*fgrad0'*dlam);
                if (pass)
                    
                    bk = (fgrad1'*fgrad1)/(fgrad0'*fgrad0 + 1e-10);
                    dx = bk*dx + fgrad1;
                    
                    % Check for stopping condition
                    if (mean(abs(lam1-lam0) < obj.stopTol))
                        done = true;
                    end
                   
                    str = 'pass';
                    lam0 = lam1;
                    fobj0 = fobj1;
                    fgrad0 = fgrad1;
                    step = 8*beta*step;
                    
                else
                    str = 'fail';
                    step = step/beta;
                end
                
                % Print progress
                if (obj.verbose >= 2)
                    fprintf(1,'it=%d,%d fobj=%12.4e step=%12.4e %s\n', ...
                        obj.adaptIter, it, fobj0, step, str);
                end
                fobjHist(it) = fobj0;
                it = it+1;
                if (it > obj.nit)
                    done = true;
                end
                
            end
              
            if (obj.verbose >= 1)
                fprintf(1,'it=%d %d iters, norm(fgrad)=%12.4e\n', obj.adaptIter, ...
                    it, norm(fgrad0) );
            end
            
            % Increment adapatation
            obj.lam = lam0;
            obj.adaptIter = obj.adaptIter + 1;
            
        end              
        
    end
end