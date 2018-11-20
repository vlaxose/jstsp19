classdef ncCAwgnEstimIn < EstimIn
    % ncCAwgnEstimIn:  noncoherent CAwgn scalar input estimation function
    
    properties
        % Prior mean and variance
        mean0;  % Mean
        var0;   % Variance
    end
    
    methods
        % Constructor 
        function obj = ncCAwgnEstimIn(mean0, var0)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.mean0 = mean0;
                if nargin<2,
                    obj.var0 = 0.01*abs(mean0).^2;
                else
                    obj.var0 = var0;
                end;
            end
        end
        
        % Initialize using prior mean and variance 
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = zeros(size(obj.mean0));
            var0 = obj.var0 + abs(obj.mean0).^2;
            valInit = -inf;
        end
        
        % noncoherent Circular AWGN estimation function:
        % Provides the posterior mean and variance of a variable u
        % from a noncoherent observation v = exp(j*theta)*( u + w ), 
	% where u = CN(umean0,uvar0), w = CN(0,wvar), theta = U[0,2pi]
        %
        % Also compute the negative KL divergence
        % klDivNeg = \int p(u|v)*\log( p(u) / p(u|v) )du
        function [umean, uvar, klDivNeg] = estim(obj, v, wvar)
            % Get prior
            umean0_abs = abs(obj.mean0);
            uvar0 = obj.var0;
	    v_abs = abs(v);
           
            % Compute posterior mean and variance of u 
	    B = 2*umean0_abs.*v_abs./(uvar0+wvar);
	    I1overI0 = min( B./(sqrt(B.^2+4)), ...
	    	B./(0.5+sqrt(B.^2+0.25)) );%upper bounds (11)&(16) from Amos
	    umean0_sca = umean0_abs./(1+uvar0./wvar);
	    v_sca = v_abs./(wvar./uvar0+1);
            umean = (v_sca + umean0_sca.*I1overI0).*sign(v);
            uvar = umean0_sca.^2 + v_sca.^2 ...
	    	+ (1+B.*I1overI0)./(1./uvar0+1./wvar) ...
		- abs(umean).^2;
            % Compute the negative KL divergence 
            if (nargout >= 3)
	      logC = - log(pi*(uvar0+wvar)) ...
	    		- ((umean0_abs-v_abs).^2)./(uvar0+wvar) ...
			+ log(besseli(0,B,1));
	      klDivNeg = logC + log(pi*wvar) + (uvar+abs(umean-v).^2)./wvar;
            end
        end

        % Generate random samples 
        function x = genRand(obj, nx)
            x = exp(1j*2*pi*rand(nx)).*(obj.mean0 +...
                sqrt(obj.var0/2)*(complex(randn(nx),randn(nx))));
        end
        
        % Computes the likelihood p(y) for y = x*exp(j*theta) + v, 
	% for v = CN(0,yvar), x=CN(obj.mean0,obj.var0), theta=U[0,2pi]
        % note: typically used with y=rhat and yvar=mur
        function py = plikey(obj,y,yvar)
	    B = 2*abs(obj.mean0).*abs(y)./(obj.var0+yvar);
	    py = besseli(0,B,1) ...
	    	.*exp(-((abs(obj.mean0)-abs(y)).^2)./(obj.var0+yvar)) ...
		./(pi*(obj.var0+yvar));
        end
       
        % Computes log version of likelihood (see above)
        function logpy = loglikey(obj,y,yvar)
	    B = 2*abs(obj.mean0).*abs(y)./(obj.var0+yvar);
	    logpy = log(besseli(0,B,1)) ...
		- ((abs(obj.mean0)-abs(y)).^2)./(obj.var0+yvar) ...
		- log(pi*(obj.var0+yvar));
        end
    end
    
end

