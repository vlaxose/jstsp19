classdef DisCScaEstim < EstimIn
% DisCScaEstim:  Complex-valued scalar estimator based on a discrete distribution
    
    properties 
        dist;   % discrete distribution
    end
    
    methods
        % Constructor
        function obj = DisCScaEstim(x0, px0)
            obj = obj@EstimIn;
            obj.dist = DisDist(x0, px0);         
        end
        
        % Compute prior mean and variance
        function [uhat, uvar, valInit] = estimInit(obj)
            x0 = obj.dist.x;
            px0 = obj.dist.px;
            uhat = x0.'*px0;
            uvar = (abs(x0-uhat).^2)'*px0;
            valInit = 0;
        end              
        
        % AWGN estimation.  
        % Produces an estimate of a variable u with discrete
        % distribution (x0,px0) from an observation v of the form
        %   v = u + w, w = CN(0,wvar)
        %
        % Also computes the K-L divergence between p(U|V) and p(U)
        function [uhat, uvar, klDivNeg] = estim(obj, v, wvar)
            nr = length(v);
            x0 = obj.dist.x;
            px0 = max(obj.dist.px,1e-20);
	    na = length(x0);
            %x = repmat(x0, 1, nr) - repmat(v.', na,1);
            x = x0*ones(1,nr) - ones(na,1)*(v.');
            %px = repmat(px0, 1, nr);
            px = px0*ones(1,nr);
            logpx = log(px0)*ones(1,nr);
            %wvar2 = repmat(wvar', na,1);
            wvar2 = ones(na,1)*(wvar');
            logpxr = logpx - abs(x).^2./(wvar2);
            %logpxr = logpxr - repmat(max(logpxr) , na ,1);
            logpxr = logpxr - ones(na,1)*max(logpxr); % for numerical reasons
            pxr = exp(logpxr);
            %pxr = pxr./repmat(sum(pxr),na,1);
            pxr = pxr./(ones(na,1)*sum(pxr));
            uhat = x0.'*pxr;
            %uvar = sum((abs(repmat(x0, 1, nr) - repmat(uhat, na,1)).^2).*pxr);
            uvar = sum((abs(x0*ones(1,nr) - ones(na,1)*uhat).^2).*pxr);
            uhat = uhat.';
            uvar = uvar.';
            
            % Compute the negative KL divergence
            if (nargout >= 3) 
                %klDivNeg = sum(pxr.*log(repmat(px0,1,nr)./max(pxr,1e-8))).';
                klDivNeg = sum(pxr.*( logpx - log(max(pxr,1e-20)) )).';
            end
            % The code above can be confusing but it basically does what is done below
            % but without the for loop. This is why matrices are replicated.
            %  for vlen = 1:nr
            %      logpxr = log(px) - (v(vlen) - x).^2./(2 * wvar(vlen));
            %      logpxr = logpxr - max(logpxr);
            %      pxr = exp(logpxr);
            %      pxr = pxr./sum(pxr);
            %      uhat(vlen) = obj.x0'*pxr;
            %      uvar(vlen) = ((obj.x0-uhat(vlen)).^2)'*pxr;
            %  end
        end
        
        % Generates n random samples from the distribution
        function x = genRand(obj, nx)
            x = obj.dist.genRand(nx);
        end
        
        % Computes the likelihood p(y) for y = x + v, v = N(0,yvar)
        function py = plikey(obj,y,yvar)
            x0 = obj.dist.x;
            px0 = obj.dist.px;
            nx0 = length(x0);
            ny = length(y);
            %dy = abs(repmat(y,1,nx0) - repmat(x0.',ny,1)).^2;
            dy = abs(y*ones(1,nx0) - ones(ny,1)*(x0.')).^2;
            %py = exp(-dy./repmat(yvar,1,nx0))*px0;
            py = exp(-dy./yvar*ones(1,nx0))*px0;
            py = 1./sqrt(pi*yvar).*py;
            
        end
        
        % Computes the log-likelihood, log p(y), for y = x + v, 
        % p(v) = N(0,yvar)
        function logpy = loglikey(obj, y, yvar)
            logpy = log(obj.plikey(y, yvar));
        end
        
        % Get the points in the distribution
        function x0 = getPoints(obj)
            x0 = obj.dist.x;
        end
            
    end
end

