classdef DisScaEstim < EstimIn
% DisScaEstim:  General scalar estimator based on a discrete distribution
    
    properties 
        dist;   % discrete distribution
    end
    
    methods
        % Constructor
        function obj = DisScaEstim(x0, px0)
            obj = obj@EstimIn;
            obj.dist = DisDist(x0, px0);         
        end
        
        % Compute prior mean and variance
        function [uhat, uvar, valInit] = estimInit(obj)
            x0 = obj.dist.x;
            px0 = obj.dist.px;
            uhat = x0'*px0;
            uvar = ((x0-uhat).^2)'*px0;
            valInit = 0;
        end              
        
        % AWGN estimation.  
        % Produces an estimate of a variable u with discrete
        % distribution (x0,px0) from an observation v of the form
        %   v = u + w, w = N(0,wvar)
        %
        % Also computes the K-L divergence between p(U|V) and p(U)
        function [uhat, uvar, klDivNeg] = estim(obj, v, wvar)
            x0 = obj.dist.x;
            px0 = obj.dist.px;
            x = repmat(x0, 1, length(v)) - repmat(v', length(x0),1);
            px = repmat(px0, 1, length(v));
            wvar2 = repmat(wvar', length(x0),1);
            logpxr = log(px) - x.^2./(2.*wvar2);
            logpxr = logpxr - repmat(max(logpxr) , length(x0) ,1);
            pxr = exp(logpxr);
            pxr = pxr./repmat(sum(pxr) , length(x0) ,1);
            uhat = x0'*pxr;
            uvar = sum(((repmat(x0, 1, length(v)) - repmat(uhat, length(x0),1)).^2).*pxr);
            uhat = uhat';
            uvar = uvar';
            
            % Compute the negative KL divergence
            if (nargout >= 3)            
                nr = length(v);
                klDivNeg = sum(pxr.*log(repmat(px0,1,nr)./max(pxr,1e-8))).';
            end
            % The code above can be confusing but it basically does what is done below
            % but without the for loop. This is why matrices are replicated.
            %  for vlen = 1:length(v)
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
            dy = abs(repmat(y,1,nx0) - repmat(x0',ny,1)).^2;
            py = exp(-dy./repmat(2*yvar,1,nx0))*px0;
            py = 1./sqrt(2*pi*yvar).*py;
            
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

