classdef IntEstimInAvg < handle
    % IntEstimInAvg:  Input SE updates via numerical integration
    
    properties (Access=private)
        inEst;      % input estim
        x;          % input points
        px;         % input probability
        w;          % noise points
        pw;         % noise probability
    end
    
    methods 
        % Constructor
        function obj = IntEstimInAvg(inEst, x, px, nw)                       
            obj.inEst = inEst;  
            obj.x = x;
            obj.px = px;
            wmax = sqrt(2*log(nw/2));
            obj.w = linspace(-wmax, wmax, nw)';
            obj.pw = exp(-(obj.w).^2/2);
            obj.pw = obj.pw/sum(obj.pw);
        end
        
        
        % Initial covariance for state evolution
        function [xcov0, taux0] = seInit(obj)
            [~,taux0] = obj.inEst.estimInit();
            xmean0 = obj.px'*obj.x;
            xvar0 = obj.px'*((obj.x-xmean0).^2);
            xcov0 = [xvar0 0; 0 0];
        end
        
        % Compute update for SE input  
        %
        %   xcov = E(x xhat)*(x xhat)'
        %   taux = taur*E(dgin/dr)
        function [xcov,taux] = seIn(obj, xir, alphar, taur)
            
            % Get dimensions
            nx = length(obj.x);
            nw = length(obj.w);            
            v = sqrt(xir)*obj.w;
            taurvec = repmat(taur,nw,1);
            
            % Compute average 
            taux = 0;
            xcov = zeros(2);
            for ix = 1:nx
                r = alphar*obj.x(ix) + v;
                [xhati,xvari] = obj.inEst.estim(r,taurvec); 
                
                % Update covariance
                xcov1 = obj.x(ix)^2;
                xcov2 = obj.x(ix)*(xhati'*obj.pw);
                xcov3 = (abs(xhati.^2)'*obj.pw);
                xcov = xcov + obj.px(ix)*[xcov1 xcov2; conj(xcov2) xcov3];
                
                % Update mean output variance
                taux = taux + obj.px(ix)*xvari'*obj.pw;
            end
                        
        end
    end
    
end

