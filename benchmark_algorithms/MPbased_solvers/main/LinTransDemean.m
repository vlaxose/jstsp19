classdef LinTransDemean < LinTrans
    % Linear operator class with mean removal
    %
    % Given an m x n linear operator A, LinTransDemean creates a 
    % "demeaned" operator (m+1) x (n+1) Ad as follows:  Define
    %
    %   Amean(i) = sqrt(n)*mean of i-th row of A
    %   delA(i,j) = A(i,j) - (1/sqrt(n))*Amean(i)
    %
    % Then
    %
    %   Ad = [delA                 Amean; 
    %         1/sqrt(n)*ones(1,n)  -1     ]
    %
    % With this definition, it can be verified that if 
    %
    %   xd = [x-xhat0; v], v = sum(x-xhat0)/sqrt(n)
    % 
    % then
    %
    %   Ad*xd = [z-zhat0; 0]  zhat0 = A*xhat0
    %
    properties
        % Base linear operator
        A;
        
        % Mean per row and square of mean
        Amean;
        Amean2;
        
        % Mean of input and output
        xhat0;
        zhat0;
    end
    
    methods 
        % Constructor
        function obj = LinTransDemean(A,xhat0)
            
            % Base class
            obj = obj@LinTrans();
            
            % Get mean and square of mean
            [m,n] = A.size();
            obj.A = A;
            obj.Amean = (1/sqrt(n))*A.mult(ones(n,1)); 
            obj.Amean2 = abs(obj.Amean).^2;
            
            % Set mean of input and output
            obj.xhat0 = xhat0;
            obj.zhat0 = A.mult(xhat0);
            
        end       

        % Size
        function [md,nd] = size(obj)
            [m,n] = obj.A.size();
            md = m+1;
            nd = n+1;
        end

        % Matrix multiply:  zd = Ad*xd + [zhat0; 0]
        function zd = mult(obj,xd)
            % Separate regular and additional components
            [m,n] = obj.A.size();
            x1 = xd(1:n,:);
            x2 = xd(n+1,:);
            
            % Compute demean output
            z1 = obj.A.mult(x1) + ...
                obj.Amean*(-sum(x1,1)/sqrt(n) + x2) + obj.zhat0;
            z2 = sum(x1,1)/sqrt(n)-x2;
            zd = [z1; z2];
        end
        
        % Multiply by square:  pvar = abs(Ad).^2*xvard
        function pvar = multSq(obj, xvar)
            
            % Separate regular and additional components
            [m,n] = obj.A.size();
            xvar1 = xvar(1:n,:);
            xvar2 = xvar(n+1,:);
            S = size(xvar1,2);
            
            % Compute updates based on z = A1*x1 + zoff = z1 + zoff
            pvar1 = obj.A.multSq(xvar1)  - ...
                (2/sqrt(n))*real((conj(obj.Amean)*ones(1,S)).*(obj.A.mult(xvar1)))...
                + obj.Amean2*(mean(xvar1,1) + xvar2);
            pvar2 = mean(xvar1,1) + xvar2;
            pvar = [pvar1; pvar2];        
    
        end
        

        % Matrix multiply transpose:  x = A'*s
        function x = multTr(obj,s)
            % Separate regular and additional components
            [m,n] = obj.A.size();
            s1 = s(1:m,:);
            s2 = s(m+1,:);
            
            % Perform transpose multiply
            x1 = obj.A.multTr(s1) + ones(n,1)*(s2 - obj.Amean'*s1)/sqrt(n);
            x2 = obj.Amean'*s1 - s2;
            x = [x1; x2];
        end
                    
 
        % Matrix multiply with componentwise square transpose:  
        %   rvari = (Ad.^2)'*svar
        function rvari = multSqTr(obj,svar)
            % Separate regular and demean components
            [m,n] = obj.A.size();
            svar1 = svar(1:m,:);
            svar2 = svar(m+1,:);
            S = size(svar1,2);
        
            % Compute multiplication transpose
            rvari1 = obj.A.multSqTr(svar1) -(2/sqrt(n))*...
                real(obj.A.multTr((obj.Amean*ones(1,S)).*svar1)) + ... 
                ones(n,1)*(sum((obj.Amean2*ones(1,S)).*svar1,1)/n + 1/n*svar2);
            rvari2 = sum((obj.Amean2*ones(1,S)).*svar1,1) + svar2;
            rvari = [rvari1; rvari2];
        end
        
        % Computes the mean and variance from the difference 
        function [xhat1, rhat1, rvar1] = getEst(obj,xhat,rhat,rvar)
            [m,n] = obj.A.size();
            alpha = (-sum(xhat(1:n,:),1)/sqrt(n) + xhat(n+1,:))/sqrt(n);
            xhat1 = xhat(1:n,:)+obj.xhat0 + alpha;
            rhat1 = rhat(1:n,:)+obj.xhat0 + alpha;
            rvar1 = rvar(1:n,:);
        end

    end
    

end
