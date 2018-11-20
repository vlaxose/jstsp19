classdef GM2EstimIn < EstimIn
    % GM2EstimIn: Input estimator class for a real valued Gaussian mixture 
    % for use with GAMP in the case where X is an N-by-T signal matrix
    % whose elements are distributed according to Bernoulli/Gaussian-
    % mixtures. The third dimension of the parameters (except for lambda, 
    % which should be N-by-T) indexes the specific active Gaussian mixture
    % component.  Currently works for MMSE only
    
    properties
        
        %Mixture parameters
        LAMBDA;     % prior prob that x_n is active (N-by-T mtx)
        OMEGA;      % weights of the mixture components (N-by-T-by-D tnsr)
        THETA;      % means of mixture components (N-by-T-by-D tnsr)
        PHI;        % variances of the mixture components (N-by-T-by-D tnsr)

    end
    
    properties (Dependent = true, SetAccess = private)
        N;          % # of rows in signal matrix, X
        T;          % # of columns in signal matrix, X
        D;          % # of active mixture components
    end
    
    
    methods
        
        % Constructor
        function obj = GM2EstimIn(lambda,omega,theta,phi)
            
            if any(omega(:) < 0 | omega(:) > 1)
                error('GM2EstIn: Encountered invalid component weight')
            else
                obj.OMEGA = omega;
            end
            obj.THETA = theta;
            if any(phi(:) < 0)
                error('GM2EstIn: Encountered negative component variance')
            else
                obj.PHI = phi;
            end
            if any(lambda(:) < -1e-3 | lambda(:) > 1+1e-3)
                warning(sprintf(['GM2EstIn: Encountered invalid activity' ...
                    ' probability: max: %g, min:%g'], max(lambda(:)), ...
                    min(lambda(:))))
            end
            obj.LAMBDA = min(max(0, lambda), 1);
            
            %Force omegas to sum to one
            obj.OMEGA = obj.OMEGA ./ repmat(sum(obj.OMEGA, 3), [1, 1, obj.D]);
            obj.OMEGA(isnan(obj.OMEGA)) = 0;    % For genie version
        end
        
        % Prior mean and variance
        function [MEAN0, VAR0, valInit] = estimInit(obj)
            
            MEAN0 = obj.LAMBDA .* sum(obj.OMEGA .* obj.THETA, 3);
            VAR0  = obj.LAMBDA .* sum(obj.OMEGA .* (obj.PHI + ...
                abs(obj.THETA).^2), 3) - abs(MEAN0).^2;
            valInit = 0;

        end
        
        
        %The actual estimator
        function [XHAT, XVAR, NKL] = estim(obj, RHAT, RVAR)
            
            %Preallocate storage
            A_d = zeros(obj.N,obj.T,obj.D); 
            B_d = zeros(obj.N,obj.T,obj.D);
            C_d = zeros(obj.N,obj.T,obj.D);
            
            %Run through mixture components
            for d = 1:obj.D
                VARSUM = obj.PHI(:,:,d) + RVAR;
                A_d(:,:,d) = obj.LAMBDA .* obj.OMEGA(:,:,d) .* ...
                    exp(-abs(RHAT - obj.THETA(:,:,d)).^2./2./VARSUM) ./ ...
                    sqrt(VARSUM);
                B_d(:,:,d) = (RVAR .* obj.THETA(:,:,d) + ...
                    obj.PHI(:,:,d) .* RHAT) ./ VARSUM;
                C_d(:,:,d) = obj.PHI(:,:,d) .* RVAR ./ VARSUM;
            end
            
            A_d(isnan(A_d)) = 0.999;
            A_d(A_d == 0) = realmin;
            B = exp(-(RHAT.^2)./2./RVAR) ./ sqrt(RVAR);
            B(isnan(B)) = 0.999;
            
            %Sum intermediate results
            ZETA = sum(A_d, 3) + (1 - obj.LAMBDA).*B;
            
            %Compute xhat
            XHAT = sum(A_d .* B_d, 3) ./ ZETA;
            XVAR = sum(A_d .* (C_d + abs(B_d).^2), 3) ./ ZETA - ...
                abs(XHAT).^2;
            
            %Compute Negative KL divergence
            if (nargout >= 3)
                dummy2 = repmat(RHAT, [1, 1, obj.D]);
%                 NKL = log(ZETA.*sqrt(RVAR))-(sum(A_d.*(C_d+...
%                     abs(B_d-dummy2).^2),2)+(1-obj.LAMBDA)*B*...
%                     sum(RHAT.^2))./(2*RVAR.*zeta);
                NKL = log(ZETA .* sqrt(RVAR)) - (sum(A_d.*(C_d + ...
                    abs(B_d - dummy2).^2), 3)) ./ (2*RVAR.*ZETA);
            end;
        end
        
        
        %Function to generate a random draw. This method returns a matrix
        %of dimension N-by-T containing i.i.d. draws from the Gaussian
        %mixture
        function A = genRand(obj) 
            %Holders
            dummy = cat(3, zeros(obj.N,obj.T), cumsum(obj.OMEGA, 3));
            dummy2 = rand(obj.N,obj.T);
            
            A = zeros(obj.N,obj.T);
            for d = 1:obj.D
                A = A + (dummy2 >= dummy(:,:,d) & dummy2 < dummy(:,:,d+1)).*...
                    (obj.THETA(:,:,d+1) + sqrt(obj.PHI(:,:,d+1)).*randn(obj.N,obj.T));
            end
            
            A = A.*(rand(obj.N,obj.T) < obj.LAMBDA);
        end
        
        % Get method for # of mixture components
        function N = get.N(obj)
            N = size(obj.LAMBDA, 1);
        end
        
        % Get method for # of mixture components
        function T = get.T(obj)
            T = size(obj.LAMBDA, 2);
        end
        
        % Get method for # of mixture components
        function D = get.D(obj)
            D = size(obj.OMEGA, 3);
        end
        
        
    end
    
end