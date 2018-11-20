classdef NNGMEstimIn < EstimIn
    % NNGMEstimIn:  Non-negative Gaussian Mixture scalar input estimation function
    
    properties 
        omega; % Weights
        theta;  % Means 
        phi;   % Variances 
        const; % normalizing constant
    end
    
    methods
        % Constructor
        % omega: weight or probability of a given model component (need not be non-normalized to 1)
        % theta: Mean of a given component (pre-truncation)
        % phi: Variance of a given component (pre-truncation)
        %
        % The arguments are 3-dimensional matrices, the first two dimensions correspond to
        % the dimensions of the matrix to be estimated. 
        % e.g. if the arguments are size (1000,5,3), the input vector is 1000x5 and its pdf has 3
        % Gaussian components
        function obj = NNGMEstimIn(omega, theta, phi)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.omega = omega;
                obj.theta = theta;
                obj.phi = phi;
                obj.const = max(erfc(-theta./sqrt(phi)/sqrt(2)),1e-300);

                % normalize omega so sum(obj.omega, 3) ==1
                L = size(obj.theta,3);
                obj.omega = obj.omega ./ repmat(sum(obj.omega, 3), [1, 1, L]);
            end
        end

        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            %Define necessary constants
            kappa = -obj.theta./sqrt(obj.phi);
            inv_mill = sqrt(2/pi)./erfcx(kappa/sqrt(2));
            %Find mean and variance of the prior
            mean0 = sum(obj.omega.*(obj.theta + sqrt(obj.phi).*inv_mill),3);
            var0  = sum(obj.omega.*(obj.phi.*(1-inv_mill.*(inv_mill - kappa))...
                + (obj.theta + sqrt(obj.phi).*inv_mill).^2),3)- mean0.^2;
            valInit = 0;
        end


        function [Uhat, Uvar, NKL] = estim(obj, Rhat, Rvar)
            Rhat = real(Rhat);
            
            %Get the number of mixture components
            L = size(obj.theta,3);

            %Grab the signal dimension
            [M, N] = size(Rhat);

            %Expand scalar estimator if needed
            omega = obj.omega;
            theta = obj.theta;
            phi = obj.phi;
            const = obj.const;
            if ((size(omega,1)==1)&(M>1))
              omega = repmat(omega,[M,1,1]);
              theta = repmat(theta,[M,1,1]);
              phi = repmat(phi,[M,1,1]);
              const = repmat(const,[M,1,1]);
            end
            if ((size(omega,2)==1)&(N>1))
              omega = repmat(omega,[1,N,1]);
              theta = repmat(theta,[1,N,1]);
              phi = repmat(phi,[1,N,1]);
              const = repmat(const,[1,N,1]);
            end

            if L == 1
                
                alpha = phi + Rvar + eps;
                gamma = (Rvar.*theta+phi.*Rhat)./alpha;
                nu = phi.*Rvar./alpha;
                eta = -gamma./sqrt(nu);

                cdf_comp = max(erfc(eta/sqrt(2)),1e-300);
                inv_mill = sqrt(2/pi)./erfcx(eta/sqrt(2));

                % Compute MMSE estimates
                Uhat = gamma + sqrt(nu).*inv_mill;
                Uvar = nu.*(1-inv_mill.*(inv_mill - eta))...
                    + (gamma + sqrt(nu).*inv_mill).^2 - Uhat.^2;
                
                % Compute the negative KL divergence            
                if (nargout >= 3)                            
                    NKL0 = 0.5*log(Rvar./alpha./(const*0.5).^2)...
		    	-(Rhat-theta).^2./2./alpha...
			+(Uvar +(Uhat - Rhat).^2)./(2*Rvar);
		    NKL = NKL0 + log(cdf_comp*0.5);
                    %Find indices that could cause numerical issues in 
                    %erfc computation and use erfc(x) \approx 
		    %(0.3480242*t-0.0958798*t^2+0.7478556*t^3)*exp(-x^2)
		    %for t=1/(1+.47047*x), from Abramowitz and Stegun
                    I = find(eta > 10);
		    tI = 1./(1+0.47047*eta(I)/sqrt(2));
		    NKL(I) = NKL0(I) -0.5*eta(I).^2 + log(0.5)...
		    	+log(0.3480242*tI-0.0958798*tI.^2+0.7478556*tI.^3);
                end
            
            else

                %Preallocate storage
                eta = zeros(M,N,L); gamma = zeros(M,N,L); 
                nu = zeros(M,N,L); kappa = zeros(M,N,L);

                %Run through mixture components
                for i = 1:L
                    eta(:,:,i) = phi(:,:,i) + Rvar + eps;
                    gamma(:,:,i) = (Rvar.*theta(:,:,i)+ ...
                        phi(:,:,i).*Rhat)./eta(:,:,i);
                    kappa(:,:,i) = (Rhat-theta(:,:,i)).^2./eta(:,:,i);
                    nu(:,:,i) = phi(:,:,i).*Rvar./eta(:,:,i);
                end;   

                %compute terms inside of cdf/pdf components
                alpha = -gamma./sqrt(nu);
                cdf_comp = min(max(erfc(alpha/sqrt(2)),1e-12),2-1e-12);
                scale = omega.*cdf_comp./const./sqrt(eta);
                inv_mill = max(sqrt(2/pi)./erfcx(alpha/sqrt(2)),eps);

                lik = zeros(M,N,L);
                for i = 1:L
                    lik = lik + repmat(scale(:,:,i),[1 1 L])./scale...
                        .*exp((kappa - repmat(kappa(:,:,i),[1 1 L]))/2);
                end

                %Compute MMSE quantities
                Uhat = sum((gamma + sqrt(nu).*inv_mill)./lik,3);
                Uvar = sum((nu.*(1-inv_mill.*(inv_mill - alpha))...
                    + (gamma + sqrt(nu).*inv_mill).^2)./lik,3) - Uhat.^2;


                % Compute the negative KL divergence            
                if (nargout >= 3)
                    zeta = sum(omega.*exp(-kappa/2)./sqrt(2*pi*eta)...
                        .*cdf_comp./const,3);
                    zeta(zeta == 0) = eps;
                    NKL = log(zeta)+ 0.5*log(2*pi*Rvar)+ 0.5*(Uvar +(Uhat - Rhat).^2)./Rvar;
%                     NKL = log(sum(omega.*exp(-kappa/2)./sqrt(eta).*cdf_comp/2,3)...
%                         ./sum(omega.*const,3)...
%                         .*sqrt(Rvar))+(Uvar +(Uhat - Rhat).^2)./(2*Rvar);

                end
            
            end

        end
        
        % Generate random samples
        function x = genRand(obj, outSize)

            if (size(obj.omega,1)~=1)||(size(obj.omega,2)~=1)
                error('genRand() implemented only for scalar NNGMEstimIn');
            end

            if isscalar(outSize)
                row = outSize;
                col = 1;
            else
                row = outSize(1);
                col = outSize(2);
            end

            L = size(obj.omega,3);
            omega = squeeze(obj.omega(1,1,:));
            theta = squeeze(obj.theta(1,1,:));
            phi = squeeze(obj.phi(1,1,:));
            %dummy0 = cumsum(obj.omega,3);
            %dummy = zeros(row, col, L+1);
            %dummy(:,:,2:end) = dummy0;
            dummy = [0;cumsum(omega)];
            dummy2 = rand(row,col);
                
            x = zeros(row,col);
            for i = 1:L
                %temp = erfc(-obj.theta(:,:,i)./sqrt(2.*obj.phi(:,:,i)))/2;
                %temp2 = (dummy2>=dummy(:,:,i) & dummy2<dummy(:,:,i+1));
                %temp = sqrt(2.*obj.phi(:,:,i)).*erfinv(2*(1- temp...
                %    + rand(row,col,1).*temp)-1)+obj.theta(:,:,i);
                %x(temp2) = temp(temp2);
                temp = erfc(-theta(i)./sqrt(2*phi(i)))/2;
                temp2 = ((dummy2>=dummy(i))&(dummy2<dummy(i+1)));
                temp = sqrt(2*phi(i))*erfinv(2*(1-temp+rand(row,col)*temp)-1)+theta(i);
                x(temp2) = temp(temp2);
            end

        end
        
        % Computes the likelihood p(y) for y = x + v, v = N(0,Yvar)
        function py = plikey(obj,Y,Yvar)
            
            Yvar = max(eps,Yvar);
            Y = real(Y);
            
            L = size(obj.omega,3);
            [M, T] = size(Y);
            lik = zeros(M,T,L);
            gamma = zeros(M,T,L); nu = zeros(M,T,L);
            
            for i = 1:L
                dummy = obj.phi(:,:,i) + Yvar + eps;
                gamma(:,:,i) = (Yvar.*obj.theta(:,:,i)+obj.phi(:,:,i).*Y)./dummy;
                nu(:,:,i) = obj.phi(:,:,i).*Yvar./dummy;
                lik(:,:,i) = obj.omega(:,:,i).*exp(-1./(2*(obj.phi(:,:,i)+Yvar+eps))...
                    .*(Y-obj.theta(:,:,i)).^2)./sqrt(2*pi*(obj.phi(:,:,i)+Yvar+eps));
            end
            
            lik = lik.*erfc(-gamma./sqrt(nu)/sqrt(2))./obj.const;
            
            %lik(isnan(lik)) = 0.99;
            lik(lik==0) = 1e-300;
            py = sum(lik,3);
        end

    end
    
end

