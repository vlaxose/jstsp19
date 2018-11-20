classdef GMEstimIn < EstimIn
    % GMEstimIn:  Real-valued Gaussian Mixture scalar input estimation function
    
    properties 
        omega = 1;              % Weights
        theta = 0;              % Means 
        phi = 1;                % Variances 
        autoTune = false;       % Set to true for taut tuning of params
        disableTune = false;    % Set to true to temporarily disable tuning
        omegaTune = true;       % Enable tuning of weights
        thetaTune = true;       % Enable Tuning of mean0
        phiTune = true;         % Enable Tuning of var0
        tuneDim = 'joint';      % Determine dimension to autoTune over
        counter = 0;            % Counter to delay tuning
    end
    
    properties (Hidden)
        mixWeight = 1;              % Weights for autoTuning
    end

    methods
        % Constructor
        % omega: weight or probability of a given model component (need not be non-normalized to 1)
        % theta: Mean of a given component
        % phi: Variance of a given component
        %
        % The arguments are 3-dimensional matrices, the first two dimensions correspond to
        % the dimensions of the matrix to be estimated. 
        % e.g. if the arguments are size (1000,5,3), the input vector is 1000x5 and its pdf has 3
        % Gaussian components
        function obj = GMEstimIn(omega, theta, phi, varargin)
            obj = obj@EstimIn;
            obj.omega = omega;
            obj.theta = theta;
            obj.phi = phi;   

            % normalize omega so sum(obj.omega, 3) ==1
            L = size(obj.omega,3);
            obj.omega = obj.omega ./ repmat(sum(obj.omega, 3), [1, 1, L]);
            for i = 1:2:length(varargin)
                obj.(varargin{i}) = varargin{i+1};
            end
        end
        
        %Set Methods
        function obj = set.omega(obj, omega)
            obj.omega = omega;
        end
        
        function obj = set.theta(obj, theta)
            obj.theta = theta;
        end
        
        function obj = set.phi(obj, phi)
            assert(all(phi(:) > 0), ...
                'GMEstimIn: variances must be positive');
            obj.phi = phi;
        end
        
        function obj = set.mixWeight(obj, mixWeight)
            assert(all(mixWeight(:) >= 0), ...
                'GMEstimIn: weights must be positive');
            obj.mixWeight = mixWeight;
        end

        function set.disableTune(obj, flag)
            assert(isscalar(flag), ...
                'GMEstimIn: disableTune must be a logical scalar');
            obj.disableTune = flag;
        end

        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = sum(obj.theta.*obj.omega,3);
            var0  = sum(obj.omega .* (obj.phi + ...
                abs(obj.theta).^2), 3) - abs(mean0).^2;
            valInit = 0;
        end

        function [xhat, xvar, NKL] = estim(obj, Rhat, Rvar)
            Rhat = real(Rhat);

            %Get the number of mixture components
            L = size(obj.omega,3);

            %Grab the signal dimension
            [N, T] = size(Rhat);

            %Expand scalar estimator if needed
            omega = resize(obj.omega, N, T, L);
            theta = resize(obj.theta, N, T, L);
            phi = resize(obj.phi, N, T, L);

            %Preallocate storage
            gamma = zeros(N,T,L); alpha = zeros(N,T,L);
            beta = zeros(N,T,L); nu = zeros(N,T,L);

            for i = 1:L
               beta(:,:,i) = phi(:,:,i) + Rvar + eps;
               alpha(:,:,i) = (Rhat-theta(:,:,i)).^2./beta(:,:,i);
               gamma(:,:,i) = (Rhat.*phi(:,:,i) + theta(:,:,i).*Rvar)./beta(:,:,i);
               nu(:,:,i) = Rvar.*phi(:,:,i)./beta(:,:,i);
            end

            lik = zeros(N,T,L);
            for i = 1:L
                lik = lik + repmat(omega(:,:,i),[1 1 L])./omega...
                    .*sqrt(beta./repmat(beta(:,:,i),[1 1 L]))...
                    .*exp((alpha-repmat(alpha(:,:,i),[1 1 L]))/2);
            end
            
            xhat = sum(gamma./lik,3);
            xvar = sum((nu + gamma.^2)./lik,3) - xhat.^2;
            
            % Autotune the GM parameters via approzimate expectation
            % maximization
            if obj.autoTune && ~obj.disableTune

              if (obj.counter>0), % don't tune yet
                obj.counter = obj.counter-1; % decrement counter 
              else % tune now

                % If mix weight is a scalar, do not expand to save memory
                % and multiplications
                if numel(obj.mixWeight) ~= 1
                    obj.mixWeight = resize(obj.mixWeight, N, T, L);
                end
                lik = obj.mixWeight./lik;
                % Learn based on the various dimensions
                switch obj.tuneDim
                    case 'joint'
                        %Compute Scaling factor
                        sumlik = sum(sum(lik,1),2);
                        if obj.phiTune
                            obj.phi = sum(sum((nu+abs(gamma-theta).^2).*lik,1),2)./sumlik;
                        end
                        if obj.thetaTune
                            obj.theta = sum(sum(gamma.*lik,1),2)./sumlik;
                        end
                        if obj.omegaTune
                            obj.omega = sumlik./sum(sum(obj.mixWeight,1),2);
                        end
                    case 'col'
                        %Compute Scaling factor
                        sumlik = sum(lik,1);
                        if obj.phiTune
                            obj.phi = repmat(sum((nu+abs(gamma-theta).^2).*lik,1)./sumlik, [N 1 1]);
                        end
                        if obj.thetaTune
                            obj.theta = repmat(sum(gamma.*lik,1)./sumlik, [N 1 1]);
                        end
                        if obj.omegaTune
                            obj.omega = repmat(sumlik./sum(obj.mixWeight,1), [N 1 1]);
                        end
                    case 'row'
                        %Compute Scaling factor
                        sumlik = sum(lik,2);
                        if obj.phiTune
                            obj.phi = repmat(sum((nu+abs(gamma-theta).^2).*lik,2)./sumlik, [1 T 1]);
                        end
                        if obj.thetaTune
                            obj.theta = repmat(sum(gamma.*lik,2)./sumlik, [1 T 1]);
                        end
                        if obj.omegaTune
                            obj.omega = repmat(sumlik./sum(sum(obj.mixWeight,2),2), [1 T 1]);
                        end
                    otherwise
                         error('Invalid tuning dimension in GMEstimIn');
                end
 
              end
            end
               
            % Compute the negative KL divergence            
            if (nargout >= 3)                  
                zeta = sum(omega.*exp(-alpha/2)./sqrt(2*pi*beta),3);
                zeta(zeta == 0) = eps;
                NKL = log(zeta)+ 0.5*log(2*pi*Rvar)+ 0.5*(xvar +(xhat - Rhat).^2)./Rvar;
            end

        end

        % Generate random samples
        function x = genRand(obj, outSize)

            if (size(obj.omega,1)~=1)||(size(obj.omega,2)~=1)
	        error('genRand() implemented only for scalar GMEstimIn');
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

            dummy = [0;cumsum(omega)];
            dummy2 = rand(row,col);
            dummy3 = zeros(row,col,L);
            for i = 1:L
                dummy3(:,:,i) = ((dummy2>=dummy(i))&(dummy2<dummy(i+1)))...
	            .*(theta(i) + sqrt(phi(i))...
		    .*randn(row,col));
            end;
            x = sum(dummy3,3);
        end

        % Computes the likelihood p(y) for y = x + v, v = N(0,Yvar)
        function py = plikey(obj,Y,Yvar)
            Y = real(Y);
            
            L = size(obj.omega,3);
            [M, T] = size(Y);
            lik = zeros(M,T,L);
            
            for i = 1:L
                lik(:,:,i) = obj.omega(:,:,i).*exp(-1./(2*(obj.phi(:,:,i)+Yvar))...
                    .*(Y-obj.theta(:,:,i)).^2)./sqrt(2*pi*(obj.phi(:,:,i)+Yvar));
            end
            
            lik(isnan(lik)) = 0.999;
            py = sum(lik,3);
        end
        
        % Computes the log-likelihood, log p(Y(i,j)), for Y = X + V, where 
        % p(X(i,j)) = sum_k omega(i,j,k)*CN(theta(i,j,k), phi(i,j,k)) and 
        % p(V(i,j)) = CN(0, Yvar(i,j))
        function logpy = loglikey(obj, Y, Yvar)
            logpy = log(obj.plikey(Y, Yvar));
        end

        % it is hard to compute the expected log likelihood of a GMM,
        % but easy to compute the loglikelihood at a given point
        function FIXME = logLike(obj,zhat,zvar)
            FIXME = obj.loglikey(zhat,zvar);
        end


    end
    
end

