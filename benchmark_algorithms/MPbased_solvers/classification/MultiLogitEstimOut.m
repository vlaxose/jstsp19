% CLASS: MultiLogitEstimOut
%
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
%
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The MultiLogitEstimOut class defines a vector observation channel,
%   p(y|z), (where y is scalar and z is vector valued) that constitutes a 
%   multinomial logistic regression model. 
%   z_i = w_i \tran x where w_i is the weight vector corresponding to the
%   i-th class and x is a feature vector.  y is in the set {1,...,D}
%   and indicates the class label for feature vector x.
%   The multinomial logistic regression model is given below
%       Pr(y=d|z) = exp(z_d)/(exp(z_1) +...+ exp(z_D))
%       (or equivalently)
%       Pr(y=d|z) = 1/(exp(z_1-z_d) +...+ 1 +...+ exp(z_D-z_d))
%
%   MultiLogitEstimOut can operate in Sum-Product or Min-Sum modes.
%   
%   In Sum-Product mode, the algorithm must compute the posterior
%   mean and the diagonal of the covariance of 
%      p(z | y, phat; pvar) \propto Pr(y | z) N(z; phat, diag(pvar)),
%   where Pr(y | z) is the multinomial logistic function.
%
%   In Min-Sum mode, the algorithm must compute 
%     zhat = argmax_z log Pr(z | y) + log N(z; phat, diag(pvar))
%   and
%     zvar = diag(d^2/dz^2 [-log Pr(zhat | y) - log N(zhat; phat,
%     diag(pvar)) ])
%    
%   In Sum-Product mode, we have evaluated several methods to approximate 
%   posterior mean and variance of p(z | y, phat; pvar):
%
%   Method 1:     (default) Gaussian Mixture Approximation 
%   Method 2:     Numerical Integration
%   Method 3:     Importance Samping
%   Method 4:     Taylor Series Approximation to the Likelihood
%
%   In Min-Sum mode, we have implemented five methods to compute zhat:
%
%   Method 1:     (default) Component-wise Newton's method    
%   Method 2:     Newton's method
%
% PROPERTIES (State variables)
%
%   note that most of these properties relate to specific methods for
%   computing the Sum-Product or Max-Sum estimates and therefore do not
%   need to be initialized.
%
%   y           An M-by-1 array of integer ({1,...,D}) class labels for the
%               training data, where M is the number of training data
%               points
%
%   D           Number of classes
%
%   maxSumVal   Sum-Product mode = false; Min-Sum mode = true. 
%               [Default: False]
%
%   method      Specify the estimation method to use, e.g., cw-Newton's 
%               method [Default: 1]
%
%
% METHODS (Subroutines/functions)
%   MultiLogitEstimOut(y, D, maxsumval, method)
%       - Constructor.  y, D and mode are required inputs. The rest are optional.
%   estim(obj, phat, pvar)
%       - Computes Sum-Product or Max-Sum quantities 
%   logLike(obj,zhat,zvar)
%       - SP mode: computes expected log-likelihood of p(y|z) (for 
%         approximating cost function)
%       - MS mode: computes log p(y|z)
%   logScale(obj,Axhat,pvar,phat)
%       - SP mode: computes quantities needed for true cost function [may
%         be slow]
%       - MS mode: N/A 
%   numColumns(obj)
%       - returns number of columns which corresponds to number of classes

% Last change: 9/21/15
% Change summary:
%       - Created (v1.0) (9/21/15; EMB)
%      
% Copyright 2015 Evan Byrne 

classdef MultiLogitEstimOut < EstimOut
    
    properties
      
        % general properties
        y;                  % vector of D-ary class labels 
        D;                  % number of categories
        maxSumVal = false; 	% Sum-Product (false) or Min-Sum (true) GAMP
        method = 1;         % specify method to compute estimates
        
              
        % data storage properties (unmodifiable)      
        gmObj = [];         % Gaussian Mixture Distribution for Method 7 
        q;                  % sample points in numerical integration (importance sampling)
        idx;                % indices of particles in the numerator of the logistic function
        ll;                 % a place to store the log likelihood to be used across methods
        zhat;               % a place to store zhat from the previous iteration
        ny;                 % a place to store the values of 1:D \ y
        y_idx;              % the first m values of idx
        ny_idx;             % the complement to y_idx
        L = 2;                  % number of Gaussian mixture components 
        weights;            % weight vector for gaussian mixture components
        means;              % mean vector for gaussian mixture components
        sigmas;             % standard deviation vector for gaussian mixture components
        tempMode;           % method of integration for one iteration of gamp
        
    end
    
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = MultiLogitEstimOut(y, D, maxsumval, method)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                
                obj.y = y;
                
                if min(obj.y) <= 0
                    obj.y = obj.y - min(obj.y) + 1;
                end
                
                obj.D = D;
                
                if (nargin >= 3) && ~isempty(maxsumval)
                    if isscalar(maxsumval)
                        obj.maxSumVal = logical(maxsumval);
                    else
                        error('maxSumVal must be a logical scalar')
                    end
                end
                
                if (nargin >= 4) && ~isempty(method)
                    obj.method = method;
                end
    
                m = size(obj.y,1);
                
                obj.ll = zeros(m,1);
                
                % set up importance sampling parameters
                if (obj.method == 3  && ~obj.maxSumVal) 
                    % generate array of random sample points of size
                    % m x D x npts
                    npts = 1500; % 1500 is the default number of importance sampling points
                    obj.q = randn(m,D,npts);  
                    
                    % get indices of points that are in the numerator of the
                    % logistic function
                    a=uint32(repmat([1:m]',npts,1));
                    b=uint32(repmat(y,npts,1));
                    c=uint32(sort(repmat([1:npts]',m,1)));
                    obj.idx=sub2ind(size(obj.q),a,b,c);
                end
                
                % set up Taylor Series parameters
                if (obj.method == 4 && ~obj.maxSumVal) 
                    set = 1:obj.D;
                    obj.ny = nan(m,obj.D-1);
                    for i = 1:numel(obj.y)
                        obj.ny(i,:) = set(set~=obj.y(i));
                    end
                    m = numel(obj.y);
                    obj.y_idx = sub2ind([m,obj.D],[1:m]',obj.y);
                    obj.ny_idx = sub2ind([m,obj.D],[1:m]'*ones(1,obj.D-1),obj.ny);
                end
                
                % set up Gaussian Mixture Parameters for Method 1
                if (obj.method == 1 && ~obj.maxSumVal)
                    
                    % if gmObj is unspecified, load pre-computed parameters
                    % else use custom parameters stored within gmObj
                    if isempty(obj.gmObj)

                        if  exist('LogitGMparams.mat','file') == 2
                            load LogitGMparams
                        else
                            error('Missing LogitGMparams')
                        end

                        % make sure D is supported
                        if obj.D ~= 1
                            try
                                obj.gmObj = GMparams{obj.D,obj.L};
                            catch
%                                 warning('GM params for D = %d, L = %d have not been learned, using alternate gm dist',obj.D, obj.L)
%                                 obj.gmObj = gmdistribution(zeros(1,obj.D-1), eye(obj.D-1), 1);
                                obj.gmObj = gmdistribution(zeros(1,obj.D-1), obj.D^(2/(obj.D-1)) / sqrt(2 * pi) * eye(obj.D-1), 1);
                            end
                            
                            if isempty(obj.gmObj)
%                                 warning('GM params for D = %d, L = %d have not been learned, using alternate gm dist',obj.D, obj.L)
                                obj.gmObj = gmdistribution(zeros(1,obj.D-1), obj.D^(2/(obj.D-1)) / sqrt(2 * pi) * eye(obj.D-1), 1);
                            end
                                
                        else
                            error('D = %d is not currently supported',obj.D)
                        end


                    else
                        
                        if (obj.D - 1 ~= obj.gmObj.NDimensions)
                            error('Dimension of GM distribution must be D - 1')
                        end
                        
                    end
                    
                    obj.L = obj.gmObj.NComponents;
                    
                    obj.weights = obj.gmObj.PComponents;
                    obj.means = reshape(obj.gmObj.mu', [obj.D-1, 1, obj.L]);
                    obj.sigmas = reshape(squeeze(sum(sqrt(obj.gmObj.Sigma),1)), [obj.D-1, 1, obj.L]);

                end
                
            end
        end
        
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.y(obj, y)
            if ~all((y(:) > 0) & rem(y,1)==0 )
                error('Elements of y must be positive integer')
            else
                obj.y = y;
            end
        end

        
        % *****************************************************************
        %                          ESTIM METHOD
        % *****************************************************************
        % This function will compute the posterior mean and variance of a
        % random vector z with prior distribution N(zhat0, zvar0), given
        % observations y obtained through the separable channel model:
        % Pr{y(m)=d|z(m)} = exp(z(d)) / sum(exp(scale*z(d))), (z(m) is a
        % length D vector
        function [zhat, zvar] = estim(obj, phat, pvar)
            
            % Check if zhat0 and zvar0 are only scalars (can occur during
            % first method call by gampEst) and resize
            if numel(phat) == 1 && numel(pvar) == 1
                phat = phat*ones(size(obj.y),numel(unique(obj.y)));
                pvar = pvar*ones(size(obj.y),numel(unique(obj.y)));
            elseif numel(phat) > numel(pvar)
                pvar = pvar*ones(size(phat));
            elseif numel(pvar) > numel(phat)
                phat = phat*ones(size(pvar));
            end
            
            %             if numel(phat) == 1, phat = phat*ones(size(obj.y)); ends
            %             if numel(pvar) == 1, pvar = pvar*ones(size(obj.y)); end
            
            % matrix dimensions
            [M, D] = size(phat);
            
            switch obj.maxSumVal
                
                case false   % Sum-Product GAMP
                    
                    switch obj.method

                        case 1 % Gaussian Mixture Approximation 

                            y = obj.y;
                            L = obj.L;
                            
                            % Gaussian mixture parameters
                            weights = obj.weights;
                            means = obj.means;
                            sigmas = obj.sigmas;
                            
                            % number of numerical integration points
                            % (default is 7)
                            K = 7; 
                            
                            setd = cell(D,1);
                            for d=1:D
                                setd{d} = setdiff(1:D,d);
                            end
                            phat2 = reshape(phat',[D,1,1,M]);
                            pvar2 = reshape(pvar',[D,1,1,M]);
                            
                            indy = sub2ind([D,1,1,M],y,ones(M,1),ones(M,1),(1:M)');
                            nindy = setdiff(1:D*M, indy);
                            
                            pvar2r = reshape(pvar2(indy),[1,1,1,M]);
                            phat2r = reshape(phat2(indy),[1,1,1,M]);
                            
                            phat2m = reshape(phat2(nindy),[D-1,1,1,M]);
                            pvar2m = reshape(pvar2(nindy),[D-1,1,1,M]);
                            
                            v = linspace(-1,1,K);
                            v = bsxfun(@times, 4 * sqrt(pvar2r), v);
                            v = bsxfun(@plus, phat2r, v);
                            
                            % important values
                            fvn = bsxfun(@times, 1/sqrt(2*pi)./pvar2r, ...
                                exp(bsxfun(@times, -(bsxfun(@plus, v, -phat2r)).^2, 1./(2*pvar2r))));
                            
                            svar = bsxfun(@plus, sigmas.^2, pvar2m);
                            x = bsxfun(@times, bsxfun(@plus, bsxfun(@plus, v, -means), -phat2m),  1./sqrt(svar));
                            r = (2/sqrt(2*pi)) * (erfcx(-x / sqrt(2)).^(-1));
                            
                            % values T0, T1, and T2 (moments of normpdf
                            % multiplied with normcdf)
                            T0 = .5*erfc(-x/sqrt(2)) + eps;
                            T0p = r.*T0;
                            
                            T1 = bsxfun(@times, bsxfun(@plus, v, -phat2m), T0) ...
                                + bsxfun(@times, bsxfun(@times, T0p, pvar2m), 1./sqrt(svar));
                            
                            T2 = T1.^2./T0 + bsxfun(@times, T0, pvar2m) ...
                                - bsxfun(@times, bsxfun(@times, T0p, pvar2m.^2), 1./svar) .* (x + r);
                            
                            % computing moments of posterior
                            prodT0 = prod(T0,1);
                            
                            val1 = bsxfun(@times, v, prodT0);
                            val2 = bsxfun(@times, v, val1);
                            
                            prodT0_ = bsxfun(@times, prodT0, 1./T0);
                            val3 = T1 .* prodT0_;
                            
                            temp1 = bsxfun(@times, fvn, val1);
                            temp2 = bsxfun(@times, fvn, val2);
                            temp3 = bsxfun(@times, fvn,  bsxfun(@plus, val1, - val3));
                            temp4 = bsxfun(@times, fvn, bsxfun(@plus, val2,  -2 * bsxfun(@times, v, val3) + prodT0_ .* T2));
                            
                            mom1 = nan(D,K,L,M);
                            mom2 = mom1;
                            for m = 1:M
                                mom1(y(m),:,:,m) = temp1(:,:,:,m);
                                mom1(setd{y(m)},:,:,m) = temp3(:,:,:,m);
                                mom2(y(m),:,:,m) = temp2(:,:,:,m);
                                mom2(setd{y(m)},:,:,m) = temp4(:,:,:,m);
                            end
                            
                            % scale factor
                            C = bsxfun(@times, fvn, prodT0);
                            C = sum(bsxfun(@times, reshape(weights, [1,1,L]), sum(C,2)),3);
                            
                            zhat = reshape(bsxfun(@times, sum(bsxfun(@times, reshape(weights, [1,1,L]), sum(mom1,2)),3), 1./C),[D,M])';
                            zvar = reshape(bsxfun(@times, sum(bsxfun(@times, reshape(weights, [1,1,L]), sum(mom2,2)),3), 1./C), [D,M])' - zhat.^2;

                            return
                            
                        case 2  % Numerical Integration
                            
                            K = 7; % number of numerical integration points
                            
                            y = obj.y;
                            x = linspace(-1,1,K);
                            q = cell(1,D);
                            dz = nan(1,D);
                            obj.ll = nan(M,1);
                            
                            for m = 1:M
                                
                                % set up points
                                z = cell(1,D);
                                
                                for k = 1:D
                                    temp = 4*x*sqrt(pvar(m,k))+phat(m,k);
                                    z{k} = temp;
                                    dz(k) = temp(2)-temp(1);
                                end
                                
                                [q{1:D}] = ndgrid(z{:});
                                
                                r = cat(D+1,q{:});
                                
                                wn = exp(q{y(m)});
                                wd = sum(exp(r),D+1);
                                
                                w = wn./wd;
                                
                                % corrections for numerical robustness
                                w((isinf(wn) & isinf(wd))) = 1/D;
                                w(((wn == 0) & (wd == 0))) = 1/D;
                                w((isinf(wn) & (wd == 0))) = 1;
                                                                
                                g = ones(size(q{1}));
                                
                                for k = 1:obj.D
                                    g = g.* normpdf(q{k},phat(m,k),sqrt(pvar(m,k)));
                                end
                                
                                p = w.*g;
                                pl = log(w).*g;
                                obj.ll = sum(pl(:))*prod(dz);
                                
                                for k = 1:D
                                    temp = q{k}.*p;
                                    temp2 = temp.*q{k};
                                    zhat(m,k) = sum(temp(:))/sum(p(:));
                                    zvar(m,k) = sum(temp2(:))/sum(p(:)) - zhat(m,k)^2;
                                end

                            end
                            
                            return
                            
                        case 3  % Importance sampling
                            
                            K = size(obj.q,3);
                            idx = obj.idx;
                            
                            q = bsxfun(@plus, bsxfun(@times,sqrt(pvar),obj.q), phat);
                            
                            % compute the weights
                            w = 1./sum(exp(bsxfun(@minus,q,reshape(q(idx),[M,1,K]))),2);
                            
                            % this line is used in the logLike method, calculated in advance
                            obj.ll = squeeze(w);
                            
                            num = sum(bsxfun(@times,q(:,1:D,:),w),3);
                            den = sum(w,3);
                            
                            zhat = bsxfun(@times,1./den,num);
%                             obj.zhat = zhat;
                            
                            num = sum(bsxfun(@times,q(:,1:D,:).*q(:,1:D,:),w),3);
                            
                            zvar = bsxfun(@times,1./den,num)-zhat.^2;
                            
                            return
                            
                        case 4  % Taylor series approximation to the likelihood
                            
                            if max(pvar(:)) > 1
                                warning('pvar may be too large for the taylor series approximation')
                            end
                            
                            % shorthand variables
                            ny = obj.ny;
                            y = obj.y;
                            y_idx = obj.y_idx;
                            ny_idx = obj.ny_idx;
                            
                            % initialize arrays
                            lin = nan(M,D);
                            quad = lin;
                            
                            % calculate the derivatives evaluated at phat
                            lin(y_idx) = exp(phat(y_idx)) .* sum(exp(phat(ny_idx)),2) ./ (sum(exp(phat),2)).^2;
                            lin(ny_idx) = bsxfun(@times, -exp(bsxfun(@plus,phat(y_idx), phat(ny_idx))),1./ (sum(exp(phat),2)).^2);
                            
                            quad(y_idx) = exp(phat(y_idx)) .* (sum(exp(phat(ny_idx)),2)) .* (sum(exp(phat(ny_idx)),2) - exp(phat(y_idx))) ./ (sum(exp(phat),2)).^3;
                            quad(ny_idx) = bsxfun(@times,- exp(bsxfun(@plus,phat(y_idx), phat(ny_idx))) .* bsxfun(@plus,sum(exp(phat),2), - 2 * exp(phat(ny_idx))), 1./ (sum(exp(phat),2)).^3);
                            
                            % logistic function evaluated at phat
                            f = exp(phat(y_idx))./sum(exp(phat),2);
                            
                            % normalizing constant
                            C = f + sum(quad/2 .* pvar,2);
                            
                            % calculate the moments
                            mom1 = bsxfun(@times,f,phat) + ...
                                lin.*pvar + ...
                                bsxfun(@times,sum(1/2 * quad .* pvar,2), phat);
                            
                            mom2 = bsxfun(@times,f,phat.^2 + pvar) + ...
                                2 * lin .* phat .* pvar + ...
                                bsxfun(@times,sum(1/2 * quad .* pvar,2), pvar + phat.^2) + ...
                                quad .* pvar.^2;
                                                       
                            obj.ll = C;
                            
                            % calculate final mean and variance
                            zhat = bsxfun(@times,mom1,1./C);
                            zvar = bsxfun(@times,mom2,1./C) - zhat.^2;
                                                        
                            return

                    end
                    
                case true % Min-Sum GAMP
                    
                    con_tol = 1e-4;
                    
                    switch obj.method

                        case 1 % Component-wise Newton's method
                            
                            maxit = 5; % max number of iterations
                            alpha = .5; % initial step-size
                            
                            y = obj.y;
                            
                            % pre-allocate
                            zhat = nan(M,D);
                            zvar = nan(M,D);

                            for j = 1:M
                                
                                phatm = phat(j,:);
                                pvarm = pvar(j,:);
                                
                                % warm start 
                                if ~isempty(obj.zhat)
                                    X = obj.zhat(j,:)';
                                else
                                    X = phatm';
                                end
                                
                                % there could be numerical problems since 
                                % exp(large_number) = inf, try to prevent
                                % them and use trivial estimator instead
                                if any(X > 300) || any(phatm > 300)
                                    zhat(j,:) = phat(j,:);
                                    zvar(j,:) = pvar(j,:);
                                else
                                    
                                    qpi = 1./pvarm(:);
                                    
                                    for it = 1:maxit
                                        
                                        % quantities needed for
                                        % gradient/hess
                                        expX = exp(X);
                                        Sum_expX = sum(expX);
                                        expX_over_SumExpX = expX/Sum_expX;
                                        
                                        grad = expX_over_SumExpX + qpi .* (X - phatm');
                                        grad(y(j)) = grad(y(j)) - 1;

                                        hess = expX_over_SumExpX.*(-1+expX_over_SumExpX) - qpi;
                                        
                                        dk = grad./hess;
                                                              
                                        % update step
                                        Xold = X;
                                        X = X + alpha^(it-1)*dk;

                                        % check for convergence 
                                        if norm(X-Xold)/(norm(X)+eps) < con_tol
                                            break
                                        end
                                    end

                                    if any(X > 300)
                                        zhat(j,:) = phat(j,:);
                                        zvar(j,:) = pvar(j,:);
                                    else
                                        zhat(j,:) = X';
                                        % compute zvar
                                        % compute negative hession of log p(y|z)
                                        expX = exp(X);
                                        Sum_expX = sum(expX);
                                        expX_over_SumExpX = expX/Sum_expX;   
                                        hess = -expX_over_SumExpX*expX_over_SumExpX' + diag(expX_over_SumExpX);
                                        Dz = hess + diag(qpi);
                                        zvar(j,:) = diag(inv(Dz))';
                                    end
                                end
                                
                            end
                            
                            % save zhat for warmstarting on subsequent GAMP
                            % iteration
                            obj.zhat = zhat;
                            
                        case 2 % Newton's method
                            
                            maxit = 5; % max number of iterations
                            alpha = .99; % initial step-size
                            
                            y = obj.y;
                            
                            % pre-allocate
                            zhat = nan(M,D);
                            zvar = nan(M,D);

                            for j = 1:M
                                
                                phatm = phat(j,:);
                                pvarm = pvar(j,:);
                                
                                % warm start (which can mess up my trivial
                                % estimator skipping if zhat is large!!)
                                if ~isempty(obj.zhat)
                                    X = obj.zhat(j,:)';
                                else
                                    X = phatm';
                                end
                                
                                if any(X > 300) || any(phatm > 300)
                                    zhat(j,:) = phat(j,:);
                                    zvar(j,:) = pvar(j,:);
                                else
                                    
                                    qpi = diag(1./pvarm);

                                    for it = 1:maxit                               
                                        
                                        % quantities needed for
                                        % gradient/hess
                                        expX = exp(X);
                                        Sum_expX = sum(expX);
                                        expX_over_SumExpX = expX/Sum_expX;

                                        grad = expX_over_SumExpX + qpi * (X - phatm');
                                        grad(y(j)) = grad(y(j)) - 1;

                                        hess = -expX_over_SumExpX*expX_over_SumExpX' + diag(expX_over_SumExpX) + qpi;
                                        dk = -hess\grad;
                                                                           
                                        % update step
                                        Xold = X;
                                        X = X + alpha^(it-1)*dk;
                                        
                                        % check for convergence
                                        if norm(X-Xold)/(norm(X)+eps) < con_tol
                                            break
                                        end
                                    end
                                   
                                    if any(X > 300)
                                        zhat(j,:) = phat(j,:);
                                        zvar(j,:) = pvar(j,:);
                                    else
                                        zhat(j,:) = X';
                                        % compute zvar
                                        % compute negative hession of log p(y|z)
                                        expX = exp(X);
                                        Sum_expX = sum(expX);
                                        expX_over_SumExpX = expX/Sum_expX;

                                        Dz = -expX_over_SumExpX*expX_over_SumExpX' + diag(expX_over_SumExpX) + qpi;
                                        zvar(j,:) = diag(inv(Dz))';
                                    end
                                end
                                
                            end

                            % save zhat for warmstarting on subsequent GAMP
                            % iteration
                            obj.zhat = zhat;
                            
                            return
                            
                    end

            end
            
        end
        
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % This function will compute an approximation to the expected
        % log-likelihood, E_z[log p(y|z)] when performing sum-product GAMP
        % (obj.maxSumVal = false).
        % The approximation is based on Jensen's
        % inequality, i.e., computing log E_z[p(y|z)] instead.
        function ll = logLike(obj,phat,pvar)
            switch obj.maxSumVal
                case 0
                    switch obj.method
                         case 1 % gaussian mixture approximation
                            % use Jensen's Inequality to approximate the expected
                            % log likelihood
                            ll = real(sum(log(obj.ll)));
                            
                        case 2 % numerical integration
                            % numerically compute the integral for the expected log
                            % likelihood
                            ll = sum((obj.ll));
                            
                        case 3 % importance sampling
                            ll = 1/obj.npts*sum(sum(log(obj.ll),2),1);
                     
                        case 4 % taylor series approximation
                            % use Jensen's Inequality to approximate the expected
                            % log likelihood
                            ll = real(sum(log(obj.ll)));
                    end
                    
                    % quick dimension fix for vertcat
                    ll = sum(ll(:))/(obj.D) * ones(1,obj.D);
                    
                case 1
                    
                    % compute log p(y|z)
                    ll = nan(length(obj.y), 1);
                    
                    for m = 1:length(obj.y)
                        
                       ll(m) = log(exp(phat(m,obj.y(m)))/sum(exp(phat(m,:))));
                        
                    end
                    
                    % for compatibility, make this MxD
                    ll = ll * ones(1,obj.D) / obj.D;
                    
                    
            end
        end
        
        
        % *****************************************************************
        %                         LOGSCALE METHOD
        % *****************************************************************
        % This function will compute the log scale factor: 
        %  \log \int_z p_{y|z}(y|z)\mc{N}(z;phatfix,pvar)
        function LScale = logScale(obj,Axhat,pvar,phat)
     
            % Compute output cost
            if ~(obj.maxSumVal)
                
%                 error('logScale() not implemented yet. Set adaptStepBethe = 0'); 

                % Compute output cost
                
                % Find the fixed-point of phat
                opt.phat0 = phat; %Axhat; % works better than phat
                opt.alg = 1; % approximate newton's method
                opt.maxIter = 200; 
                opt.tol = 1e-3; 
                opt.stepsize = .1;
                opt.regularization = 1e-6;  
                opt.debug = false;
                phatfix = estimInvert(obj,Axhat,pvar,opt);
                
                % Compute log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
                
                % compute the intregral using the same p(y|z) that is used
                % in the estimator, even if it is not the true p(y|z)
                
                [M,D] = size(phat);
                
                switch obj.method
                    case 1 % via Gaussian Mixture Approximation [default]
                        
                        y = obj.y;
                        L = obj.L;
                        
                        % Gaussian mixture parameters
                        weights = obj.weights;
                        means = obj.means;
                        sigmas = obj.sigmas;
                        
                        % number of numerical integration points
                        % (default is 7)
                        K = 7;
                        
                        setd = cell(D,1);
                        for d=1:D
                            setd{d} = setdiff(1:D,d);
                        end
                        phat2 = reshape(phatfix',[D,1,1,M]);
                        pvar2 = reshape(pvar',[D,1,1,M]);
                        
                        indy = sub2ind([D,1,1,M],y,ones(M,1),ones(M,1),(1:M)');
                        nindy = setdiff(1:D*M, indy);
                        
                        pvar2r = reshape(pvar2(indy),[1,1,1,M]);
                        phat2r = reshape(phat2(indy),[1,1,1,M]);
                        
                        phat2m = reshape(phat2(nindy),[D-1,1,1,M]);
                        pvar2m = reshape(pvar2(nindy),[D-1,1,1,M]);
                        
                        v = linspace(-1,1,K);
                        v = bsxfun(@times, 4 * sqrt(pvar2r), v);
                        v = bsxfun(@plus, phat2r, v);
                        dv = v(1,2,1,:) - v(1,1,1,:); 
                        
                        % important values
                        fvn = bsxfun(@times, 1/sqrt(2*pi)./pvar2r, ...
                            exp(bsxfun(@times, -(bsxfun(@plus, v, -phat2r)).^2, 1./(2*pvar2r))));
                        
                        svar = bsxfun(@plus, sigmas.^2, pvar2m);
                        x = bsxfun(@times, bsxfun(@plus, bsxfun(@plus, v, -means), -phat2m),  1./sqrt(svar));

                        T0 = .5*erfc(-x/sqrt(2)) + eps;
                        
                        prodT0 = prod(T0,1);
                        
                        % scale factor
                        C = bsxfun(@times, fvn, prodT0);
                        C = bsxfun(@times, C, dv);
                        C = sum(bsxfun(@times, reshape(weights, [1,1,L]), sum(C,2)),3);
                        
                        % normalizing constant
                        ls = log(squeeze(C));
                        
                        
                    case {2, 3} % via numerical integration
                        
                        y = obj.y;
                        
                        x = linspace(-1,1,obj.grid);
                        
                        q = cell(1,obj.D);
                        
                        dz = nan(1,obj.D);
                        
                        ls = nan(M,1);

                        for j = 1:M
                            
                            % set up points
                            z = cell(1,obj.D);
                            
                            for k = 1:obj.D
                                temp = 4*x*sqrt(pvar(j,k))+phatfix(j,k);
                                z{k} = temp;
                                dz(k) = temp(2)-temp(1);
                            end
                            
                            [q{1:obj.D}] = ndgrid(z{:});
                            
                            r = cat(obj.D+1,q{:});
                            
                            % things may become nan here (try shifting
                            % values)
                            wn = exp(q{y(j)});
                            
                            wd = sum(exp(r),obj.D+1);
                            
                            w = wn./wd;
                            
                            w((isinf(wn) & isinf(wd))) = 1/obj.D;
                            w(((wn == 0) & (wd == 0))) = 1/obj.D;
                            w((isinf(wn) & (wd == 0))) = 1;
                            
                            
                            g = ones(size(q{1}));
                            
                            for k = 1:obj.D
                                g = g.* normpdf(q{k},phatfix(j,k),sqrt(pvar(j,k)));
                            end
                            
                            ls(j) = log(sum(w(:).*g(:))*prod(dz));

                        end
                        
                    case 4 % via Taylor Series Approximation
                        
                        % shorthand variables
                        ny = obj.ny;
                        y = obj.y;
                        y_idx = obj.y_idx;
                        ny_idx = obj.ny_idx;
                        
                        % initialize arrays
                        lin = nan(M,D);
                        quad = lin;
                        
                        % calculate the derivatives evaluated at phat
                        lin(y_idx) = exp(phatfix(y_idx)) .* sum(exp(phatfix(ny_idx)),2) ./ (sum(exp(phatfix),2)).^2;
                        lin(ny_idx) = bsxfun(@times, -exp(bsxfun(@plus,phatfix(y_idx), phatfix(ny_idx))),1./ (sum(exp(phatfix),2)).^2);
                        
                        quad(y_idx) = exp(phatfix(y_idx)) .* (sum(exp(phatfix(ny_idx)),2)) .* (sum(exp(phatfix(ny_idx)),2) - exp(phatfix(y_idx))) ./ (sum(exp(phatfix),2)).^3;
                        quad(ny_idx) = bsxfun(@times,- exp(bsxfun(@plus,phatfix(y_idx), phatfix(ny_idx))) .* bsxfun(@plus,sum(exp(phatfix),2), - 2 * exp(phatfix(ny_idx))), 1./ (sum(exp(phatfix),2)).^3);
                        
                        % logistic function evaluated at phat
                        f = exp(phatfix(y_idx))./sum(exp(phatfix),2);
                        
                        % normalizing constant
                        C = f + sum(quad/2 .* pvar,2);
                        
                        ls = log(C);

                end
                
                % Combine to form output cost
                % this is currently mismatched
%                 LScale = ls + 0.5*(real(Axhat - phatfix)).^2./pvar;
                
                % simple fix is to sum the second portion
                LScale = ls + sum(0.5*(real(Axhat - phatfix)).^2./pvar,2);
                LScale = LScale/D * ones(1,D);
                
            else         
                
                % compute log p(y|z)
                    
                    LScale = nan(length(obj.y), 1);
                    
                    for m = 1:length(obj.y)
                        
                       LScale(m) = log(exp(phat(m,obj.y(m)))/sum(exp(phat(m,:))));
                        
                    end
            end
            
        end
        
        function S = numColumns(obj)
            % Return number of categories (not number of columns!)
            S = obj.D;
        end
        
    end
    
end
