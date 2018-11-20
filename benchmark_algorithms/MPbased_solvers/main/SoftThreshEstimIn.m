classdef SoftThreshEstimIn < EstimIn
    % SoftThreshEstimIn:  Inputs a soft thresholding scalar input function.
    % Allows GAMP to be used to solve "min_x 1/2/var*norm(y-A*x,2)^2 + lambda*norm(x,1)".
    
    properties
        %lambda = the gain on the ell1 term in the MAP cost.
        %The soft threshold is set according to the expression thresh = lambda * mur;
        lambda;
        maxSumVal = true;   % Max-sum GAMP (true) or sum-product GAMP (false)?
        debias = false;      % perform debiasing (Max-sum only!)
        autoTune = false;   % Perform tuning of lambda (true) or not (false)?
        disableTune = false;% Set to true to temporarily disable tuning
        tuneDim = 'joint';  % Parameter tuning across rows and columns ('joint')
                            %   or just columns ('col') or rows ('row')
        counter = 0;        % Counter to delay tuning
        sureParams = ...
            struct('method',1,...
            'damp',1,...
            'decayDamp',1,...
            'dampFac',.95,...
            'delay',0,...
            'step',10,... 
            'gm_minvar',0,...
            'bgm_alg',0,...
            'gm_step',1,...
            'GM',[],...
            'initVar',[],...
            'initSpar',[]);       
        % sureParams:
            % method: selects the method to minimize SURE
                % (1) a bisection search method on Gaussian Mixture SURE
                % (2) gradient descent on Gaussian Mixture SURE
                % (3) approximate gradient descent 
            % damp: stepsize between new and old lambda (1 = no amount
            % of damping)
            % decayDamp: decrease damping parameter (stepsize) by 1-dampFac
            % damp_fac: factor to reduce damp by
            % delay: number of gamp iterations between lambda tuning 
            % step: initial step size in approximate gradient descent 
            % gm_minvar: set to 1 to force minimum GM component variance of rvar
            % gm_alg: set to 1 to learn Bernoulli-GM on X, then convolve
            % with N(0,rvar) to obtain distribution on rhat, otherwise,
            % learn Gaussian mixture distribution without the Bernoulli
            % component
            % gm_step: initial stepsize for gradient descent on Gaussian
            % Mixture SURE
            % GM: a place to store the GM approximation to rhat 
          
    end
    
    properties (Hidden)
        mixWeight = 1;     % Mixture weight (used for EM tuning w/ SparseScaEstimIn)
        lam_left = [];     
        lam_right = [];
        tune_it = 0;
    end
    
    methods
        % Constructor
        function obj = SoftThreshEstimIn(lambda, maxSumVal, varargin)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.lambda = lambda;
                if nargin >= 2 && ~isempty(maxSumVal) && isscalar(maxSumVal)
                    obj.maxSumVal = logical(maxSumVal);
                    
                    if nargin >= 3
                        for i = 1:2:length(varargin)
                            obj.(varargin{i}) = varargin{i+1};
                        end
                    end
                end
            end
        end
        
        function set.disableTune(obj, flag)
            assert(isscalar(flag), ...
                'SoftThreshEstimIn: disableTune must be a logical scalar');
            obj.disableTune = flag;
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = 0;
            if obj.maxSumVal
                %var0 = 5e-4; % a reasonable constant?
                var0 = 2./(obj.lambda.^2);
                valInit = -inf; % should fix this...
            else
                var0 = 2./(obj.lambda.^2);
                valInit = 0; % should fix this...
            end
        end
        
        % Carry out soft thresholding
        function [xhat,xvar,val] = estim(obj, rhat, rvar)
            if ~obj.maxSumVal
                % Compute sum-product GAMP updates
                
                % To avoid numerical problems (0/0) when evaluating
                % ratios of Gaussian CDFs, impose a firm cap on the
                % maximum value of entries of rvar
                rvar = min(rvar, 700);
                
                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean and variance depend
                sig = sqrt(rvar);                       	% Gaussian prod std dev
                muL = rhat + obj.lambda.*rvar;          	% Lower integral mean
                muU = rhat - obj.lambda.*rvar;          	% Upper integral mean
                muL_over_sig = muL ./ sig;
                muU_over_sig = muU ./ sig;
                cdfL = normcdf(-muL_over_sig);              % Lower cdf
                cdfU = normcdf(muU_over_sig);               % Upper cdf
                cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
                SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*rvar) ) .* ...
                    cdfRatio;
                NaN_Idx = isnan(SpecialConstant);        	% Indices of trouble constants
                
                % For the "trouble" constants (those with muL's and muU's
                % that are too large to give accurate numerical answers),
                % we will effectively peg the special constant to be Inf or
                % 0 based on whether muL dominates muU or vice-versa
                SpecialConstant(NaN_Idx & (-muL >= muU)) = Inf;
                SpecialConstant(NaN_Idx & (-muL < muU)) = 0;
                
                % Compute the ratio normpdf(a)/normcdf(a) for
                % appropriate upper- and lower-integral constants, a
                RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
                RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));
                
                % Now compute the first posterior moment...
                xhat = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                    (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* ...
                    (muU + sig.*RatioU);
                
                % ...and second central posterior moment
                varL = rvar .* (1 - RatioL.*(RatioL - muL_over_sig));
                varU = rvar .* (1 - RatioU.*(RatioU + muU_over_sig));
                meanL = muL - sig.*RatioL;
                meanU = muU + sig.*RatioU;
                SecondMoment = (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                    (varL + meanL.^2) + (1 ./ (1 + SpecialConstant)) .* ...
                    (varU + meanU.^2);
                xvar = SecondMoment - xhat.^2;
                % *********************************************************
                
                % Perform EM parameter tuning, if desired
                if obj.autoTune && ~obj.disableTune
                    
                    if (obj.counter>0), % don't tune yet
                        obj.counter = obj.counter-1; % decrement counter
                    else % tune now
                        
                        % Start by computing E[|x_n| | y]...
                        mu = (1 ./ (1 + SpecialConstant)) .* (muU + sig.*RatioU) ...
                            - (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                            (muL - sig.*RatioL);
                        [N, T] = size(xhat);
                        
                        %Average over all elements, per column, or per row
                        switch obj.tuneDim
                            case 'joint'
                                lambda = N*T / sum(obj.mixWeight(:).*mu(:));
                            case 'col'
                                lambda = repmat(N ./ sum(obj.mixWeight.*mu), ...
                                    [N 1]);
                            case 'row'
                                lambda = repmat(T ./ sum(obj.mixWeight ...
                                    .*mu, 2), [1 T]);
                            otherwise
                                error('Invalid tuning dimension in SoftThreshEstimIn'); 
                        end
                        
                        if any( lambda(:) <= 0 | isnan(lambda(:)) )
                            warning('EM update of lambda was negative or NaN...ignoring')
                            lambda = obj.lambda;
                        end
                        obj.lambda = lambda;
                        
                    end
                end
                
                % Lastly, compute negative KL divergence:
                % \int_x p(x|r) log(p(x)/p(x|r)) dx
                
                %                 % Old way of computing.  It handles difficult cases
                %                 % incorrectly.
                %                 NormConL = obj.lambda/2 .* ...              % Mass of lower integral
                %                     exp( (muL.^2 - rhat.^2) ./ (2*rvar) ) .* cdfL;
                %                 NormConU = obj.lambda/2 .* ...              % Mass of upper integral
                %                     exp( (muU.^2 - rhat.^2) ./ (2*rvar) ) .* cdfU;
                %                 NormCon = NormConL + NormConU;      % Posterior normaliz. constant recip.
                %                 NormCon(isnan(NormCon)) = 1;        % Not much we can do for trouble ones
                %                 NormCon(NormCon == Inf) = 1;
                
                if (nargout >= 3)
                    %Calculate lower and Upper non-shared integration factors
                    CL = erfcx(muL_over_sig/sqrt(2));
                    CU = erfcx(-muU_over_sig/sqrt(2));
                    % The individual terms can still be infinite. For these
                    % cases use the approximation erfc(x) = 1/sqrt(pi)/x for
                    % large x
                    I = find(isinf(CL));
                    CL(I) = 1./(sqrt(pi/2)*abs(muL_over_sig(I)));
                    I = find(isinf(CU));
                    CU(I) = 1./(sqrt(pi/2)*abs(muU_over_sig(I)));
                    
                    % Calculate the log scale factor
                    logNormCon = log(obj.lambda/4) - rhat.^2./(2*rvar) + log(CL + CU);
                    
                    val = logNormCon + ...
                        0.5*(log(2*pi*rvar) + ...
                        (xvar + (xhat - rhat).^2)./rvar);
                    
                    %Old fix, no longer needed
                    %val(val == -Inf) = -1e4;    % log(NormCon) can == -Inf
                end
                
            else %if obj.maxSumVal
                
                % tune lambda to minimize the SURE of the estimator's MSE.
                % more details are in "Sparse multinomial logistic 
                
                if obj.autoTune && ~obj.disableTune
                    
                    debug = 0; % applies to all modes, produces diagnostic plots at every GAMP iteration

                    if (obj.counter>0), % don't tune yet
                        obj.counter = obj.counter-1; % decrement counter
                    else % tune now
                        
                        obj.tune_it = obj.tune_it+1;

                        [N,T] = size(rhat);
                        
                        switch obj.tuneDim
                            case 'joint'
                                dim = 1;
                                % assume uniform variance
                                c = mean(rvar(:));                     
                            case 'col'
                                dim = T;
                                % assume uniform variance
                                c = mean(rvar,1);
                            case 'row'
                                dim = N;
                                % assume uniform variance
                                c = mean(rvar,2);
                        end
                        if length(obj.sureParams.damp) ~= dim
                            obj.sureParams.damp = obj.sureParams.damp(1)*ones(dim,1);
                        end
                        if length(obj.sureParams.step) ~= dim
                            obj.sureParams.step = obj.sureParams.step(1)*ones(dim,1);
                        end
                        if length(obj.sureParams.gm_step) ~= dim
                            obj.sureParams.gm_step = obj.sureParams.gm_step(1)*ones(dim,1);
                        end
                        if size(obj.sureParams.GM,1) ~= dim
                            obj.sureParams.GM = cell(dim, 1);
                        end
                        
                        % loop over tuning dimensions (joint, row or
                        % column)
                        for t = 1:dim
                            
                            % lam_max is the smallest value of lambda which
                            % will set every x to zero
                            % lam_min is the smallest value of lambda which
                            % empricial SURE is still valid, GM sure may be
                            % different
                            switch obj.tuneDim
                                case 'joint'
                                    % find initial lambda
                                    lambda0 = mean(obj.lambda);
                                    % compute lambda max
                                    lam_max = min(max(abs(rhat)./c));
                                    lam_min = max(min(abs(rhat)./c));
                                    idx = 1:(N*T);
                                case 'col'
                                    % find initial lambda
                                    lambda0 = mean(obj.lambda(:,t));
                                    % compute lambda max
                                    lam_max = max(abs(rhat(:,t))./c(t));
                                    lam_min = min(abs(rhat(:,t))./c(t));
                                    idx = (1:N) + (t-1)*N;
                                case 'row'
                                    % find initial lambda
                                    lambda0 = mean(obj.lambda(t,:));
                                    % compute lambda max
                                    lam_max = max(abs(rhat(t,:))./c(t));
                                    lam_min = min(abs(rhat(t,:))./c(t));
                                    idx = (1:N:(N*T)) + (t-1);
                            end
                            
                            % select method to optimize SURE
                            switch obj.sureParams.method
                                case 1 % Gaussian mixture with bisection search on gradient
                                    lambda = obj.minSureGMbisect(lambda0, rhat(idx), rvar(idx), c(t), lam_max, lam_min, t, obj.tune_it, debug);
                                case 2 % Gaussian mixture with gradient descent
                                    lambda = obj.minSureGMgrad(lambda0, rhat(idx), rvar(idx), c(t), lam_max, lam_min, t, obj.tune_it, debug);
                                case 3 % approximate gradient descent
                                    lambda = obj.minSureGrad(lambda0, rhat(idx), c(t), lam_max, lam_min, t, obj.tune_it, debug);
                                    
                            end
                            
                            % implement damping on lambda
                            damp = obj.sureParams.damp(t); % damping on lambda update 
                            lambda = 10^(damp * log10(lambda) + (1-damp)*log10(lambda0));

                            % set new lambda
                            switch obj.tuneDim
                                case 'joint'
                                    obj.lambda = lambda;
                                case 'col'
                                    if numel(obj.lambda) == 1
                                        obj.lambda = obj.lambda * ones(N,T);
                                    end
                                    obj.lambda(:,t) = lambda * ones(N,1);
                                case 'row'
                                    if numel(obj.lambda) == 1
                                        obj.lambda = obj.lambda * ones(N,T);
                                    end
                                    obj.lambda(t,:) = lambda * ones(1,T);
                            end
                            
                            if obj.sureParams.decayDamp 
                                obj.sureParams.damp(t) = obj.sureParams.damp(t) * obj.sureParams.dampFac; 
                            end
                        end % for t
                        
                        obj.counter = obj.sureParams.delay;
                        
                    end % tune now
                    
                end % autoTune
                
                % Compute max-sum GAMP updates
                if rvar<0, warning('rvar is negative!'); end;
                
                %Compute the thresh
                thresh = obj.lambda .* rvar;
                
                %Estimate the signal
                xhat = max(0,abs(rhat)-thresh) .* sign(rhat);

                %Estimate the variance
                %xvar = rvar .* (mean(double(abs(xhat) > 0))*ones(size(xhat)));
                xvar = rvar .* (abs(xhat) > 0);

                %Debias if needed
                if obj.debias
                   rhat_on = rhat(find(xhat~=0));
                   if ~isempty(rhat_on)
                       scale = 1 + thresh*sum(abs(rhat_on)-thresh)/ ...
                                          sum((abs(rhat_on)-thresh).^2);
                       xhat = scale*xhat;
                       xvar = scale*xvar;
                   end
                end

                if (nargout >= 3)
                    %Output negative cost
                    %val = -1*obj.lambda*abs(rhat);
                    val = -1*obj.lambda.*abs(xhat);	% seems to work better
                end
            end
        end
        
        % Computes p(y) for y = x + v, with x ~ p(x), v ~ N(0,yvar)
        function py = plikey(obj, y, yvar)
            mu = y;
            sig2 = yvar;
            sig = sqrt(sig2);                           % Gaussian prod std dev
            muL = mu + obj.lambda.*sig2;                % Lower integral mean
            muU = mu - obj.lambda.*sig2;                % Upper integral mean
            muL_over_sig = muL ./ sig;
            muU_over_sig = muU ./ sig;
            erfsum = erfcx((1/sqrt(2))*muL_over_sig) + ...
                erfcx((-1/sqrt(2))*muU_over_sig);
            erfsum(erfsum == Inf) = realmax;
            py = 0.25*obj.lambda .* exp(-(y.^2) ./ (2*yvar)) .* erfsum;
        end
        
        function logpy = loglikey(obj, y, yvar)
            mu = y;
            sig2 = yvar;
            sig = sqrt(sig2);                           % Gaussian prod std dev
            muL = mu + obj.lambda.*sig2;                % Lower integral mean
            muU = mu - obj.lambda.*sig2;                % Upper integral mean
            muL_over_sig_sqrt2 = muL ./ (sig*sqrt(2));
            muU_over_sig_sqrt2 = muU ./ (sig*sqrt(2));
            log_erfsum = log( erfcx(muL_over_sig_sqrt2) + ...
                erfcx(-muU_over_sig_sqrt2) );
            
            % handle erfcx inputs smaller than THRESH in a special way
            % noting that erfcx(x) ~= 2*exp(x^2) for x<<0
            % and that both inputs muL and -muU can't be simultaneously small
            THRESH = -20;
            indx = find( muL_over_sig_sqrt2 < THRESH );
            log_erfsum(indx) = log(2)+muL_over_sig_sqrt2(indx).^2;
            indx = find( -muU_over_sig_sqrt2 < THRESH );
            log_erfsum(indx) = log(2)+muU_over_sig_sqrt2(indx).^2;
            
            % compute log likelihood
            logpy = log(0.25*obj.lambda) -(y.^2)./(2*yvar) + log_erfsum;
            
        end
        
        % Bisection Search method to minimize GM-SURE function
        function lambda = minSureGMbisect(obj, lambda0, rhat, rvar, c, lam_max, lam_min, t, tune_it, debug)
            
            num_it = 5; % number of bisections
            lambda_hist = nan(num_it+1,1);
            
            % learn gaussian mixture
            if obj.sureParams.bgm_alg
                L = 4; % number of GM components (after conv. with rvar)
                gm = obj.embgm(rhat,rvar,c,L,t);
            else
                L = 3; % number of GM components
                gm = obj.emgm(rhat,c,L,t);
            end

            % we assume the GM SURE function's gradient begins negative and
            % has only one root. Thus, if the gradient at lambda_max is
            % negative, we skip the bisection method and simply set lambda
            % = lambda_max.
            
            gradient = obj.gmSUREgrad(lam_max,c,gm);
            
            if gradient < 0
                % minimum not contained in "effective" lambda, terminate
                lambda = lam_max;
                lambda_hist(end) = lam_max;
            else
                % perform bisection search to find minimum
                
                % first, find initial bounds using the final bounds from 
                % the previous iteration and backtracking if necessary.
                
                % set initial search 
                if isempty(obj.lam_left)
                    obj.lam_left = lam_min;
                end
                if isempty(obj.lam_right)
                    obj.lam_right = lam_max;
                end
                
                gradient_left = obj.gmSUREgrad(obj.lam_left,c,gm);
                gradient_right = obj.gmSUREgrad(obj.lam_right,c,gm);
                
                if gradient_left <= 0 && gradient_right >= 0
                    % case 1: gradient left is negative, gradient right is
                    % positive, indicating the root is between them

                    % do nothing here...
                    
                elseif gradient_left >= 0 && gradient_right >= 0
                    % case 2: gradient left and right are positive,
                    % indicating the root is to the left
                    
                    stop = 0;
                    delta = abs(obj.lam_left - obj.lam_right);
                    scale = 1;
                    it = 1;
                    while ~stop
                        it = it + 1;
                        obj.lam_right = obj.lam_left;
                        obj.lam_left = max(obj.lam_left - scale * delta, lam_min);
                        scale = 2*scale;
                        % check gradient of new lam_right
                        gradient_left = obj.gmSUREgrad(obj.lam_left,c,gm);
                        if gradient_left < 0 || obj.lam_left == lam_min || it == 20
                            stop = 1;
                        end
                    end
                    
                elseif gradient_left <= 0 && gradient_right <= 0
                    % case 3: gradient left and right are negative,
                    % indicating the root is to the right
                    
                    stop = 0;
                    delta = abs(obj.lam_left - obj.lam_right);
                    scale = 1;
                    it = 1;
                    while ~stop
                        it = it + 1;
                        obj.lam_left = obj.lam_right;
                        obj.lam_right = min(obj.lam_right + scale * delta, lam_max);
                        scale = 2*scale;
                        % check gradient of new lam_right
                        gradient_right = obj.gmSUREgrad(obj.lam_right,c,gm);
                        if gradient_right > 0 || obj.lam_right == lam_max || it == 20
                            stop = 1;
                        end
                    end
                else
                    % in this case, our unique root assumption of GM-SURE
                    % was violated, so refrain from tuning lambda this
                    % iteration
                    lambda = min(lambda0,lam_max);
                    return;
                end

                % Bisection search 
                for it = 1:num_it;
                    
                    % bisect in log-domain
                    lambda = 10^(log10(obj.lam_left*obj.lam_right)/2);
                    lambda_hist(it) = lambda;
                    
                    % compute gradient
                    gradient = obj.gmSUREgrad(lambda,c,gm);
                    
                    if gradient < 0
                        obj.lam_left = lambda;
                    else
                        obj.lam_right = lambda;
                    end
                end
                
                % one final bisection...
                lambda = 10^(log10(obj.lam_left*obj.lam_right)/2);
                lambda_hist(end) = lambda;
                
            end

            if debug
                figure(100);clf;
                % plot samples
                semilogx(lambda_hist(1:end-1), obj.gmSUREcost(lambda_hist(1:end-1),c,gm),'bo')
                hold on
                % plot final lambda
                semilogx(lambda_hist(end), obj.gmSUREcost(lambda_hist(end),c,gm),'rs')
                % plot cost
                lgrid = logspace(log10(lam_min),log10(lam_max),1e3);
                cost = obj.gmSUREcost(lgrid,c,gm);
                semilogx(lgrid,cost,'g')
                hold off
                xlabel('lambda')
                ylabel('SURE')
                title(sprintf('minimization of objective function; gamp it = %d',tune_it))
                legend('samples','final lambda','sure cost')
                figure(101);clf;
                semilogy(lambda_hist)
                xlabel('iter')
                ylabel('lambda')
                title('lambda vs bisection iteration')
                figure(102);clf;
                histnorm(rhat,40)
                hold on
                x = linspace(.9*min(rhat), 1.1*max(rhat), 100);
                y = zeros(size(x));
                for l = 1:L
                    y = y + gm.omega(l) * normpdf(x, gm.theta(l), sqrt(gm.phi(l)));
                end
                plot(x,y,'g')
                hold off
                title('GM fit to rhat')
                legend('rhat','GM')
                drawnow;
                pause;
            end
            
        end
        
        % Gradient descent method to minimize GM SURE
        function lambda = minSureGMgrad(obj, lambda0, rhat, rvar, c, lam_max, lam_min, t, tune_it, debug)
            
            lambda0 = min(lambda0, lam_max);
            lambda = lambda0;
            
            maxit = 100;
            lambda_hist = nan(1,maxit);
            tol = 0;
            
            alpha = obj.sureParams.gm_step(t); % gradient descent initial step size
            
            % learn gaussian mixture
            if obj.sureParams.bgm_alg
                L = 4; % number of GM components
                gm = obj.embgm(rhat,rvar,c,L,t);
            else
                L = 3; % number of GM components
                gm = obj.emgm(rhat,c,L,t);
            end
            
            grad_old = 0;
            
            % apply gradient projection
            for it = 1:maxit
                lam_old = lambda;
                gradient = obj.gmSUREgrad(lambda,c,gm);
                if gradient < 0 && lambda == lam_max
                    break
                end
                lambda = max(min(lambda - alpha * gradient, lam_max),lam_min);
                
                if sign(gradient)*sign(grad_old) == 1
                    % increase step
                    stepmax = 1e6;
                    alpha = min(alpha*1.1, stepmax);
                else
                    % descrease step
                    alpha = alpha*.5;
                end
                grad_old = gradient;
                lambda_hist(it) = lambda;
                if abs(lam_old - lambda)/(lam_old+eps) < tol
                    break
                end
            end
            
            obj.sureParams.gm_step(t) = alpha;
            
            if debug
                figure(100);clf;
                % plot samples
                semilogx(lambda_hist(1:end-1), obj.gmSUREcost(lambda_hist(1:end-1),c,gm),'bo')
                hold on
                % plot final lambda
                semilogx(lambda_hist(end), obj.gmSUREcost(lambda_hist(end),c,gm),'rs')
                % plot cost
                lgrid = logspace(log10(lam_min),log10(lam_max),1e3);
                cost = obj.gmSUREcost(lgrid,c,gm);
                semilogx(lgrid,cost,'g')
                hold off
                xlabel('lambda')
                ylabel('SURE')
                title(sprintf('minimization of objective function; gamp it = %d',tune_it))
                legend('samples','final lambda','sure cost')
                figure(101);clf;
                semilogy(lambda_hist)
                xlabel('iter')
                ylabel('lambda')
                title('lambda vs iteration')
                figure(102);clf;
                histnorm(rhat,40)
                hold on
                x = linspace(.9*min(rhat), 1.1*max(rhat), 100);
                y = zeros(size(x));
                for l = 1:L
                    y = y + gm.omega(l) * normpdf(x, gm.theta(l), sqrt(gm.phi(l)));
                end
                plot(x,y,'g')
                hold off
                title('GM fit to rhat')
                legend('rhat','GM')
                drawnow;
                pause;
            end
            
        end
        
        % Approximate gradient descent method to minimize empirical SURE
        function lambda = minSureGrad(obj, lambda0, rhat, c, lam_max, lam_min, t, tune_it, debug)
            
            % perform optimization via approx grad
            % descent as described in "Parameterless
            % Optimal Approximate Message Passing" by
            % A. Mousavi, A. Maleki, and R. Baraniuk.
            
            rhat2 = rhat.^2;
            N = numel(rhat);
            
            % options
            %dt = 1;  % for empirical gradient calc (recommended .05-.5)
            maxit = 50; % max number of iterations
            minit = 10;  % min number of iterations
            
            % history over the course of a single
            % GAMP iteration
            lambda_hist = nan(maxit,1);
            step_hist = lambda_hist;
            cost_hist = lambda_hist;
            grad_hist = lambda_hist;
            grad_old = 0;
            lambda = lambda0;
            
            rr = sort(abs(rhat)/c); % sorted points of major change in SURE cost
            tol = 1e-4; % convergence tolerance
            step = obj.sureParams.step(t);
            
            for it=1:maxit
                % compute empirical gradient
                cost = obj.SUREcost(lambda,rhat,rhat2,c);
                cost_hist(it) = cost;
                
                [~,indx]=min(abs( lambda - rr ));
                if indx==1 % detect special case
                    points = 10;
                    cost_left = obj.SUREcost(rr(1:points)+eps,rhat,rhat2,c);
                    [~,p_opt] = min(cost_left);
                    lambda = rr(p_opt);
                    break;
                end
                dt = rr(min(N,indx + 5)) - rr(max(1,indx-5));
                
                grad = (obj.SUREcost(rr(min(N,indx+5)),rhat,rhat2,c) ...
                    - obj.SUREcost(rr(max(1,indx-5)),rhat,rhat2,c))/dt;
                
                grad_hist(it) = grad;
                step = step + 0.1*step*sign(grad)*sign(grad_old);
                if lambda - step*grad < 0
                    step = 0.5*lambda/grad;
                end
                step_hist(it) = step;
                grad_old = grad;
                
                % gradient projection (lambda can't be negative or larger than lambda max)
                lambda_old = lambda;
                lambda = max(lam_min,min(lambda-step*grad, lam_max));
                lambda_hist(it) = lambda;
                
                % check for convergence
                if (it>minit) && (abs(lambda - lambda_old)/lambda_old < tol)
                    break
                end
                
            end
            
            obj.sureParams.step(t) = step;
            
            if debug
                lgrid = logspace(log10(lam_min),log10(lam_max),1e3);
                cost = obj.SUREcost(lgrid,rhat,rhat2,c);
                figure(100);clf;
                semilogx(lgrid,cost,'g')
                hold on
                plot(lambda0,obj.SUREcost(lambda0,rhat,rhat2,c),'go')
                plot(lambda_hist,obj.SUREcost(lambda_hist,rhat,rhat2,c),'bo')
                plot(lambda,obj.SUREcost(lambda,rhat,rhat2,c),'rs')
                hold off
                xlabel('lambda')
                ylabel('sure val')
                title(sprintf('gamp it = %d',tune_it))
                legend('objective','lam0','lam final')
                
                figure(101);clf;
                subplot(411)
                semilogy(lambda_hist,'.-')
                xlabel('iter')
                ylabel('lambda')
                subplot(412)
                semilogy(step_hist,'.-')
                xlabel('iter')
                ylabel('step')
                subplot(413)
                plot(grad_hist,'.-')
                xlabel('iter')
                ylabel('grad')
                subplot(414)
                plot(cost_hist,'.-')
                xlabel('iter')
                ylabel('cost')
                drawnow;
                pause;
            end
            
        end
        
        
        function scost = SUREcost(obj,lambda,rhat,rhat2,c)
            % compute SURE cost using empirical average
            n = numel(rhat);
            scost = nan(size(lambda));
            for ll = 1:numel(lambda)
                scost(ll) = sum(obj.g2(lambda(ll),rhat(:),rhat2(:),c) + 2*c*obj.gp(lambda(ll),rhat(:),c))/n;
            end
        end
        
        function val = gp(~,t,r,c)
            % g-prime
            val = zeros(size(r));
            val(abs(r)<t*c) = -1;
        end
        
        function r2 = g2(~,t,r,r2,c)
            %g-squared
            r2(abs(r)>=t*c) = (t.^2).*(c.^2);
        end
        
        function gm = emgm(obj,rhat,c,L,t)
            % use EM to fit GM distribution to rhat
            
            if L~=3
                error('L must equal 3')
            end
            
            % use EM to fit GM to rhat
            rhat = rhat(:);
            N = numel(rhat);
            maxit = 8;
            tol = 1e-3;
            
            % determine whether to warmstart
            warmstart = 1;
            if isempty(obj.sureParams.GM{t})
                warmstart = 0;
            else
               if all(abs(obj.sureParams.GM{t}.theta) < .1) %% any(obj.sureParams.GM{t}.omega < 1e-4) || 
                  warmstart = 0; 
               end
            end
            
            % random mean initialization/warmstarting
            if ~warmstart  % isempty(obj.sureParams.GM{t}) 
                omega = ones(L,1)/L;
                phi = var(rhat(:))*omega;
                theta = [-0.3333 0 0.3333];
            else
                omega = obj.sureParams.GM{t}.omega;
                theta = obj.sureParams.GM{t}.theta;
                phi = obj.sureParams.GM{t}.phi;
            end
            
            p = nan(N,L);
            twopi = 2*pi;
            g = @(x, m, s) 1/sqrt(twopi * s) * exp(-1/2/s * (x - m).^2);
            omega_hist = nan(L,maxit+1);
            theta_hist = nan(L,maxit+1);
            phi_hist = nan(L,maxit+1);
            
            omega_hist(:,1) = omega';
            theta_hist(:,1) = theta';
            phi_hist(:,1) = phi;
            
            for it = 1:maxit
                
                omega_old = omega;
                theta_old = theta;
                phi_old = phi;
                
                % E-step (slow...)
                for l = 1:L
                    p(:,l) = omega(l)*g(rhat, theta(l), phi(l));
                end
                p(isnan(p)) = 0;
                
                % sum to one
                p = bsxfun(@times, p, 1./sum(p,2));
                
                % M-step
                theta = sum(bsxfun(@times, p, rhat))./sum(p);
                theta(isnan(theta)) = 0;
                
                for l = 1:L
                    phi(l) = sum(p(:,l).*(rhat - theta(l)).^2)/sum(p(:,l));
                end
                phi(isnan(phi)) = 1;
                if obj.sureParams.gm_minvar
                    phi = max(phi, c);
                end
                
                omega = sum(p)/N;
                omega = omega/sum(omega);
                
                omega_hist(:,it+1) = omega';
                theta_hist(:,it+1) = theta';
                phi_hist(:,it+1) = phi;
                
                % check for convergence
                if norm(omega(:)-omega_old(:))/norm(omega_old+eps) < tol && norm(theta(:)-theta_old(:))/norm(theta_old+eps) < tol && norm(phi(:)-phi_old(:))/norm(phi_old+eps) < tol
                    break
                end
            end
            
            gm.omega = omega(:);
            gm.theta = theta(:);
            gm.phi = phi(:);
            
            obj.sureParams.GM{t} = gm;
            
        end
        
        function gm = embgm(obj, rhat, rvar, c, L, t)
            % use EM to fit Bern-GM to xhat (which corresponds to GM on
            % rhat)
            
            rhat = rhat(:);
            rvar = rvar(:);
            N = numel(rhat);
            
            % random mean initialization/warmstarting
            if isempty(obj.sureParams.GM{t})
                lambda = .5; %#ok<*PROP>
                custom_scale = 1;
                load('inits.mat');
                omega = init(L-1).active_weights;
                theta = init(L-1).active_mean;
                phi = init(L-1).active_var;
                if ~isempty(obj.sureParams.initVar)
                    theta = theta*sqrt(12*obj.sureParams.initVar)*custom_scale;
                    phi = phi*12*obj.sureParams.initVar*custom_scale; 
                end
                if ~isempty(obj.sureParams.initSpar)
                   lambda = obj.sureParams.initSpar;
                end
            else
                omega = obj.sureParams.GM{t}.omega;
                theta = obj.sureParams.GM{t}.theta;
                phi = obj.sureParams.GM{t}.phi;
                
                % "deconvolve" with N(0,rvar)
                lambda = 1 - omega(end);
                omega = omega(1:end-1)/lambda;
                theta = theta(1:end-1);
                phi = phi(1:end-1) - phi(end);
                
            end
           
            % expand
            one = ones(N,1,L-1);
            omega = bsxfun(@times, reshape(omega, 1,1,L-1), one);
            theta = bsxfun(@times, reshape(theta, 1,1,L-1), one);
            phi = bsxfun(@times, reshape(phi, 1,1,L-1), one);
            
            D_l = zeros(N,1,L-1); a_nl = zeros(N,1,L-1);
            gamma = zeros(N,1,L-1); nu = zeros(N,1,L-1);
            
            abs_rhat2_over_rvar = abs(rhat).^2./rvar;
            %Evaluate posterior likelihoods
            for i = 1:L-1
                post_var_scale = rvar+phi(:,:,i)+eps;
                rvar_over_post_var_scale = rvar./post_var_scale;
                D_l(:,:,i) = lambda*omega(:,:,i)./sqrt(post_var_scale)...
                    .*exp(-abs(theta(:,:,i)-rhat).^2./(2*post_var_scale));
                gamma(:,:,i) = (rhat.*phi(:,:,i)+rvar.*theta(:,:,i))./post_var_scale; 
                nu(:,:,i) = rvar_over_post_var_scale.*phi(:,:,i);
                a_nl(:,:,i) = sqrt(rvar_over_post_var_scale).*omega(:,:,i)...
                    .*exp((abs(rhat-theta(:,:,i)).^2./abs(post_var_scale)-abs_rhat2_over_rvar)./(-2));  
            end;
            
            %Find posterior that the component x(n,t) is active
            a_n = lambda./(1-lambda).*sum(a_nl,3);
            a_n = 1./(1+a_n.^(-1));
            a_n(isnan(a_n)) = 0.001;
            
            lambda = sum(a_n)/N*ones(N,1);
            
            %Find the Likelihood that component n,t belongs to class l and is active
            E_l = bsxfun(@times, D_l, 1./(sum(D_l,3)+(1-lambda)./sqrt(rvar).*exp(-abs_rhat2_over_rvar/2))); 
            
            %Ensure real valued probability
            E_l(isnan(E_l)) = 0.999;
            
            %Update parameters based on EM equations
            N_l = sum(E_l);
            theta = resize(sum(E_l.*gamma)./N_l,N,1,L-1);
            phi = resize(sum(E_l.*(nu+abs(gamma-theta).^2))./N_l,N,1,L-1);
            omega = N_l/N;
            omega = omega./repmat(sum(omega, 3), [1, 1, L-1]);
            omega = resize(omega,N,1,L-1);
            
            % convolve with N(0,rvar)
            lambda = squeeze(lambda(1,1,:));
            weights = squeeze(omega(1,1,:));
            means = squeeze(theta(1,1,:));
            variances = squeeze(phi(1,1,:));
            
            gm.omega = [lambda * weights; 1-lambda];
            gm.theta = [means;0];
            gm.phi = [variances + c;c];
            
            obj.sureParams.GM{t} = gm;
            
        end
        
        function scost = gmSUREcost(~,lambda,c,gm)
            % compute SURE value using statisical expectation instead of
            % empirical average
            
            omega = gm.omega(:);
            theta = gm.theta(:);
            phi = gm.phi(:);
            
            L = length(omega);
            
            scost = nan(size(lambda));
            c1 = scost;
            c2 = scost;
            c3 = scost;
            
            mu = sum(omega.*theta);
            va = sum(omega.*((theta - mu).^2 + phi)); %#ok<NASGU>
            
            prllc = @(tau, omega, theta, phi) sum(omega.*normcdf(tau, theta, sqrt(phi)));
            prglc = @(tau, omega, theta, phi) 1 - prllc(tau, omega, theta, phi);
            
            for ll = 1:numel(lambda)
                
                tau = lambda(ll)*c;
                val1 = max(0,prglc(tau, omega, theta, phi));
                val2 = max(0,prllc(-tau,omega, theta, phi));
                
                Er2 = 0;
                for l = 1:L
                    b = (tau - theta(l))/sqrt(phi(l));
                    a = (-tau - theta(l))/sqrt(phi(l));
                    phia = normpdf(a);
                    phib = normpdf(b);
                    Phiab = normcdf([a,b]);
                    Phia = Phiab(1);Phib = Phiab(2);
                    %                     Phib = normcdf(b);
                    Z = Phib - Phia;
                    va = phi(l) * ( 1 + (a*phia - b*phib)/Z - ((phia - phib)/Z)^2);
                    mu = theta(l) + (phia - phib)/Z*sqrt(phi(l));
                    Er2 = Er2 + omega(l) * Z * (va + mu^2);
                end
                
                c1(ll) = tau^2 * val1;
                c2(ll) = Er2 - 2*c*(1 - val1 - val2);
                c3(ll) = tau^2 * val2;
                
                scost(ll) =  c1(ll) + c2(ll) + c3(ll);
                
            end
            
        end
        
        function sgrad = gmSUREgrad(~,lambda,c,gm)
            % compute gradient of GM-SURE cost
            
            sgrad = nan(size(lambda));
                        
            % evaluate Gradient
            prllc = @(tau, omega, theta, phi) sum(omega.*normcdf(tau, theta, sqrt(phi)));
            prglc = @(tau, omega, theta, phi) 1 - prllc(tau, omega, theta, phi);
            pprllc = @(tau, omega, theta, phi) sum(omega.*normpdf(tau, theta, sqrt(phi)));
            
            grad1 = @(gm, lam, c) ...
                2 * lam * c^2 .* (prglc(lam*c, gm.omega, gm.theta, gm.phi) + prllc(-lam*c, gm.omega, gm.theta, gm.phi));
            
            grad2 = @(gm, lam, c) ...
                - 2 * c^2 * (pprllc(lam*c, gm.omega, gm.theta, gm.phi) + pprllc(-lam*c, gm.omega, gm.theta, gm.phi));
            
            for ll = 1:numel(lambda)
                sgrad(ll) = grad1(gm, lambda(ll), c) + grad2(gm, lambda(ll), c);
            end
            
        end
        
        
    end
    
end

