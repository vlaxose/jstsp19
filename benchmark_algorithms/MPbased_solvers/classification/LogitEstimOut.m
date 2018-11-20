% CLASS: LogitEstimOut
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: EstimOut
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The LogitEstimOut class defines a scalar observation channel, p(y|z),
%   that constitutes a logit binary classification model, i.e., y is an
%   element of the set {0,1}, z is a real number, and
%           Pr{y=0|z} = 1 - Pr{y=1|z} = 1 / (1+exp(scale*z)),
%   where scale is a scalar parameter that determines the relative shape of
%   the sigmoidal curve.
%
%   LogitEstimOut can operate in either of two different modes
%   corresponding to two different flavors of GAMP: sum-product mode (for
%   MMSE estimation) or max-sum mode (for MAP estimation).  In both modes,
%   approximations are required for the purpose of tractability.  In
%   sum-product mode, an intractable integral is replaced by a numerical
%   integration.  The number of discrete points included in the integration
%   is determined by the parameter Npts, while the range of discrete points
%   is the interval (-Wmax, Wmax), where Wmax is a real-valued parameter.
%   In max-sum mode, the maximization of a concave function is approximated
%   by a maximization of a second-order Taylor approximation of that
%   function.  An advantage of the max-sum mode is that it is quicker than
%   the alternative sum-product version.
%
% PROPERTIES (State variables)
%   y           An M-by-1 array of binary ({0,1}) class labels for the
%               training data, where M is the number of training data
%               points
%   scale       Scalar controlling the shape of the sigmoidal function.
%               Higher values produce sharper sigmoids [Default: 1]
%   npts        Number of points used in discrete integration, for
%               sum-product GAMP [Default: 100]
%   wmax        Range of points included in discrete integration, [-wmax,
%               wmax], for sum-product GAMP [Default: 4]
%   maxSumVal   Perform MMSE estimation (false) or MAP estimation (true)?
%               [Default: false]
%
% METHODS (Subroutines/functions)
%   LogitEstimOut(y)
%       - Default constructor.  Assigns remaining properties to default
%         values
%   LogitEstimOut(y, scale)
%       - Optional constructor.  Sets both y and scale.
%   LogitEstimOut(y, scale, npts)
%       - Optional constructor.  Sets y, scale, and npts.
%   LogitEstimOut(y, scale, npts, wmax)
%       - Optional constructor.  Sets y, scale, npts, and wmax.
%   LogitEstimOut(y, scale, npts, wmax, maxSumVal)
%       - Optional constructor.  Sets y, scale, npts, wmax, and maxSumVal.
%   estim(obj, zhat, zvar)
%       - Provides the posterior mean and variance of a variable z when
%         p(y|z) is the logit model and maxSumVal = false (see 
%         DESCRIPTION), and p(z) = Normal(zhat,zvar).  When maxSumVal =
%         true, estim returns MAP estimates of each element of z, as well
%         as the second derivative of log p(y|z).
%

%
% Last change: 08/28/12
% Change summary: 
%       - Created (v0.1) (10/20/11; SR)
%       - Added numColumns method (08/16/12; JAZ)
%       - Added support for max-sum GAMP (v0.2) (08/28/12; JAZ)
% Version 0.2
%

classdef LogitEstimOut < EstimOut
    
    properties        
        y;                  % Vector of labels: 0 or 1
        scale = 1;          % scale factor
        npts = 100;         % number of points used in discrete integration
        wmax = 4;           % maximum value of integration
        maxSumVal = false;      % Sum-product (false) or max-sum (true) GAMP
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        function obj = LogitEstimOut(y, scale, npts, wmax, maxsumval)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.y = y;
                if (nargin >= 2) && ~isempty(scale)
                    obj.scale = scale;
                end
                if (nargin >= 3) && ~isempty(npts)
                    obj.npts = npts;
                end
                if (nargin >= 4) && ~isempty(wmax)
                    obj.wmax = wmax;
                end
                if (nargin >= 5) && ~isempty(maxsumval)
                    if isscalar(maxsumval)
                        obj.maxSumVal = logical(maxsumval);
                    else
                        error('maxSumVal must be a logical scalar')
                    end
                end
            end
        end
        
        
        % *****************************************************************
        %                           SET METHODS
        % *****************************************************************
        function obj = set.y(obj, y)
            if ~all((y(:) == 0) | (y(:) == 1))
                error('Elements of y must be binary (0,1)')
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
        % Pr{y(m)=0|z(m)} = 1 - Pr{y(m)=1|z(m)} = 1 / (1+exp(scale*z(m))),
        % if obj.maxSumVal = false, otherwise it will return zhat = argmax
        % log p(y|z) - 1/2/zvar0 (z - zhat0)^2 and the second derivative of
        % log p(y|z) evaluated at zhat in zvar, if obj.maxSumVal = true
        function [zhat, zvar] = estim(obj, zhat0, zvar0)
            
            % Check if zhat0 and zvar0 are only scalars (can occur during
            % first method call by gampEst) and resize
%           if numel(zhat0) == 1, zhat0 = zhat0*ones(size(obj.y)); end
%           if numel(zvar0) == 1, zvar0 = zvar0*ones(size(obj.y)); end
            
            switch obj.maxSumVal
                case false
                    % Compute the sum-product GAMP updates
                    
                    % Numerical integration approximation
                    % -----------------------------------
%                     % Gaussian pdf
%                     w = linspace(-obj.wmax,obj.wmax,obj.npts)';
%                     pw = exp(-0.5*w.^2);
%                     
%                     % Loop over points
%                     ny = length(obj.y);
%                     zhat = zeros(ny,1);
%                     zvar = zeros(ny,1);
%                     for iy = 1:ny
%                         z = zhat0(iy) + sqrt(zvar0(iy))*w;
%                         if (obj.y(iy))
%                             pyz = 1./(1+exp(-obj.scale*z));
%                         else
%                             pyz = 1./(1+exp(obj.scale*z));
%                         end
%                         pzy = pyz.*pw;
%                         pzy = pzy / sum(pzy);
%                         zhat(iy) = z'*pzy;
%                         zvar(iy) = ((z-zhat(iy)).^2)'*pzy;
%                     end
                    
                    % Variational inference approximation
                    % -----------------------------------
                    % Computing the posterior mean and variance, zhat and
                    % zvar, under the logit likelihood, and with a Gaussian
                    % prior with mean zhat0 and variance zvar0, is
                    % analytically intractable.  Rather than numerically
                    % integrate, we can use a modified version of the
                    % variational inference method described in Section
                    % 10.6 of Bishop's "Pattern Recognition and Machine
                    % Learning" book.
                    xi = sqrt(zvar0 + abs(zhat0).^2);   % Initialization
                    a = obj.scale;  % Shorthand for scale
                    NumIt = 100;     % Max # of optimize iters [was 25]
                    Tol = 1e-4;     % Early terminate tolerance [was 1e-8]
                    for i = 1:NumIt
                        % Optimize the variational lower bound
                        % approximation by iteratively updating the
                        % variational parameter vector, xi.  A handful of
                        % iterations should do the trick.
                        
%                         lam = (a/2./xi) .* ((1./(1 + exp(-a*xi))) - 1/2);
                        lam = (a/4./xi) .* tanh(a/2*xi);
%                         zvar = (1./zvar0 + 2*lam).^(-1);
                        zvar = zvar0 ./ (1 + 2*zvar0 .* lam);
                        zhat = zvar .* (zhat0./zvar0 + a*(obj.y - 1/2));
                        xi_upd = sqrt(zvar + abs(zhat).^2);
                        
                        % Check for early termination
                        err = norm(xi_upd(:) - xi(:))/norm(xi(:));
                        if err < Tol
                            break
                        else
                            xi = xi_upd;
                        end
                    end
                    %Print progress for debugging:
                    if 0
                      fprintf(1,'LogitEstimOut: iter=%1i, err=%6.3g\n',i,err); 
                    end
                    
                    
                case true
                  % Compute the max-sum GAMP updates
                  a = obj.scale;  % Shorthand for scale

                  alg = 2; % in {1,2,3}
                  switch alg

                   case 1 % Newton's method
                    debug = 1;
                    tol = 1e-8;
                    nit = 100;
                    step = 0.5;
                    ay = a*obj.y;
                    g = @(z) z - zhat0 - ay.*zvar0./(1+exp(ay.*z));
                    dg = @(z) 1 + (a^2*zvar0)./(2+exp(a*z)+exp(-a*z));
                    zhatOld = zhat0; % start at phat
                    if debug
                      zhat_ = nan(length(zhat0),nit+1);
                      err_ = nan(length(zhat0),nit);
                      zhat_(:,1) = zhatOld;
                    end
                    for i=1:nit,
                      zhat = zhatOld-step*g(zhatOld)./dg(zhatOld);
                      err = max(abs(zhat-zhatOld)./abs(zhatOld));
                      if debug
                        zhat_(:,i+1)=zhat;
                        err_(:,i)=abs(zhat-zhatOld)./abs(zhatOld);
                      end
                      %[err,...
                      % min(abs(zhatOld)),max(abs(zhatOld)),...
                      % min(g(zhatOld)),max(g(zhatOld)),...
                      % min(dg(zhatOld)),max(dg(zhatOld)),...
                      % min(dg(zhatOld)./dg(zhatOld)),...
                      % max(dg(zhatOld)./dg(zhatOld))]
                      if err<tol, 
                        break; 
                      else
                        zhatOld = zhat;
                      end
                    end
                    if i==nit, 
                      warning('LogisticEstimOut ran out of iterations');
                    end
                    %Print progress for debugging:
                    if debug
                      fprintf(1,'LogitEstimOut: iter=%1i, err=%6.3g\n',i,err); 
                      figure(100)
                      subplot(211)
                        plot(1:nit+1,zhat_')
                        ylabel('zhat')
                      subplot(212)
                        plot(1:nit,err_')
                        ylabel('err')
                      pause
                    end

                   case 2 % fsolve approach
                    PMonesY = sign(obj.y - mean(obj.y));% Convert {0,1} to {-1,1}
                    
                    % Specify the starting point for the subsequent
                    % iterative maximization scheme
                    EP = (sign(PMonesY) == sign(zhat0)) .* zhat0;
                    
                    % fsolve approach - Iteratively locate the value of z,
                    % zhat, that sets the derivative equal to zero
                    opts = optimset('Jacobian', 'on', 'MaxIter', 10, ...
                        'Display', 'off', 'TolX', 1e-3);
                    F = @(z) zero_deriv(obj, z, zhat0, zvar0);
                    zhat = fsolve(F, EP, opts);
                    
                   case 3 % fminunc approach 
                    % Use gradient methods to solve the
                    % prox-operator optimization problem

                    PMonesY = sign(obj.y - mean(obj.y));% Convert {0,1} to {-1,1}
                    EP = (sign(PMonesY) == sign(zhat0)) .* zhat0;
                    funHandle = @(z) logit_cost_fxn(obj, z, zhat0, zvar0);
                    options = optimset('GradObj', 'on', 'Hessian', ...
                          'off', 'TolX', 1e-5, 'Display', 'off', 'MaxIter', 100);
                    [zhat, ~, exitflag, output] = fminunc(funHandle, ...
                          EP, options);
                    % Check exit flag for possible problems
                    if exitflag ~= 1
                        warning(['Encountered numerical difficulties ' ...
                              'in computing max-sum updates.\n Message: %s\n'],...
                              output.message)
                    end

                  end % switch alg

                  % Also compute the 2nd derivative (wrt z) of log p(y|z)
                  % evaluated at the zhat that was just computed
                  deriv = -a^2 ./ (2 + exp(a*zhat) + exp(-a*zhat));
                    
                  % Output in zvar a function of the 2nd derivative that,
                  % once manipulated by gampEst, yields the desired
                  % max-sum expression for -g'_{out}
                  zvar = zvar0 ./ (1 - zvar0.*deriv);
            end
       
        end
                
        
        % *****************************************************************
        %                         LOGLIKE METHOD
        % *****************************************************************
        % This function will compute *an approximation* to the expected
        % log-likelihood, E_z[log p(y|z)] when performing sum-product GAMP
        % (obj.maxSumVal = false).  The approximation is based on Jensen's 
        % inequality, i.e., computing log E_z[p(y|z)] instead.  If
        % performing max-sum GAMP (obj.maxSumVal = true), logLike returns
        % log p(y|z) evaluated at z = zhat
        function ll = logLike(obj,zhat,zvar)
            switch obj.maxSumVal
                case false
%                     % Numerically integrate to estimate E_z[log p(y|z)]
%                     
%                     % Gaussian pdf
%                     w = linspace(-obj.wmax,obj.wmax,obj.npts)';
%                     pw = exp(-0.5*w.^2);
%                     pw = pw / sum(pw);
%                     
%                     % Loop over points
%                     ny = length(obj.y);
%                     logpy = zeros(ny,1);
%                     for iy = 1:ny
%                         z = zhat(iy) + sqrt(zvar(iy))*w;
%                         z = min(max(z,-10),10);
%                         if (obj.y(iy))
%                             logpy(iy) = -pw'*log(1+exp(-obj.scale*z));
%                         else
%                             logpy(iy) = -pw'*log(1+exp(obj.scale*z));
%                         end
%                     end
%                     ll = sum(logpy);
                    
                    % Variational lower bound approximation of E_z[log
                    % p(y|z)]
                    
                    % Start by computing the value of the variational parameter
                    % vector, xi
                    Xi = sqrt(zvar + abs(zhat).^2);
                    a = obj.scale;              % Shorthand for scale
                    
                    % Value of variational approximation LB
                    ll = sum(a*obj.y.*zhat - (a/2)*(zhat + Xi) - ...
                        log(1 + exp(-a*Xi)));
                case true
                    % Evaluate log p(y|zhat)
                    PMonesY = sign(obj.y - mean(obj.y));  % Convert {0,1} to {-1,1}
                    a = obj.scale;              % Shorthand for scale
                    NegPM = -PMonesY;           % Negated {-1,1}
                    ll = -log(1 + exp(a*NegPM.*zhat));
            end
        end
        
        % Compute output cost:
        % For sum-product compute
        %   (Axhat-phatfix)^2/(2*pvar) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar) 
        %   with phatfix such that Axhat=estim(phatfix,pvar).
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function LScale = logScale(obj,Axhat,pvar,phat)
                   
            error('logScale method not implemented for this class. Set the GAMP option adaptStepBethe = false.');         
            
        end
        
        
        % *****************************************************************
        %                       NUMCOLUMNS METHOD
        % *****************************************************************
        function S = numColumns(obj)
            %Return number of columns of y
            S = size(obj.y, 2);
        end
        
    end
    
    methods (Access = private)
        % *****************************************************************
        %                         FSOLVE METHOD
        % *****************************************************************
        % This method is used by MATLAB's fsolve function in the max-sum
        % case to locate the value of z that sets the prox-operator
        % derivative to zero
        function [Fval, Jacobian] = zero_deriv(obj, z, phat, pvar)
            M = size(obj.y, 1);
            
            % Compute value of derivative at z
            ExpZ = min(realmax, exp(-obj.scale*z));
            OhYeah = (1./pvar).*(z - phat);
            Fval = obj.scale*obj.y - obj.scale./(1 + ExpZ) - OhYeah;
            
            % Optionally compute Jacobian of F at z
            if nargout >= 2
                Jvec = -(obj.scale^2) * ExpZ ./ (1 + ExpZ).^2 - 1./pvar;
                Jacobian = sparse(1:M, 1:M, Jvec);
            end
        end
        
        
        % *****************************************************************
        %                       FMINUNC METHOD
        % *****************************************************************
        % This method is used by MATLAB's fminunc function in the max-sum
        % case to locate the prox-operator-minimizing value of z
        function [Fval, Deriv, Hessian] = logit_cost_fxn(obj, z, phat, pvar)
            M = size(obj.y, 1);
            
            y = sign(obj.y - 0.1);  % {0,1} -> {-1,1}
            
            % Compute prox-operator cost function value
            ExpZ = min(realmax, exp(-obj.scale*y.*z));
            ZminusP = z - phat;
            P_inv = 1 ./ pvar;
            Fval = -sum(-log(1 + ExpZ) - ((1/2) * P_inv .* ZminusP.^2));
            
            % Optionally compute the derivative at z
            if nargout >= 2
                Deriv = -(obj.scale * y) ./ (1 + exp(obj.scale*y.*z)) + ...
                    P_inv .* ZminusP;

                % Optionally compute Hessian at z
                if nargout >= 3
                    Hvec = (obj.scale^2) * ExpZ ./ (1 + ExpZ).^2 + P_inv;
                    Hessian = sparse(1:M, 1:M, Hvec);
                end
            end
        end
    end
end

