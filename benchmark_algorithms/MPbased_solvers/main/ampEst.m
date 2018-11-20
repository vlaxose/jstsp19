function [estFin,optFin,estHist] = ampEst(scaEstIn, y, A, opt)
% ampEst:  AMP estimation algorithm ... a basic implementation
%
% DESCRIPTION:
% ------------
% The AMP estimation algorithm is intended for the estimation of a
% random vector x observed through an observation y from the Markov chain
%
%   x -> y = A*x + AWGN 
%
% where the prior p(x) is separable.
%
% SYNTAX:
% -------
% [estFin,optFin,estHist] = ampEst(scaEstIn, y, A, opt)
%
% INPUTS:
% -------
% scaEstIn:  An input estimator derived from the EstimIn class
%    based on the input distribution p_X(x_j).
% y:  A vector of measurements
% A:  Either a matrix or a linear operator defined by the LinTrans class.
% opt:  A set of options of the class AmpOpt.
%
% OUTPUTS:
% --------
% estFin:  Final AMP estimation quantities
%   .xhat: estimate of the vector x (output of x-estimator)
%   .xvar: quadratic term for vector x (output of x-estimator)
%   .rhat: estimate of the vector r (input to x-estimator)
%   .rvar: quadratic term for vector r (input to x-estimator)
%
% optFin:  Final settings of AmpOpt options object (see AmpOpt.m)
%
% estHist:  History of AMP across iterations, 
%   .xhat: history of xhat
%   .xvar: history of xvar
%   .rhat: history of rhat
%   .rvar: history of rvar
%
% Note that, in sum-product mode, the marginal posterior pdfs are
%    p(x(j)|y) ~= Cx*p(x(j))*exp( -(x(j)-rhat(j))^2/(2*rvar(j) )
% where Cx is a normalization constant.


% Get options
if (nargin < 4) || isempty(opt)
    opt = AmpOpt();
end
nit     = opt.nit;            % number of iterations
verbose = opt.verbose;        % Print results in each iteration
tol = opt.tol;                % Convergence tolerance
rvarMethod = opt.rvarMethod;  % method used to compute rvar
rvarMin = opt.rvarMin;        % minimum value of rvar
histIntvl = opt.histIntvl;    % history interval
checkA = opt.checkA;          % check if A is properly normalize
normalizeA = opt.normalizeA;  % perform normalization if needed
Stransform = opt.Stransform;  % run S-AMP from [Cakmak,Winther,Fleury]?
evalsAAh = opt.evalsAAh;      % eigenvalues of A*A', used in S-AMP
wvar = opt.wvar;              % measurement noise variance, needed in S-AMP

% Handle output format
saveHist = (nargout >= 3);
if (nargout > 3),
    error('too many output arguments')
end;

% Check for the presence of a custom stopping criterion in the options
% structure, and set flags as needed
if ~isempty(opt.stopFcn),
    customStop = 1;
    stopFcn = opt.stopFcn;
else
    customStop = false;
end

% Handle rvarMethod
if strcmp(rvarMethod,'wvar')
    if isempty(wvar)
        error('must specify opt.wvar when opt.rvarMethod = wvar');
    end
end

% Handle some parameters needed for S-AMP
if Stransform 
    if isempty(wvar)
        error('must specify opt.wvar when opt.Stransform = true');
    else
        opt.rvarMethod = 'wvar'; % the only method implemented for S-AMP
    end

    if isempty(evalsAAh)
        if verbose
            fprintf('Computing eigendecomposition of A*Ah...\n');
            pause(eps);
        end

        if isa(A,'numeric')
            evalsAAh = eig(A*A');
        else % assume A is a LinTrans object
            [m,n] = A.size();
            try
                evalsAAh = eig(A.mult(A.multTr(speye(m))));
            catch
                if verbose
                    fprintf('A.mult doesnt support matrix argument!  Will slow down eigendecomposition...\n')
                end
                AAh = zeros(m);
                I = eye(m);
                for i=1:m, AAh(:,i) = A.mult(A.multTr(I(:,i))); end;
                evalsAAh = eig(AAh);
                clear I AAh;
            end
        end
    end
end

% If A is an explicit matrix, replace by an operator
if isa(A,'numeric')
    A = MatrixLinTrans(A);
end

% Get dimensions
[m,n] = A.size();
s = size(y,2);

% Get default initialization values
xhat = scaEstIn.estimInit();

% For a scalar output, the same distribution is applied to all components
if (size(xhat,1) == 1)
    xhat = repmat(xhat,n,1);
end

% Make sure that number of input columns match number of output columns
if (size(xhat,2) == 1)
    xhat = repmat(xhat,1,s);
end

% Continue with initialization: want (mean(Xvar,1)./rvarPrev)*vhatPrev = 0
vhatPrev = zeros(m,s);	
Xvar = zeros(n,s);
rvarPrev = inf*ones(1,s);	

% Replace default initialization with user-provided values 
if ~isempty(opt.xhat0)
    xhat = opt.xhat0;
end
if ~isempty(opt.vhatPrev0)
    vhatPrev = opt.vhatPrev0;
end

% Check if norm(A,'fro')^2 ~= size(x,1)
scale = 1; % scale factor used to normalize AMP when A is improperly scaled
if checkA
    ratio = squaredNorm(A)/n;
    if verbose
        fprintf('matrix normalization check: ||A||_F^2 = %g N\n',ratio)
    end

    if abs(ratio-1) > 4/sqrt(m*n), % ... a 4 sigma event for iid Gaussian case
        if normalizeA && (~Stransform) % normalization not yet implemented for S-AMP!
            scale = 1/sqrt(ratio);
        else
            warning('A operator is not properly scaled!')
        end
    end
end

% Declare variables
if (saveHist)
    nitSave=floor(nit/histIntvl);
    estHist.xhat = nan(n*s,nitSave);
    estHist.xvar = nan(n*s,nitSave);
    estHist.Axhat = nan(m*s,nitSave);
    estHist.vhat = nan(m*s,nitSave);
    estHist.rhat = nan(n*s,nitSave);
    estHist.rvar = nan(n*s,nitSave);
end

% Control variable to end the iterations
stop = false;
it = 0;

% Main iteration loop
while ~stop
    
    % Iteration count
    it = it + 1;
    
    % Check for final iteration
    if it >= nit
        stop = true;
    end

    % Check whether to save this iteration in history
    if saveHist && rem(it,histIntvl)==0
        itSaveHist = it/histIntvl;
    else
        itSaveHist = []; % don't record
    end

    % Handle non-initial iterations
    if it > 1
        vhatPrev = vhat;
        rvarPrev = rvar;
        xhat = xhatNext;
        Xvar = XvarNext;
    end

    % Compute Onsager-corrected residual
    Axhat = A.mult(xhat);

    % Compute Onsager gain
    xvar = mean(Xvar,1);
    div = xvar./rvarPrev; 
    if ~Stransform % standard AMP
        gain = (n/m)*div; 

    else % S-AMP
        div = min(div, m/n - 1e-3); % clip for s_transform
        gain = 1 - 1./s_transform( -div, evalsAAh, n );

    end

    % Compute Onsager-corrected residual
    vhat = scale*(y - Axhat) + bsxfun(@times,gain,vhatPrev);
    
    % Estimate noise variance at denoiser input
    if it==1
      rvar = (1/m)*sum(abs(vhat).^2,1);

    else
      if ~Stransform % standard AMP
        switch rvarMethod
          case 'mean' 
            rvar = (1/m)*sum(abs(vhat).^2,1);
          case 'median' 
            if any(~isreal(vhat))
              rvar = (sqrt(2/log(4))*median(abs(vhat),1)).^2;
            else
              rvar = ((1/0.6745)*median(abs(vhat),1)).^2; % inv_cdf_norm(3/4) ~= 0.6745
            end
          case 'wvar' 
            rvar = wvar + (n/m)*xvar;
        end

      else % S-AMP
        % use bisection to find rvar such that
        %    rvar = wvar*s_transform( -xvar./rvar, evalsAAh, n ) 

        % initialize bisection algorithm
        rvar_low = (n/m)*xvar; % smallest value allowed by s_transform()
        err_low = rvar_low - wvar*s_transform( -(m/n), evalsAAh, n );
        if any(err_low > 0), 
            warning('need to increase rvar_low'); 
        end
        rvar_high = rvar_low*100; % some larger value 
        err_high = rvar_high - wvar*s_transform( -xvar./rvar_high, evalsAAh, n );
        if any(err_high < 0), 
            warning('need to increase rvar_high'); 
        end
        rvar = 0.5*(rvar_low+rvar_high);

        % run bisection algorithm
        Tmax = 50;
        for t = 1:Tmax 
            err = rvar - wvar*s_transform( -xvar./rvar, evalsAAh, n ); % want =0
            high = find(err>0); rvar_high(high) = rvar(high);
            low = find(err<0); rvar_low(low) = rvar(low);
            rvar_old = rvar;
            rvar = 0.5*(rvar_low+rvar_high);  % bisect
            if ~any(abs(rvar-rvar_old)./rvar > tol), break; end;
        end
      end % if Stransform
    end % if it==1

    % prevent too-small rvar and expand to vector 
    rvar = max(rvar, rvarMin); % at high SNR, use small rvarMin!
    Rvar = ones(n,1)*rvar; % vector-valued for compatibility with EstimIn

    % Compute denoiser input 
    rhat = xhat + scale*A.multTr(vhat); 
    
    % Save results
    if (itSaveHist)
        estHist.xhat(:,itSaveHist) = xhat(:);
        estHist.xvar(:,itSaveHist) = Xvar(:);
        estHist.Axhat(:,itSaveHist) = Axhat(:);
        estHist.vhat(:,itSaveHist) = vhat(:);
        estHist.rhat(:,itSaveHist) = rhat(:);
        estHist.rvar(:,itSaveHist) = Rvar(:);
    end
    
    % Perform denoising
    [xhatNext,XvarNext] = scaEstIn.estim(rhat, Rvar);

    % Check for convergence
    if (norm(xhatNext - xhat,'fro')/norm(xhatNext,'fro') < tol)
        stop = true;
    elseif customStop==1
        stop = stopFcn(0, xhatNext, xhat, Axhat);
    end

    % Print results
    if (verbose)
        fprintf(1,'it=%3d  rvar=%12.4e  |dx|/|x|=%12.4e\n', ...
            it, mean(rvar), norm(xhatNext-xhat,'fro')/norm(xhat,'fro'));
    end
    
end % main loop

%Trim the history if early termination occurred
if saveHist
    nitTrim = 1:floor(it/histIntvl);
    if (it < nit)
        estHist.xhat = estHist.xhat(:,nitTrim);
        estHist.xvar = estHist.xvar(:,nitTrim);
        estHist.Axhat = estHist.Axhat(:,nitTrim);
        estHist.vhat = estHist.vhat(:,nitTrim);
        estHist.rhat = estHist.rhat(:,nitTrim);
        estHist.rvar = estHist.rvar(:,nitTrim);
    end

    % Set other quantities for compatibility with gampShowHist.m
    estHist.it = (nitTrim(:))*histIntvl;
    estHist.step = ones(1,length(nitTrim));
    estHist.stepMax = ones(1,length(nitTrim));
    estHist.val = nan(1,length(nitTrim));
    estHist.pass = nan(1,length(nitTrim));
    estHist.pass = nan(1,length(nitTrim));
end

% Export outputs
estFin.xhatNext = xhatNext;
estFin.xvarNext = XvarNext;
estFin.xhat = xhat;
estFin.xvar = Xvar;
estFin.Axhat = Axhat;
estFin.vhat = vhat;
estFin.rhat = rhat;
estFin.rvar = Rvar;
estFin.nit = it;
optFin = opt;
optFin.evalsAAh = evalsAAh;
