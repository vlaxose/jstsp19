function [xhatFinal, xvarFinal, rhatFinal, rvarFinal,...
    shatFinal, svarFinal, zhatFinal,zvarFinal,estHist] = ...
    gampEstSimp(scaEstIn, scaEstOut, A, opt)
% gampEstSimp:  GAMP estimation algorithm (simplified implementation)
%
% The G-AMP estimation algorithm is intended for the estimation of a
% random vector x observed through an observation y from the Markov chain
%
%   x -> z = A*x -> y,
%
% where the components of x are independent and the mapping z -> y is
% separable.  The function takes three arguments:
%
% scaEstIn:  An input estimator derived from the EstimIn class
%    based on the input distribution p_X(x_j).
% scaEstOut:  An output estimator derived from the EstimOut class
%    based on the output distribution p_{Y|Z}(y_i|z_i).
% A:  Either a matrix or a linear operator defined by the LinTrans class.
% opt:  A set of options of the class GampOpt.
%
% xhatFinal: The final estimate of the vector x.
% rhatFinal, rvarFinal:  Final estimate for rhat and rvar (see entries in
%     estHist below).
%
% estHist:  History of the estimator per iteration.  Note that this history
%    includes steps that may have been aborted.
% estHist.rhat, estHist.rvar:  When the estimator functions implement
%    the sum-product algorithm, the conditional distribution of x(j)
%    given the vector y is  approximately
%       p(x(j)|y) = C*p(x(j))*exp( -(rhat(j)-x(j))^2/(2*rvar(j) )
%    where C is a normalization constant.
% estHist.val:  The history of the value function.
% estHist.xhat:  The history of the estimates.
% estHist.xvar:  The history of the estimates.
% estHist.scaleFac:  The history of the variance-normalization scaling factor.

% Get options
if (nargin < 4)
    opt = GampOpt();
elseif (isempty(opt))
    opt = GampOpt();
end
nit     = opt.nit;              % number of iterations
step    = opt.step;             % step size
stepMin = opt.stepMin;          % minimum step size
stepMax = opt.stepMax;          % maximum step size
stepIncr = opt.stepIncr;	% step increase
stepDecr = opt.stepDecr;	% step decrease
adaptStep = opt.adaptStep;      % adaptive step size
stepWindow = opt.stepWindow;    % step size check window size
bbStep = opt.bbStep;            % Barzilai Borwein step size
verbose = opt.verbose;          % Print results in each iteration
tol = opt.tol;                  % Convergence tolerance
stepTol = opt.stepTol;          % minimum allowed step size
compVal = true;
saveHist = (nargout >= 9);
pvarStep = opt.pvarStep;        % incldue step size in pvar/pvarBar
unifVar = opt.uniformVariance;  % uniform variance
varNorm = opt.varNorm;          % normalize variances
scaleFac = opt.scaleFac;	% initial variance normalization

% If A is a double matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

% Get dimensions
[m,n] = size(A);


% Initial x step
[xhat0,xvar0, valIn0] = scaEstIn.estimInit();
if (size(xhat0,1) == 1)
    xhat = repmat(xhat0,n,1);
    xvar = repmat(xvar0,n,1);
    valIn = n*valIn0;
else
    xhat = xhat0;
    xvar = xvar0;
    valIn = sum(valIn0);
end

% Initial z step and value calculation
zhat0 = A*xhat;
pvar = A.absSq()*xvar;
shat = zeros(m,1);
valOut = sum( scaEstOut.logLike(zhat0, pvar) );
val = valIn + valOut;

% Save previous results
valFinal = val;
xhatFinal = xhat;
xvarFinal = xvar;
rhatFinal = zeros(n,1);
rvarFinal = zeros(n,1);
shatFinal = shat;
svarFinal = zeros(m,1);
zhatFinal = zeros(m,1);
zhat0Final = zhat0;
zvarFinal = zeros(m,1);

% Initialize history
if (saveHist)
    estHist.xhat = nan(n,nit);
    estHist.rhat = nan(n,nit);
    estHist.phat = nan(m,nit);
    estHist.xvar = nan(n,nit);
    estHist.rvar = nan(n,nit);
    estHist.pvar = nan(m,nit);
    estHist.val = nan(nit,1);
    estHist.dx = nan(nit,1);
    estHist.resid = nan(nit,1);
    estHist.step = nan(nit,1);
    estHist.pass = nan(nit,1);
    estHist.scaleFac = nan(nit,1);
    estHist.indOpt = nan(nit,1);
end

% Main iteration loop
stop = false;
it = 1;
while ~stop
    
    % Output linear step
    stepz = 1;
    pvar  = stepz* (A.absSq()*xvar);
    if (unifVar)
        pvar = repmat(mean(pvar),m,1);
    end
    pvar = max(pvar, opt.pvarMin);
    phat = zhat0 - pvar.*shat;

    % Output nonlinear step
    [zhat,zvar] = scaEstOut.estim(phat, pvar);  
    if (unifVar)
        zvar = repmat(mean(zvar),m,1);
    end
    
    % Transform for s
    if varNorm
        scaleFac = mean(pvar);
    else
        scaleFac = 1;
    end
    shat = step*(scaleFac./pvar).*(zhat-phat) + (1-step)*shat;
    svar = (scaleFac./pvar).*(1-zvar./pvar);
    
    % Input linear step
    stepx = 0.1;
    shatx = shat + (1-step)*(scaleFac./pvar).*zhat;
    rhat0 = A'*shatx;
    rvar  = stepx./(A.absSq()'*svar);  
    rhat  = xhat + rvar.*rhat0;
    
    % Input nonlinear step
    [xhat,xvar,valIn] = scaEstIn.estim(rhat, rvar);
    if (unifVar)
        xvar = repmat(mean(xvar),n,1);
    end
    xvar = max(xvar, opt.xvarMin);
    
    % Compute value and residual
    valIn = sum(valIn);       
    zhat0   = A*xhat;    
    valOut = sum( scaEstOut.logLike(zhat0, pvar) );
    val = valIn + valOut;
    resid = (zhat-zhat0)./pvar;
    resid = sum(abs(resid).^2);
    dx = sum(abs(xhat-xhatFinal).^2);
        
    if (verbose)
        fprintf(1,'it=%d val=%12.4e resid=%12.4e dx=%12.4e step=%12.4e\n', ...
            it, val, resid, dx, step);
    end
    
    % Check if pass
    % Check against worst value in last stepWindow good steps    
    if (it > 1)        
        stopInd =  it-1;
        startInd = max(1,it-1-stepWindow);
        pass = (val > min(estHist.val(startInd:stopInd))) ||...
            ~adaptStep || (step <= stepMin);        
    else
        pass = true;
    end
    
    % Adaptive step    
    if (pass)
        % Pass -- increment step
        if ((adaptStep) && (val > valFinal))
            step = min(step*stepIncr, stepMax);
        end
        
        % Save new results
        xhatFinal = xhat;
        xvarFinal = xvar;
        rhatFinal = rhat;
        rvarFinal = rvar;
        shatFinal = shat;
        svarFinal = svar;
        zhatFinal = zhat;
        zhat0Final = zhat0;
        zvarFinal = zvar;
        valFinal = val;
    else
        % Fail -- decrement step
        step = max( step*stepDecr, stepMin);

        % Restore old values
        xhat = xhatFinal;
        zhat0 = zhat0Final;
        xvar = xvarFinal;
        shat = shatFinal;
    end
    
    % Save results
    if (saveHist)
        estHist.xhat(:,it) = xhatFinal(:);
        estHist.rhat(:,it) = rhatFinal(:);
        estHist.phat(:,it) = phat;
        estHist.pvar(:,it) = reshape(pvar(1:m,:),[],1);
        estHist.xvar(:,it) = xvarFinal(:);
        estHist.rvar(:,it) = rvarFinal(:);
        estHist.val(it) = valFinal;
        estHist.dx(it) = dx;
        estHist.resid(it) = resid;
        estHist.step(it) = step;
    end
    
    % Iterate
    it = it + 1;
    if (it > nit)
        stop = true;
    end
    
end

% Trim the outputs if early termination occurred
if saveHist && (it < nit)
    estHist.xhat = estHist.xhat(:,1:it);
    estHist.rhat = estHist.rhat(:,1:it);
    estHist.phat = estHist.phat(:,1:it);
    estHist.val = estHist.val(1:it);
    estHist.step = estHist.step(1:it);
    estHist.pass = estHist.pass(1:it);
    estHist.rvar = estHist.rvar(:,1:it);
    estHist.pvar = estHist.pvar(:,1:it);
    estHist.xvar = estHist.xvar(:,1:it);
end

% Set outputs


