function [estFin,optFin,estHist] = gampEstBasic(scaEstIn, scaEstOut, A, opt)
% gampEstBasic:  G-AMP estimation algorithm ... a basic implementation
%
% DESCRIPTION:
% ------------
% The G-AMP estimation algorithm is intended for the estimation of a
% random vector x observed through an observation y from the Markov chain
%
%   x -> z = A*x -> y,
%
% where the prior p(x) and likelihood function p(y|z) are both separable.
%
% SYNTAX:
% -------
% [estFin,optFin,estHist] = gampEstBasic(scaEstIn, scaEstOut, A, opt)
%
% INPUTS:
% -------
% scaEstIn:  An input estimator derived from the EstimIn class
%    based on the input distribution p_X(x_j).
% scaEstOut:  An output estimator derived from the EstimOut class
%    based on the output distribution p_{Y|Z}(y_i|z_i).
% A:  Either a matrix or a linear operator defined by the LinTrans class.
% opt:  A set of options of the class GampOpt.
%
% OUTPUTS:
% --------
% estFin:  Final G-AMP estimation quantities
%   .xhat: estimate of the vector x (output of x-estimator)
%   .xvar: quadratic term for vector x (output of x-estimator)
%   .Axhat: same as A*estFin.xhat
%   .phat: estimate of the vector p (input to z-estimator)
%   .pvar: quadratic term for vector p (input to z-estimator)
%   .zhat: estimate of the vector z=Ax (output of z-estimator)
%   .zvar: quadratic term for vector z=Ax (output of z-estimator)
%   .shat: estimate of the vector s (lagrange penalty on z-Ax)
%   .svar: quadratic term for vector s (lagrange penalty on z-Ax)
%   .rhat: estimate of the vector r (input to x-estimator)
%   .rvar: quadratic term for vector r (input to x-estimator)
%   .xhatPrev: previous iteration of xhat (for legacy purposes)
%   .xhatNext: next iteration of xhat (used for warm start)
%   .xvarNext: next iteration of xvar (used for warm start)
%   .xhatDamp: damping state on xhat (used for warm start)
%   .pvarOpt = damping state on pvar (used for warm start)
%   .rvarOpt = damping state on rvar (used for warm start)
%   .A2xvarOpt = damping state on A2xvar (used for warm start)
%   .shatNext: next iteration of shat (used for warm start)
%   .svarNext: next iteration of svar (used for warm start)
%   .val: (for compatibility with full gampEst)
%   .scaleFac: (for compatibility with full gampEst)
%   .step: (used for warm start)
%   .stepMax: (for compatibility with full gampEst)
%
% optFin:  Final settings of GampOpt options object (see GampOpt.m)
%
% estHist:  History of G-AMP across iterations, 
%   .xhat: history of xhat
%   .xvar: history of xvar
%   .Axhat: history of A*xhat
%   .phat: history of phat
%   .pvar: history of pvar
%   .zhat: history of zhat
%   .zvar: history of zvar
%   .shat: history of shat
%   .svar: history of svar
%   .rhat: history of rhat
%   .rvar: history of rvar
%   .pass: history of pass/fail (for compatibility with gampEst)
%   .val: history of the utility (for compatibility with gampEst)
%   .scaleFac: history of the scalefactor (for compatibility with gampEst)
%   .step: history of the stepsize (i.e., damping term)
%   .stepMax: history of the max allowed stepsize (for compatibility)
%   .it = lists the iterations reported in the history
%
% Note that, in sum-product mode, the marginal posterior pdfs are
%    p(x(j)|y) ~= Cx*p(x(j))*exp( -(x(j)-rhat(j))^2/(2*rvar(j) )
%    p(z(i)|y) ~= Cz*p(y(i)|z(i))*exp( -(z(i)-phat(i))^2/(2*pvar(i) )
% where Cx and Cz are normalization constants.


% Get options
if (nargin < 4) || isempty(opt)
    opt = GampOpt();
end
nit     = opt.nit;              % number of iterations
step    = opt.step;             % stepsize
stepMin = opt.stepMin;          % minimum stepsize
stepMax = opt.stepMax;          % maximum stepsize
stepIncr = opt.stepIncr;        % stepsize increase
verbose = opt.verbose;          % Print results in each iteration
tol = opt.tol;                  % Convergence tolerance
pvarStep = opt.pvarStep;        % include stepsize in pvar?
rvarStep = opt.rvarStep;        % include stepsize in rvar?
pvarMin = opt.pvarMin;          % minimum value of pvar
rvarMin = opt.xvarMin;          % minimum value of rvar
histIntvl = opt.histIntvl;      % history interval
if (opt.legacyOut)
    error('requires opt.legacyOut=false')
end
if (opt.zvarToPvarMax<Inf)
    warning('opt.zvarToPvarMax is not implemented in this function')
end
if (opt.adaptStep)
    warning('opt.adaptStep is not implemented in this function')
end
if (opt.bbStep)
    warning('opt.bbStep is not implemented in this function')
end
if (opt.maxBadSteps<Inf)
    warning('opt.maxBadSteps is not implemented in this function')
end
if (opt.varNorm)
    warning('opt.varNorm is not implemented in this function')
end

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
elseif ~isempty(opt.stopFcn2),
    customStop = 2;
    stopFcn2 = opt.stopFcn2;
else
    customStop = false;
end

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

% Get dimensions
[m,n] = A.size();
s = scaEstOut.numColumns();

% Get default initialization values
[xhat,xvar] = scaEstIn.estimInit();

% Replace default initialization with user-provided values 
if ~isempty(opt.xhat0)
    xhat = opt.xhat0;
end
if ~isempty(opt.xvar0)
    xvar = opt.xvar0;
end

% For a scalar output, the same distribution is applied to all components
if (size(xhat,1) == 1)
    xhat = repmat(xhat,n,1);
end
if (size(xvar,1) == 1)
    xvar = repmat(xvar,n,1);
end

% Make sure that number of input columns match number of output columns
if (size(xhat,2) == 1)
    xhat = repmat(xhat,1,s);
end
if (size(xvar,2) == 1)
    xvar = repmat(xvar,1,s);
end

% If desired, automatically set xvar
if (opt.xvar0auto)
    % temporarily disable autoTuning
    if any(strcmp('disableTune', properties(scaEstOut)))
        disOut = scaEstOut.disableTune; % remember state 
        scaEstOut.disableTune = true; % temporarily set 
    end
    if any(strcmp('disableTune', properties(scaEstIn)))
        disIn = scaEstIn.disableTune; % remember state 
        scaEstIn.disableTune = true; % temporarily set 
    end

    % setup estimInvert options for both z & x variables
    xvarTol = 1e-4;
    zopt.maxIter = 100;
    zopt.stepsize = 0.25;
    zopt.regularization = 1e-20;
    zopt.tol = 1e-4;
    zopt.debug = false;
    xopt = zopt; % same options for x as for z

    % iterate to find fixed-point xvar
    xhat0 = xhat;
    Axhat0 = A.mult(xhat0);
    for t=1:100
        pvar = max(pvarMin, A.multSq(xvar));
        [phat,zhat,zvar,zstep] = estimInvert(scaEstOut,Axhat0,pvar,zopt);
        zopt.stepsize = zstep; % update stepsize in case it changed
        zopt.phat0 = phat; % warm-start
           %NRz = norm(zhat(:)-Axhat0(:))/norm(Axhat0(:))
        svar = (1-zvar./pvar)./pvar;
        svar(abs(svar)<eps) = eps;
        rvar = max(rvarMin, 1./(A.multSqTr(svar)));
        xvarOld = xvar;
        [rhat,xhat,xvar,xstep] = estimInvert(scaEstIn,xhat0,rvar,xopt);
        xopt.stepsize = xstep; % update stepsize in case it changed
        xopt.phat0 = rhat; % warm-start
           %NRx = norm(xhat(:)-xhat0(:))/norm(xhat0(:))
        if norm(xvar(:)-xvarOld(:))<norm(xvar(:))*xvarTol, break; end;
    end
    xhat = xhat0;

    % restore autoTuning
    if any(strcmp('disableTune', properties(scaEstOut)))
        scaEstOut.disableTune = disOut; % restore state
    end
    if any(strcmp('disableTune', properties(scaEstIn)))
        scaEstIn.disableTune = disIn; % restore state
    end
end

% Continue with initialization
shatPrev = zeros(m,s);	% default value is zero
svarPrev = nan(m,s);	% will test for NaN later
xhatDampPrev = nan(n,s);% will test for NaN later
pvarPrev = nan(m,s);	% will test for NaN later
rvarPrev = nan(n,s);	% will test for NaN later

% Replace default initialization with user-provided values 
if ~isempty(opt.shat0)
    shatPrev = opt.shat0;	
end
if ~isempty(opt.svar0)
    svarPrev = opt.svar0; 	
end
if ~isempty(opt.xhatDampPrev0)
    xhatDampPrev = opt.xhatDampPrev0;
end
if ~isempty(opt.pvarOpt0)
    pvarPrev = opt.pvarOpt0;
end
if ~isempty(opt.rvarOpt0)
    rvarPrev = opt.rvarOpt0;
end

% Check for valid initial stepsize 
step = min([max([step stepMin]) stepMax]);

% If the mean-removal option is set, create an augmented system
% with the mean removed.  (See LinTransDemeanRC.m for more details.)
% Then initialize variables.
if (opt.removeMean)
    A = LinTransDemeanRC(A,opt.removeMeanExplicit);
    [m,n] = A.size();
      maxSumVal = false; % NEED A PROPER WAY TO SET THIS! Make maxSumVal a property of all estimOut?
      isCmplx = false;  % NEED A PROPER WAY TO SET THIS! Make isCmplx a property of all estimOut?
    scaEstOut = A.expandOut(scaEstOut,maxSumVal,isCmplx);
    scaEstIn = A.expandIn(scaEstIn); % SHOULD ALSO DEPEND ON maxSumVal and isCmplx!
    xhat = A.expandXhat(xhat);
    xvar = A.expandXvar(xvar);
    shatPrev = A.expandShat(shatPrev);
    svarPrev = A.expandSvar(svarPrev);
    xhatDampPrev = A.expandXhat(xhatDampPrev);
    pvarPrev = A.expandSvar(pvarPrev);
    rvarPrev = A.expandXvar(rvarPrev);
end

% If uniform-variance mode is requested by the user, implement it by
% redefining the A.multSq and A.multSqTr operations
if (opt.uniformVariance)
    if ~(opt.removeMean)
        A = UnifVarLinTrans(A);
    else
        A = UnifVarLinTrans(A,1:m-2,1:n-2); % don't average augmented elements
    end;
end

% Declare variables
zhat = nan(m,s);
zvar = nan(m,s);
phat = nan(m,s);
pvar = nan(m,s);
rhat = nan(n,s);		
rvar = nan(n,s);		
xhatPrev = nan(n,s);		
if (saveHist)
    nitSave=floor(nit/histIntvl);
    estHist.xhat = nan(n*s,nitSave);
    estHist.xvar = nan(n*s,nitSave);
    estHist.Axhat = nan(m*s,nitSave);
    estHist.phat = nan(m*s,nitSave);
    estHist.pvar = nan(m*s,nitSave);
    estHist.shat = nan(m*s,nitSave);
    estHist.svar = nan(m*s,nitSave);
    estHist.zhat = nan(m*s,nitSave);
    estHist.zvar = nan(m*s,nitSave);
    estHist.rhat = nan(n*s,nitSave);
    estHist.rvar = nan(n*s,nitSave);
    estHist.step = nan(nitSave,1);
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
      pvarPrev = pvar; 
      rvarPrev = rvar; 
      shatPrev = shat;
      svarPrev = svar;
      xhatPrev = xhat;
      xhatDampPrev = xhatDamp;
      xhat = xhatNext;
      xvar = xvarNext;
    end

    % Output linear stage 
    pvar = A.multSq(xvar);

    % Apply damping
    if pvarStep
        if (it==1)
            if any(isnan(pvarPrev)),    % if user didn't specify pvarPrev0
                pvarPrev = pvar; 	% equivalent to making step=1
            end
        end
        pvar = (1-step)*pvarPrev + step*pvar;
    end

    % Continued output linear stage
    Axhat = A.mult(xhat);
    phat = Axhat - pvar.*shatPrev;
        
    % Output nonlinear stage 
    pvarRobust = max(pvar,pvarMin); % At very high SNR, use very small pvarMin!
    [zhat,zvar] = scaEstOut.estim(phat,pvarRobust);
    shat = (1./pvarRobust).*(zhat-phat);
    svar = (1./pvarRobust).*(1-zvar./pvarRobust);

    % Adjust stepsize
    step = min([stepIncr*max([step stepMin]) stepMax]);
    
    % Print results
    if (verbose)
        fprintf(1,'it=%3d  stepsize=%f  |dx|/|x|=%12.4e\n', ...
            it, step, norm(xhatPrev - xhat)/norm(xhat));
    end
    
    % Apply damping to shat, svar, and xhat
    if (it==1)
        if any(isnan(svarPrev)), % if user didn't specify svar0
            svarPrev = svar; % equivalent to making step=1
        end
        if any(isnan(xhatDampPrev)), % if user didn't specify xhatDampPrev0
            xhatDampPrev = xhat; % equivalent to making step=1
        end
    end
    shat = (1-step)*shatPrev + step*shat;
    svar = (1-step)*svarPrev + step*svar;
    svar(abs(svar)<eps) = eps; % prevents infinite rvar (in which case rhat is often NaN)
    xhatDamp = (1-step)*xhatDampPrev + step*xhat;

    % Apply damping
    rvar = 1./A.multSqTr(svar);   % rvar = 1./((A.^2)*svar)
    if rvarStep
        if (it==1)
            if any(isnan(rvarPrev)),    % if user didn't specify rvarPrev0
                rvarPrev = rvar; 	% equivalent to making step=1
            end
        end
        rvar = (1-step)*rvarPrev + step*rvar;
    end
    
    % Input linear stage
    rhat = xhatDamp + rvar.*(A.multTr(shat)); % rhat = xhat + rvar.*(A'*shat)
    
    % Input nonlinear stage
    rvarRobust = max(rvar, rvarMin); % At very high SNR, use very small rvarMin!
    [xhatNext,xvarNext] = scaEstIn.estim(rhat, rvarRobust);

    % Save results
    if (itSaveHist)
        estHist.Axhat(:,itSaveHist) = Axhat(:);
        estHist.phat(:,itSaveHist) = phat(:);
        estHist.pvar(:,itSaveHist) = pvar(:);
        estHist.zhat(:,itSaveHist) = zhat(:);
        estHist.zvar(:,itSaveHist) = zvar(:);
        estHist.shat(:,itSaveHist) = shat(:);
        estHist.svar(:,itSaveHist) = svar(:);
        estHist.rhat(:,itSaveHist) = rhat(:);
        estHist.rvar(:,itSaveHist) = rvar(:);
        estHist.xhat(:,itSaveHist) = xhat(:);
        estHist.xvar(:,itSaveHist) = xvar(:);
        estHist.step(itSaveHist) = step;
    end
    
    % Check for convergence
    if (it>1) && (stop==false)
        if (norm(xhatNext - xhat)/norm(xhatNext) < tol)
            stop = true;
        elseif customStop==1
            stop = stopFcn(0, xhatNext, xhat, Axhat);
        elseif customStop==2
            S = struct(...
                'it',it,...
                'val',0,'xhatPrev',xhatPrev,'Axhat',Axhat, ...
                'xhat',xhat,'xvar',xvar,...
                'rhat',rhat,'rvar',rvar,...
                'phat',phat,'pvar',pvar,...
                'zhat',zhat,'zvar',zvar,...
                'shat',shat,'svar',svar ...
                );
            stop = stopFcn2(S);
        end
    end

end % main loop

%Trim the history if early termination occurred
if saveHist
    nitTrim = 1:floor(it/histIntvl);
    if (it < nit)
        estHist.xhat = estHist.xhat(:,nitTrim);
        estHist.xvar = estHist.xvar(:,nitTrim);
        estHist.Axhat = estHist.Axhat(:,nitTrim);
        estHist.phat = estHist.phat(:,nitTrim);
        estHist.pvar = estHist.pvar(:,nitTrim);
        estHist.zhat = estHist.zhat(:,nitTrim);
        estHist.zvar = estHist.zvar(:,nitTrim);
        estHist.shat = estHist.shat(:,nitTrim);
        estHist.svar = estHist.svar(:,nitTrim);
        estHist.rhat = estHist.rhat(:,nitTrim);
        estHist.rvar = estHist.rvar(:,nitTrim);
        estHist.step = estHist.step(nitTrim);
    end
    % Set other quantities for compatibility with gampShowHist.m
    estHist.val = -inf*ones(size(estHist.step));
    estHist.stepMax = stepMax*ones(size(estHist.step));
    estHist.pass = ones(size(estHist.step));
    estHist.scaleFac = ones(size(estHist.step));
    estHist.it = (nitTrim(:))*histIntvl;
end

%Trim the outputs if mean removal was turned on
if (opt.removeMean)
    xhatNext = A.contract(xhatNext);
    xvarNext = A.contract(xvarNext);
    xhatDamp = A.contract(xhatDamp);
    xhat = A.contract(xhat);
    xvar = A.contract(xvar);
    Axhat = A.contract(Axhat);
    phat = A.contract(phat);
    pvar = A.contract(pvar);
    zhat = A.contract(zhat);
    zvar = A.contract(zvar);
    shat = A.contract(shat);
    svar = A.contract(svar);
    rhat = A.contract(rhat);
    rvar = A.contract(rvar);
end

% Export outputs
estFin.xhatNext = xhatNext;
estFin.xvarNext = xvarNext;
estFin.xhatDamp = xhatDamp;
estFin.xhat = xhat;
estFin.xvar = xvar;
estFin.Axhat = Axhat;
estFin.phat = phat;
estFin.pvar = pvar;
estFin.zhat = zhat;
estFin.zvar = zvar;
estFin.shat = shat;
estFin.svar = svar;
estFin.rhat = rhat;
estFin.rvar = rvar;
estFin.scaleFac = 1;
estFin.step = step;
estFin.stepMax = stepMax;
estFin.nit = it;
optFin = opt;

estFin.A2xvarOpt = nan(size(pvar)); % needed for compatibility with gampEst
estFin.pvarOpt = pvar; % for compatibility with gampEst
estFin.rvarOpt = rvar; % for compatibility with gampEst
estFin.shatNext = shat; % for compatibility with gampEst
estFin.svarNext = svar; % for compatibility with gampEst
estFin.valIn = nan; % needed for compatibility with gampEst
estFin.valOpt = []; % needed for compatibility with gampEst
estFin.failCount = 0; % needed for compatibility with gampEst

