function [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10] = ...
    gampEst(scaEstIn, scaEstOut, A, opt)
% gampEst:  Generalized Approximate Message Passing -- Estimation algorithm
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
% [out1,out2,out3,out4,out5,out6,out7,out8,out9] = ...
%                               gampEst(scaEstIn, scaEstOut, A, opt)
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
%  LEGACY FORMAT:
%  out1 = xhatFinal
%  out2 = xvarFinal [optional]
%  out3 = rhatFinal [optional]
%  out4 = rvarFinal [optional]
%  out5 = shatFinal [optional]
%  out6 = svarFinal [optional]
%  out7 = zhatFinal [optional]
%  out8 = zvarFinal [optional]
%  out9 = estHist [optional]
%
%  NEW FORMAT:
%  out1 = estFin
%  out2 = optFin
%  out3 = estHist [optional]
%
%   ... where ...
%
%  xhatFinal: final estimate of the vector x (output of x-estimator).
%  xvarFinal: final quadratic term for vector x (output of x-estimator).
%  phatFinal: final estimate of the vector p (input to z-estimator).
%  pvarFinal: final quadratic term for vector p (input to z-estimator).
%  zhatFinal: final estimate of the vector z=Ax (output of z-estimator).
%  zvarFinal: final quadratic term for vector z=Ax (output of z-estimator).
%  shatFinal: final estimate of the vector s (lagrange penalty on z-Ax).
%  svarFinal: final quadratic term for vector s (lagrange penalty on z-Ax).
%  rhatFinal: final estimate of the vector r (input to x-estimator).
%  rvarFinal: final quadratic term for vector r (input to x-estimator).
%
% estFin:  Final G-AMP estimation quantities
%   .xhat: same as xhatFinal above
%   .xvar: same as xvarFinal above
%   .Axhat: same as A*xhatFinal above
%   .phat: same as phatFinal above
%   .pvar: same as pvarFinal above
%   .zhat: same as zhatFinal above
%   .zvar: same as zvarFinal above
%   .shat: same as shatFinal above
%   .svar: same as svarFinal above
%   .rhat: same as rhatFinal above
%   .rvar: same as rvarFinal above
%   .xhatPrev: previous iteration of xhatFinal
%   .xhatNext: next iteration of xhat (used for warm start)
%   .xvarNext: next iteration of xvar (used for warm start)
%   .xhatDamp: damping state on xhat (used for warm start)
%   .pvarOpt = damping state on pvar (used for warm start)
%   .rvarOpt = damping state on rvar (used for warm start)
%   .A2xvarOpt = damping state on A2xvar (used for warm start)
%   .shatNext: next iteration of shat (used for warm start)
%   .svarNext: next iteration of svar (used for warm start)
%   .val: final value of utility (i.e., negative cost)
%   .valIn: final value of input utility (i.e., negative cost)
%   .valOpt: final record of input utilities (i.e., negative cost)
%   .scaleFac: final value of scaling used when varNorm=true
%   .step: final value of the stepsize (i.e., damping term)
%   .stepMax: final value of the maximum stepsize
%
% optFin:  Final settings of GampOpt options object (see GampOpt.m)
%
% estHist:  History of G-AMP across iterations  
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
%   .pass: history of pass/fail
%   .val: history of the utility (i.e., negative cost)
%   .scaleFac: history of the scalefactor used when varNorm=true
%   .step: history of the stepsize (i.e., damping term)
%   .stepMax: history of the maximum allowed stepsize
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
stepDecr = opt.stepDecr;        % stepsize decrease
adaptStep = opt.adaptStep;      % utility-based adaptive stepsize on?
adaptStepBethe = opt.adaptStepBethe; %Use the cost computed from Bethe free energy
stepWindow = opt.stepWindow;    % adaptive stepsize: size of moving window 
bbStep = opt.bbStep;            % Barzilai-Borwein-based stepsize adaptation on?
verbose = opt.verbose;          % Print results in each iteration?
tol = opt.tol;                  % Convergence tolerance
maxBadSteps = opt.maxBadSteps;  % maximum number of allowed failed iterations
maxStepDecr = opt.maxStepDecr;  % amount to decrease maxStep after failures
stepTol = opt.stepTol;          % minimum allowed stepsize
pvarStep = opt.pvarStep;        % include stepsize in pvar?
rvarStep = opt.rvarStep;        % include stepsize in rvar?
varNorm = opt.varNorm;          % normalize variances?
scaleFac = opt.scaleFac;        % initial variance normalization
pvarMin = opt.pvarMin;          % minimum value of pvar
rvarMin = opt.xvarMin;          % minimum value of rvar
zvarToPvarMax = opt.zvarToPvarMax;  % maximum zvar/pvar ratio
histIntvl = opt.histIntvl;      % history interval

% Handle output format
legacyOut = opt.legacyOut;      % use legacy output format?
if (legacyOut),
    saveHist = (nargout >= 10);
    if (nargout > 10),
        error('too many output arguments')
    end;
else % modern output format
    saveHist = (nargout >= 3);
    if (nargout > 3),
        error('too many output arguments')
    end;
end

% Determine whether the utility must be computed each iteration
if adaptStep,
    compVal = true;             % must be true
else
    compVal = false;            % can set at false for faster runtime
end

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
[xhat,xvar,valIn] = scaEstIn.estimInit();
valIn = sum( valIn(:) );

% Replace default initialization with user-provided values
if ~isempty(opt.xhat0)
    if sum(xhat~=opt.xhat0), 
        valIn = -Inf; % default valIn no longer valid
    end; 
    xhat = opt.xhat0;
end
if ~isempty(opt.xvar0)
    if sum(xvar~=opt.xvar0), 
        valIn = -Inf; % default valIn no longer valid
    end; 
    xvar = opt.xvar0;
end
%valIn = -Inf; % only for backwards compatibility.  Remove in next major revision?
if ~isempty(opt.valIn0)
    valIn = opt.valIn0; 
end
%valOpt = [];     % empty initialization will cause the first iteration to be a "pass"
valOpt = -Inf;     
if ~isempty(opt.valOpt0)
    valOpt = opt.valOpt0;       
end
val = nan;

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

% Continue with initialization
shat = zeros(m,s);      % default value is zero
svar = nan(m,s);        % will test for NaN later
xhatDamp = nan(n,s);    % will test for NaN later
pvarOpt = nan(m,s);     % will test for NaN later 
rvarOpt = nan(n,s);     % will test for NaN later 
A2xvarOpt = nan(m,s);   % will test for NaN later 

% Replace default initialization with user-provided values 
if ~isempty(opt.shat0)
    shat = opt.shat0*scaleFac;  % variance normalization included
end
if ~isempty(opt.svar0)
    svar = opt.svar0*scaleFac;  % variance normalization included
end
if ~isempty(opt.xhatPrev0)
    xhatDamp = opt.xhatPrev0;
end
if ~isempty(opt.pvarOpt0)
    pvarOpt = opt.pvarOpt0;
end
if ~isempty(opt.rvarOpt0)
    rvarOpt = opt.rvarOpt0;
end
if ~isempty(opt.A2xvarOpt0)
    A2xvarOpt = opt.A2xvarOpt0;
end

% Replace the stepMax adaptation quantities with user-provided values
failCount = 0;
if ~isempty(opt.failCount0)
    failCount = opt.failCount0;
end

% If the mean-removal option is set, create an augmented system
% with the mean removed.  (See LinTransDemeanRC.m for more details.)
if (opt.removeMean)
    A = LinTransDemeanRC(A,opt.removeMeanExplicit);
    [m,n] = A.size();
      maxSumVal = false; % NEED A PROPER WAY TO SET THIS! Make maxSumVal a property of all estimOut?
      isCmplx = false;  % NEED A PROPER WAY TO SET THIS! Make isCmplx a property of all estimOut?
    scaEstOut = A.expandOut(scaEstOut,maxSumVal,isCmplx);
    scaEstIn = A.expandIn(scaEstIn); % SHOULD ALSO DEPEND ON maxSumVal and isCmplx!
    xhat = A.expandXhat(xhat);
    xvar = A.expandXvar(xvar);
    shat = A.expandShat(shat);
    svar = A.expandSvar(svar);
    xhatDamp = A.expandXhat(xhatDamp);
    pvarOpt = A.expandSvar(pvarOpt);
    rvarOpt = A.expandXvar(rvarOpt);
    A2xvarOpt = A.expandSvar(A2xvarOpt);
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

% Declare variables
zhat = nan(m,s);
zvar = nan(m,s);
phat = nan(m,s);
rhat = nan(n,s);                
rvar = nan(n,s);                
xhatFinal = nan(n,s);
if (saveHist)
    nitSave = floor(nit/histIntvl);
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
    estHist.val = nan(nitSave,1);
    estHist.stepMax = nan(nitSave,1);
    estHist.pass = nan(nitSave,1);
    estHist.scaleFac = nan(nitSave,1);
    elapsed_time = 0;
end

% Check for the presence of two methods within the LinTrans and EstimIn
% objects and set flags accordingly
MtxUncertaintyFlag = ismethod(A,'includeMatrixUncertainty');
MsgUpdateFlag = ismethod(scaEstIn, 'msgUpdate');

% If using BB stepsize adaptation, compute column norms for use in scaling
if bbStep
    columnNorms = A.multSqTr(ones(m,1)).^0.5;
    columnNorms = repmat(columnNorms,1,s);
end

% Control variables to terminate the iterations
stop = false;
it = 0;
elapsed_time = 0;

% Main iteration loop
while ~stop
    tic;
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

    % Output linear stage with no A uncertainty
    A2xvar = A.multSq(xvar);
    
    % Incorporate A uncertainty
    if MtxUncertaintyFlag
        pvar = A.includeMatrixUncertainty(A2xvar,xhat,xvar);
    else
        pvar = A2xvar;
    end
    
    % Continued output linear stage
    Axhat = A.mult(xhat);
    
    % Step in pvar
    if pvarStep
        if (it==1)
            if any(isnan(pvarOpt)),    % if user didn't specify opt.pvarOpt0
                pvarOpt = pvar;        % equivalent to making step=1
            end
            if any(isnan(A2xvarOpt)),    % if user didn't specify opt.A2xvarOpt0
                A2xvarOpt = A2xvar;    % equivalent to making step=1
            end
        end
        pvar = (1-step)*pvarOpt + step*pvar;
        A2xvar = (1-step)*A2xvarOpt + step*A2xvar;
    end
    
    % Continued output linear stage
    phat = Axhat - ((1/scaleFac)*A2xvar).*shat; % Note: uses A2xvar rather than pvar
    pvarRobust = max(pvar,pvarMin); % At very high SNR, use very small pvarMin!
        
    % Compute expected log-likelihood of the output and add to negative 
    % KL-divergence of the input, giving the current utility function 
    if (compVal)
        if ~adaptStepBethe
            valOut = sum(sum(scaEstOut.logLike(Axhat,pvar)));
        else
            valOut = sum(sum(scaEstOut.logScale(Axhat,pvar,phat)));
        end
        val = valOut + valIn;
    end

    % An iteration "passes" if any of below is true: 
    % 1. Adaptive stepsizing is turned off
    % 2. Current stepsize is so small it can't be reduced 
    % 3. The current utility at least as large as the worst in the stepWindow  
    % Also, we force a pass on the first iteration else many quantities undefined
    stopInd = length(valOpt);
    startInd = max(1,stopInd - stepWindow);
    valMin = min(valOpt(startInd:stopInd));
%   pass = (~adaptStep) || (step <= stepMin) || isempty(valMin) || (val >= valMin);
    pass = (it==1) || (~adaptStep) || (step <= stepMin) || (val >= valMin);
    
    % Save the stepsize and pass/fail result if history requested
    if itSaveHist
        estHist.step(itSaveHist) = step;
        estHist.stepMax(itSaveHist) = stepMax;
        estHist.pass(itSaveHist) = pass;
    end
    
    % If pass, set the optimal values and compute a new target shat and snew.
    if (pass)
        
        % Save states that "passed" 
        A2xvarOpt = A2xvar;
        pvarOpt = pvar;
        shatOpt = shat;
        svarOpt = svar;
        rvarOpt = rvar;
        xhatDampOpt = xhatDamp; 
        xhatOpt = xhat;

        % Save record of "passed" utilities 
        if (compVal)
            valOpt = [valOpt val]; %#ok<AGROW> 
        end
        
        % Store variables for export
        phatFinal = phat;
        pvarFinal = pvar;
        zhatFinal = zhat;
        zvarFinal = zvar;
        xhatPrevFinal = xhatFinal; % previous xhat 
        xhatFinal = xhat;
        xvarFinal = xvar;
        rhatFinal = rhat;
        rvarFinal = rvarOpt*scaleFac;   % report unscaled version
        AxhatFinal = Axhat;
        shatFinal = shatOpt/scaleFac;   % report unscaled version
        svarFinal = svarOpt/scaleFac;   % report unscaled version
        
        % Check for convergence
        if (it>1) && (stop==false)
            if (norm(xhatPrevFinal(:) - xhatFinal(:)) / norm(xhatFinal(:)) < tol)
                stop = true;
            elseif customStop==1 
                stop = stopFcn(val, xhatFinal, xhatPrevFinal, AxhatFinal);
            elseif customStop==2 
                S = struct(...
                    'it',it,...
                    'val',val,'xhatPrev',xhatPrevFinal,'Axhat',AxhatFinal, ...
                    'xhat',xhatFinal,'xvar',xvarFinal,...
                    'rhat',rhatFinal,'rvar',rvarFinal,...
                    'phat',phatFinal,'pvar',pvarFinal,...
                    'zhat',zhatFinal,'zvar',zvarFinal,...
                    'shat',shatFinal,'svar',svarFinal ...
                    );
                stop = stopFcn2(S);
            end
        end
        
        % Set scaleFac to mean of pvar if variance-normalization is on.
        % Else scaleFac remains at the initialized value of 1 and has no effect
        if varNorm
            scaleFac = mean(pvarRobust(:));
        end
        
        % Output nonlinear stage
        [zhat,zvar] = scaEstOut.estim(phat,pvarRobust);
        shatNew = (scaleFac./pvarRobust).*(zhat-phat);
        svarNew = (scaleFac./pvarRobust).*(1-min(zvar./pvarRobust,zvarToPvarMax));
        
        % Compute new BB Step size if requested
        if bbStep && it > 2
            % Compute previous step-direction/size weighted with column norms
            sBB = (xhatOpt(1:n,:) - xhatDampOpt(1:n,:));
            
            % Compute new stepsize using columnNorms weighting
            % Select the smallest stepsize over all the columns for a matrix
            % valued signal
            values = sum(abs(sBB .* columnNorms).^2,1) ./...
                sum(abs(A.mult(sBB).^2),1);
            step = min(values);
        end
        
        % Increase stepsize, keeping within bounds
        step = min([stepIncr*max([step stepMin]) stepMax]);
        
    else % if not pass
        
        % Automatically decrease stepMax (when opt.maxBadSteps<Inf)
        failCount = failCount + 1;
        if failCount > maxBadSteps
            failCount = 0;
            stepMax = max(stepMin,maxStepDecr*stepMax);
        end
        
        % Decrease stepsize, keeping within bounds
        step = min(max(stepMin, stepDecr*step),stepMax);
        
        % Check for if stepsize is small enough to trigger termination
        if step < stepTol
            stop = true;
        end
    end % pass
    
    % Save results in history
    if (itSaveHist)
        estHist.phat(:,itSaveHist) = phatFinal(:);
        estHist.pvar(:,itSaveHist) = pvarFinal(:);
        estHist.zhat(:,itSaveHist) = zhatFinal(:);
        estHist.zvar(:,itSaveHist) = zvarFinal(:);
        estHist.shat(:,itSaveHist) = shatFinal(:);
        estHist.svar(:,itSaveHist) = svarFinal(:);
        estHist.rhat(:,itSaveHist) = rhatFinal(:);
        estHist.rvar(:,itSaveHist) = rvarFinal(:);
        estHist.xhat(:,itSaveHist) = xhatFinal(:);
        estHist.xvar(:,itSaveHist) = xvarFinal(:);
        estHist.Axhat(:,itSaveHist) = AxhatFinal(:);
        estHist.val(itSaveHist) = val; % includes "failed" utilities
        estHist.scaleFac(itSaveHist) = scaleFac;
    end
    
    % Print results
    if (verbose)
        fprintf(1,'it=%3d  val=%12.4e  stepsize=%f  |dx|/|x|=%12.4e\n', ...
            it, val, step, norm(xhatPrevFinal(:) - xhatFinal(:)) / norm(xhatFinal(:)));
    end
    
    % Apply damping to shat, svar, and xhat
    if (it==1)
        if any(isnan(svarOpt)), % if user didn't specify opt.svar0
            svarOpt = svarNew;  % equivalent to making step=1
        end
        if any(isnan(xhatDampOpt)), % if user didn't specify opt.xhatPrev0
            xhatDampOpt = xhatOpt; % equivalent to making step=1
        end
    end
    shat = (1-step)*shatOpt + step*shatNew;
    svar = (1-step)*svarOpt + step*svarNew;
    svar(abs(svar)<eps) = eps; % prevents infinite rvar (in which case rhat is often NaN)
    xhatDamp = (1-step)*xhatDampOpt + step*xhatOpt;
    
    % Step in rvar
    rvar = 1./A.multSqTr(svar);   % rvar = 1./((A.^2)*svar)  
    if rvarStep
        if (it==1)
            if any(isnan(rvarOpt)),    % if user didn't specify opt.rvarOpt0
                rvarOpt = rvar;        % equivalent to making step=1
            end
        end
        rvar = (1-step)*rvarOpt + step*rvar;
    end

    % Input linear stage
    rhat = xhatDamp + rvar.*(A.multTr(shat)); % rhat = xhat + rvar.*(A'*shat)
    rvarRobust = max(rvar, rvarMin);  % At very high SNR, use very small rvarMin!
    
    % Input nonlinear stage
    if compVal
        % Send messages to input estimation function.
        if MsgUpdateFlag
            valMsg = scaEstIn.msgUpdate(it, rhat, rvarRobust);
        else
            valMsg = 0;
        end

        % Compute mean, variance, and negative KL-divergence
        [xhat,xvar,valIn] = scaEstIn.estim(rhat, rvarRobust*scaleFac);
        valIn = sum( valIn(:) ) + valMsg;
    else
        % Compute mean and variance 
        [xhat,xvar] = scaEstIn.estim(rhat, rvarRobust*scaleFac);
    end

    elapsed_time(it+1) = elapsed_time(it) + toc;
end % main loop

% Store "next" (i.e., post-"final") estimates for export
xhatNext = xhat;
xvarNext = xvar;
shatNext = shat/scaleFac;
svarNext = svar/scaleFac;


% Trim the history if early termination occurred
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
        estHist.pass = estHist.pass(nitTrim);
        estHist.val = estHist.val(nitTrim);
        estHist.scaleFac = estHist.scaleFac(nitTrim);
        estHist.step = estHist.step(nitTrim);
        estHist.stepMax = estHist.stepMax(nitTrim);
    end
    estHist.it = (nitTrim(:))*histIntvl;
end

% Trim the outputs if mean removal was turned on
if (opt.removeMean)
    xhatNext = A.contract(xhatNext);
    xvarNext = A.contract(xvarNext);
    xhatDamp = A.contract(xhatDamp);
    xhatFinal = A.contract(xhatFinal);
    xvarFinal = A.contract(xvarFinal);
    xhatPrevFinal = A.contract(xhatPrevFinal);
    AxhatFinal = A.contract(AxhatFinal);
    phatFinal = A.contract(phatFinal);
    pvarFinal = A.contract(pvarFinal);
    pvarOpt = A.contract(pvarOpt);
    A2xvarOpt = A.contract(A2xvarOpt);
    zhatFinal = A.contract(zhatFinal);
    zvarFinal = A.contract(zvarFinal);
    shatFinal = A.contract(shatFinal);
    svarFinal = A.contract(svarFinal);
    shatNext = A.contract(shatNext);
    svarNext = A.contract(svarNext);
    rhatFinal = A.contract(rhatFinal);
    rvarFinal = A.contract(rvarFinal);
end

% Export outputs
if (legacyOut)
    out1 = xhatFinal;
    out2 = xvarFinal;
    out3 = rhatFinal;
    out4 = rvarFinal;
    out5 = shatFinal;
    out6 = svarFinal;
    out7 = zhatFinal;
    out8 = zvarFinal;
    out10 = elapsed_time;
    if (saveHist)
        out9 = estHist;
    end
else
    estFin.xhat = xhatFinal;
    estFin.xvar = xvarFinal;
    estFin.phat = phatFinal;
    estFin.pvar = pvarFinal;
    estFin.zhat = zhatFinal;
    estFin.zvar = zvarFinal;
    estFin.shat = shatFinal;
    estFin.svar = svarFinal;
    estFin.rhat = rhatFinal;
    estFin.rvar = rvarFinal;
    estFin.Axhat = AxhatFinal; 
    estFin.xhatPrev = xhatPrevFinal; % legacy warm-start
    estFin.xhatNext = xhatNext; % new warm-start
    estFin.xvarNext = xvarNext; % new warm-start
    estFin.xhatDamp = xhatDamp; % new warm-start
    estFin.pvarOpt = pvarOpt; % new warm-start
    estFin.rvarOpt = rvarOpt; % new warm-start
    estFin.A2xvarOpt = A2xvarOpt; % new warm-start
    estFin.shatNext = shatNext; % new warm-start
    estFin.svarNext = svarNext; % new warm-start
    estFin.val = val;
    estFin.valIn = valIn;
    estFin.valOpt = valOpt;
    estFin.scaleFac = scaleFac;
    estFin.step = step;
    estFin.stepMax = stepMax;
    estFin.failCount = failCount;
    estFin.nit = it;
    estFin.elapsed_time = elapsed_time;
    out1 = estFin;
    out2 = opt;
    if (saveHist)
        out3 = estHist;
    end
end;
