function [out1,out2,out3,out4,out5,out6,out7,out8,out9] = ...
    gampEst(scaEstIn, scaEstOut, A, opt)
% gampEst:  G-AMP estimation algorithm
%
% DESCRIPTION:
% ------------
% The G-AMP estimation algorithm is intended for the estimation of a
% random vector x observed through an observation y from the Markov chain
%
%   x -> z = A*x -> y,
%
% where the components of x are independent and the mapping z -> y is
% separable.
%
% SYNTAX:
% -------
% [out1,out2,out3,out4,out5,out6,out7,out8,out9] = ...
%   				gampEst(scaEstIn, scaEstOut, A, opt)
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
%  xhatFinal: final estimate of the vector x.
%  xvarFinal: final quadratic term for vector x.
%  phatFinal: final estimate of the vector p (see below for meaning).
%  pvarFinal: final quadratic term for vector p (see below for meaning).
%  zhatFinal: final estimate of the vector z=Ax.
%  zvarFinal: final quadratic term for vector z=Ax.
%  shatFinal: final estimate of the vector s.
%  svarFinal: final quadratic term for vector s.
%  rhatFinal: final estimate of the vector r (see below for meaning).
%  rvarFinal: final quadratic term for vector r (see below for meaning).
%
% estHist:  History of G-AMP across iterations, including
%   unsuccessful steps that did not "pass" the acceptance criterion.
%   estHist.xhat: history of xhat
%   estHist.xvar: history of xvar
%   estHist.Axhat: history of A*xhat
%   estHist.phat: history of phat
%   estHist.pvar: history of pvar
%   estHist.zhat: history of zhat
%   estHist.zvar: history of zvar
%   estHist.shat: history of shat
%   estHist.svar: history of svar
%   estHist.rhat: history of rhat
%   estHist.rvar: history of rvar
%   estHist.pass: history of step pass/fail
%   estHist.val: history of the "val" (or negative cost)
%   estHist.scaleFac: history of the scalefactor used when varNorm=true
%   estHist.step: history of the stepsize (i.e., damping term)
%   estHist.stepMax: history of the maximum allowed stepsize
%
% optFin:  final settings of GampOpt options object
%
% estFin:  Structure containing final G-AMP outputs
%   estFin.xhat: same as xhatFinal above
%   estFin.xvar: same as xvarFinal above
%   estFin.xhatPrev: previous version of xhat
%   estFin.xhatNext: next version of xhat (used for warm start)
%   estFin.xvarNext: next version of xvar (used for warm start)
%   estFin.Axhat: same as AxhatFinal above
%   estFin.phat: same as phatFinal above
%   estFin.pvar: same as pvarFinal above
%   estFin.zhat: same as zhatFinal above
%   estFin.zvar: same as zvarFinal above
%   estFin.shat: same as shatFinal above
%   estFin.svar: same as svarFinal above
%   estFin.rhat: same as rhatFinal above
%   estFin.rvar: same as rvarFinal above
%   estFin.pass: same as above
%   estFin.val: as above
%   estFin.scaleFac: as above
%   estFin.step: as above
%   estFin.stepMax: as above
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
maxBadSteps = opt.maxBadSteps;  % maximum number of allowed bad steps
maxStepDecr = opt.maxStepDecr;  % amount to decrease maxStep after failures
stepTol = opt.stepTol;          % minimum allowed step size
pvarStep = opt.pvarStep;        % include step size in pvar & A2xvar
varNorm = opt.varNorm;          % normalize variances
scaleFac = opt.scaleFac;	% initial variance normalization
pvarMin = opt.pvarMin;		% minimum pvar
rvarMin = opt.xvarMin;		% minimum rvar
zvarToPvarMax = opt.zvarToPvarMax;  % maximum zvar/pvar ratio
histIntvl = opt.histIntvl;  % history interval

% Handle output format
legacyOut = opt.legacyOut;	% use legacy output format?
if (legacyOut),
    saveHist = (nargout >= 9);
    if (nargout > 9),
        error('too many output arguments')
    end;
else % modern output format
    saveHist = (nargout >= 3);
    if (nargout > 3),
        error('too many output arguments')
    end;
end
warnOut = opt.warnOut;		% warn about change in output format?
if (warnOut),
    warning('Note that as of June 2013 the output format of gampEst changed from 9 arguments to 2 arguments and subsequently 3 arguments.  Set GampOpt.legacyOut=true to use legacy output format from before June 2013.')
end

% Determine whether the cost function must be computed each iteration
if adaptStep,
    compVal = true; 		% must be true
else
    compVal = false; 		% can set at false for faster runtime
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
[xhat0,xvar0] = scaEstIn.estimInit();
valIn = -Inf;	% the valIn returned by estimInit can be problematic!
val = -Inf;     % placeholder if adaptive step size is not used

%Replace default initialization with warm start values if provided
if ~isempty(opt.xhat0)
    xhat0 = opt.xhat0;
end
if ~isempty(opt.xvar0)
    xvar0 = opt.xvar0;
end
%if ~isempty(opt.valIn)
%    valIn = opt.valIn;	% should implement this if we want to be picky
%end

% For a scalar output, the same distribution is applied to all components
if (size(xhat0,1) == 1)
    xhat0 = repmat(xhat0,n,1);
end
if (size(xvar0,1) == 1)
    xvar0 = repmat(xvar0,n,1);
end

% Make sure that number of input columns match number of output columns
if (size(xhat0,2) == 1)
    xhat0 = repmat(xhat0,1,s);
end
if (size(xvar0,2) == 1)
    xvar0 = repmat(xvar0,1,s);
end

% Continue with initialization
shat0 = zeros(m,s);	% default value is zero
svar0 = nan(m,s);	% will test for NaN later
xhatPrev0 = nan(n,s);	% will test for NaN later

% Replace default initialization with warm start values if provided
if ~isempty(opt.shat0)
    shat0 = opt.shat0*scaleFac;	% variance normalization included
end
if ~isempty(opt.svar0)
    svar0 = opt.svar0*scaleFac; 	% variance normalization included
end
if ~isempty(opt.xhatPrev0)
    xhatPrev0 = opt.xhatPrev0;
end

% If the mean-removal option is set, create an augmented system
% with the mean removed.  (See LinTransDemeanRC.m % for more details.)
% Then initialize variables.
if (opt.removeMean)
    A = LinTransDemeanRC(A);
    [m,n] = A.size();
    scaEstIn = A.expandIn(scaEstIn);
    scaEstOut = A.expandOut(scaEstOut);
    xhat = A.expandXhat(xhat0);
    xvar = A.expandXvar(xvar0);
    shat = A.expandShat(shat0);
    svar = A.expandSvar(svar0);
    xhatPrev = A.expandXhat(xhatPrev0);
else
    xhat = xhat0;
    xvar = xvar0;
    shat = shat0;
    svar = svar0;
    xhatPrev = xhatPrev0;
end
rhat = nan(size(xhat));		% inconsequential; will be overwritten
rvar = nan(size(xhat));		% inconsequential; will be overwritten

% if uniform-variance mode is requested by the user, implement it by
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
xhatFinal  = nan(n,s);
valOpt = [];
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
    estHist.val = nan(nitSave,1);
    %estHist.valIn = nan(nitSave,1);
    estHist.stepMax = nan(nitSave,1);
    estHist.pass = nan(nitSave,1);
    estHist.scaleFac = nan(nitSave,1);
end

% Check for the presence of two methods within the LinTrans and EstimIn
% objects and set flags accordingly
MtxUncertaintyFlag = ismethod(A,'includeMatrixUncertainty');
MsgUpdateFlag = ismethod(scaEstIn, 'msgUpdate');

%If computing BB steps, compute column norms for use in scaling
if bbStep
    columnNorms = A.multSqTr(ones(m,1)).^0.5;
    columnNorms = repmat(columnNorms,1,s);
end

%Control variable to end the iterations
stop = false;
it = 0;

%Init to dummy value
step1 = 1;		% over-rides step in first half of first iteration
pvarOpt = 0;		% value is inconsequential when step1=1
A2xvarOpt = 0;		% value is inconsequential when step1=1
failCount = 0;  	% Counts failed steps

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

    % Output linear step with no A uncertainty
    A2xvar = A.multSq(xvar);
    
    %Incorporate A uncertainty
    if MtxUncertaintyFlag
        pvar = A.includeMatrixUncertainty(A2xvar,xhat,xvar);
    else
        pvar = A2xvar;
    end
    
    %Step in pvar
    if pvarStep
        pvar = step1*pvar + (1-step1)*pvarOpt;
        A2xvar = step1*A2xvar + (1-step1)*A2xvarOpt;
    end
    
    %Update Axhat
    Axhat = A.mult(xhat);
    
    % Compute log likelihood at the output and add it the total negative
    % K-L distance at the input.
    if (compVal)
        valOut = sum(sum(scaEstOut.logLike(Axhat,pvar)));
        val = valOut + valIn;
    end
    
    % Determine if candidate passed
    if (it > 1)
        
        %Check against worst value in last stepWindow good steps
        stopInd = length(valOpt);
        startInd = max(1,stopInd - stepWindow);
        
        %Check the step
        pass = (val > min(valOpt(startInd:stopInd))) ||...
            ~adaptStep || (step <= stepMin);
        
    else
        pass = true;
    end
    
    %Save the step size and pass result if history requested
    if itSaveHist
        estHist.step(itSaveHist) = step;
        estHist.stepMax(itSaveHist) = stepMax;
        estHist.pass(itSaveHist) = pass;
    end
    
    % If pass, set the optimal values and compute a new target shat and
    % snew.
    if (pass)
        
        % Set new optimal values
        shatOpt = shat;
        svarOpt = svar;
        xhatPrevOpt = xhatPrev;
        xhatOpt = xhat;
        pvarOpt = pvar;
        A2xvarOpt = A2xvar;
        
        %Set zhat outputs
        phatFinal = phat;
        pvarFinal = pvar;
        zhatFinal = zhat;
        zvarFinal = zvar;
        
        %We keep a record of only the succesful step valOpt values
        valOpt = [valOpt val]; %#ok<AGROW>
        
        %Store previous optimal solution
        xhatPrevFinal = xhatFinal;
        
        % Save current optimal solution
        xhatFinal = xhat;
        xvarFinal = xvar;
        rhatFinal = rhat;
        rvarFinal = rvar*scaleFac;	% report unscaled version
        AxhatFinal = Axhat;
        
        %Set shat and svar outputs
        shatFinal = shatOpt/scaleFac;	% report unscaled version
        svarFinal = svarOpt/scaleFac;	% report unscaled version
        
        % Check for convergence
        if (it>1) && (stop==false)
            if (norm(xhatPrevFinal - xhatFinal) / norm(xhatFinal) < tol)
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
        
        % Continued output step
        phat = Axhat - (A2xvar/scaleFac).*shat; %uses A2xvar rather than pvar
        pvar = max(pvar,pvarMin);
        
        %Set scaleFac to mean of pvar if normalization is on.
        %Else scaleFac remains at the initialized value of 1 and has no effect
        if varNorm
            scaleFac = mean(pvar(:));
        end
        
        % Output nonlinear step -- regular components
        [zhat,zvar] = scaEstOut.estim(phat,pvar);
        shatNew = (scaleFac./pvar).*(zhat-phat);
        svarNew = (scaleFac./pvar).*(1-min(zvar./pvar,zvarToPvarMax));
        
        %Compute new BB Step size if requested
        if bbStep && it > 2
            %Compute previous step direction/size weighted with column
            %norms
            sBB = (xhatOpt(1:n,:) - xhatPrevOpt(1:n,:));
            
            %Compute new step size using columnNorms weighting
            %Select the smallest step over all the columns for a matrix
            %valued signal
            values = sum(abs(sBB .* columnNorms).^2,1) ./...
                sum(abs(A.mult(sBB).^2),1);
            step = min(values);
        end
        
        %Enforce step size bounds
        step = min([stepIncr*max([step stepMin]) stepMax]);
        
    else % not pass
        
        %Check on failure count
        failCount = failCount + 1;
        if failCount > maxBadSteps
            failCount = 0;
            stepMax = max(stepMin,maxStepDecr*stepMax);
        end
        
        % Decrease step size
        step = max(stepMin, stepDecr*step);
        
        %Check for minimum step size
        if step < stepTol
            stop = true;
        end
    end % if pass
    
    % Save results
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
        estHist.val(itSaveHist) = val;
        %estHist.valIn(itSaveHist) = valIn;
        estHist.scaleFac(itSaveHist) = scaleFac;
    end
    
    % Create new candidate shat
    step1 = step;
    if (it==1)
        if any(isnan(svarOpt)), % if user didn't specify svar0
            svarOpt = svarNew; 	% equivalent to making step1=1
        end
        if any(isnan(xhatPrevOpt)), % if user didn't specify xhatPrev0
            xhatPrevOpt = xhatOpt; % equivalent to making step1=1
        end
    end
    shat = (1-step1)*shatOpt + step1*shatNew;
    svar = (1-step1)*svarOpt + step1*svarNew;
    xhatPrev = (1-step1)*xhatPrevOpt + step1*xhatOpt;
    
    % Print results
    if (verbose)
        fprintf(1,'it=%3d  value=%12.4e  step=%f  |dx|/|x|=%12.4e\n', ...
            it, val, step, norm(xhatPrevFinal - xhatFinal) / ...
            norm(xhatFinal));
    end
    
    % Input linear step
    rvar = 1./A.multSqTr(svar);   % rvar = 1./((A.^2)*svar)
    
    %Rest of input linear step
    rhat = xhatPrev + rvar.*(A.multTr(shat)); % rhat = xhatPrev + rvar.*(A'*shat)
    rvar = max(rvar, rvarMin);
    
    % Send messages to input estimation function.
    if MsgUpdateFlag
        valMsg = scaEstIn.msgUpdate(it, rhat, rvar);
    else
        valMsg = 0;
    end
    
    % Input nonlinear step
    if compVal
        [xhat,xvar,valIn] = scaEstIn.estim(rhat, rvar*scaleFac);
        valIn = sum( valIn(:) ) + valMsg;
    else
        [xhat,xvar] = scaEstIn.estim(rhat, rvar*scaleFac);
    end
    
end % main loop

%Save "next" estimates
xhatNext = xhat;
xvarNext = xvar;
shatNext = shat;
svarNext = svar;
pvarNext = pvar;


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
        estHist.pass = estHist.pass(nitTrim);
        estHist.val = estHist.val(nitTrim);
        %estHist.valIn = estHist.valIn(nitTrim);
        estHist.scaleFac = estHist.scaleFac(nitTrim);
        estHist.step = estHist.step(nitTrim);
        estHist.stepMax = estHist.stepMax(nitTrim);
    end
    estHist.it = (nitTrim(:))*histIntvl;
end

%Trim the outputs if mean removal was turned on
if (opt.removeMean)
    xhatNext = A.contract(xhatNext);
    xvarNext = A.contract(xvarNext);
    xhatFinal = A.contract(xhatFinal);
    xhatPrevFinal = A.contract(xhatPrevFinal);
    xvarFinal = A.contract(xvarFinal);
    AxhatFinal = A.contract(AxhatFinal);
    phatFinal = A.contract(phatFinal);
    pvarFinal = A.contract(pvarFinal);
    pvarNext = A.contract(pvarNext);
    zhatFinal = A.contract(zhatFinal);
    zvarFinal = A.contract(zvarFinal);
    shatNext = A.contract(shatNext);
    svarNext = A.contract(svarNext);
    shatFinal = A.contract(shatFinal);
    svarFinal = A.contract(svarFinal);
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
    if (saveHist)
        out9 = estHist;
    end
else
    estFin.xhatNext = xhatNext;
    estFin.xvarNext = xvarNext;
    estFin.xhat = xhatFinal;
    estFin.xvar = xvarFinal;
    estFin.xhatPrev = xhatPrevFinal;
    estFin.Axhat = AxhatFinal;
    estFin.phat = phatFinal;
    estFin.pvar = pvarFinal;
    estFin.pvarNext = pvarNext;
    estFin.zhat = zhatFinal;
    estFin.zvar = zvarFinal;
    estFin.shat = shatFinal;
    estFin.svar = svarFinal;
    estFin.shatNext = shatNext;
    estFin.svarNext = svarNext;
    estFin.rhat = rhatFinal;
    estFin.rvar = rvarFinal;
    estFin.pass = pass;
    estFin.val = val;
    %out1.val = valIn;
    estFin.scaleFac = scaleFac;
    estFin.step = step;
    estFin.stepMax = stepMax;
    estFin.nit = it;
    out1 = estFin;
    out2 = opt;
    if (saveHist)
        out3 = estHist;
    end
end;
