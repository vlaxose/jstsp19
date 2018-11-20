function [estFin,optFin,estHist] = gampEstNoAdapt(scaEstIn, scaEstOut, A, opt)
% gampEstNoAdapt:  G-AMP estimation algorithm ... an implementation without adaptive damping
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
% [estFin,optFin,estHist] = gampEstNoAdapt(scaEstIn, scaEstOut, A, opt)
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
%   estFin.xhat: final estimate of the vector x.
%   estFin.xvar: final quadratic term for vector x.
%   estFin.xhatPrev: previous version of xhat.
%   estFin.xhatNext: next version of xhat (used for warm start).
%   estFin.xvarNext: next version of xvar (used for warm start).
%   estFin.Axhat: final estimate of the vector A*x.
%   estFin.phat: final estimate of the vector p (see below for meaning).
%   estFin.pvar: final quadratic term for vector p (see below for meaning).
%   estFin.zhat: final estimate of the vector z=Ax.
%   estFin.zvar: final quadratic term for vector z=Ax.
%   estFin.shat: final estimate of the vector s.
%   estFin.svar: final quadratic term for vector s.
%   estFin.rhat: final estimate of the vector r (see below for meaning).
%   estFin.rvar: final quadratic term for vector r (see below for meaning).
%   estFin.step: final value of the stepsize (i.e., damping term)
%
% optFin:  Final settings of GampOpt options object
%
% estHist:  History of G-AMP across iterations, 
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
%   estHist.step: history of the stepsize (i.e., damping term)
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
verbose = opt.verbose;          % Print results in each iteration
tol = opt.tol;                  % Convergence tolerance
pvarStep = opt.pvarStep;        % include step size in pvar
pvarMin = opt.pvarMin;		% minimum pvar
rvarMin = opt.xvarMin;		% minimum rvar
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
[xhat0,xvar0] = scaEstIn.estimInit();

%Replace default initialization with warm start values if provided
if ~isempty(opt.xhat0)
    xhat0 = opt.xhat0;
end
if ~isempty(opt.xvar0)
    xvar0 = opt.xvar0;
end

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
xhatDamp0 = nan(n,s);	% will test for NaN later
pvarOpt0 = nan(m,s);   % will test for NaN later 

% Replace default initialization with warm start values if provided
if ~isempty(opt.shat0)
    shat0 = opt.shat0;	
end
if ~isempty(opt.svar0)
    svar0 = opt.svar0; 	
end
if ~isempty(opt.xhatPrev0)     
    xhatDamp0 = opt.xhatPrev0;
end
if ~isempty(opt.pvarOpt0)
    pvarOpt0 = opt.pvarOpt0;
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
    xhatDamp = A.expandXhat(xhatDamp0); 
    pvarOpt = A.expandSvar(pvarOpt0);
else
    xhat = xhat0;
    xvar = xvar0;
    shat = shat0;
    svar = svar0;
    xhatDamp = xhatDamp0;
    pvarOpt = pvarOpt0;
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
%pvar = nan(m,s);
xhatFinal = nan(n,s);
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

    % Output linear step with no A uncertainty
    pvar = A.multSq(xvar);
    
    %Step in pvar
    if pvarStep
        if (it==1)
	    if any(isnan(pvarOpt)),    % if user didn't specify opt.pvarOpt0
	        pvarOpt = pvar;	% equivalent to making step=1
            end
	end
        pvar = (1-step)*pvarOpt + step*pvar;
    end
    pvarOpt = pvar;
    
    %Update Axhat
    Axhat = A.mult(xhat);
    
    %Save the step size if history requested
    if itSaveHist
        estHist.step(itSaveHist) = step;
    end
    
    % Save current solution
    AxhatFinal = Axhat;
    phatFinal = phat;
    pvarFinal = pvar;
    zhatFinal = zhat;
    zvarFinal = zvar;
    shatFinal = shat;
    svarFinal = svar;
    rhatFinal = rhat;
    rvarFinal = rvar;
    xhatPrevFinal = xhatFinal;
    xhatFinal = xhat;
    xvarFinal = xvar;
            
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
    phat = Axhat - pvar.*shat;
    pvar = max(pvar,pvarMin);
        
    % Output nonlinear step -- regular components
    [zhat,zvar] = scaEstOut.estim(phat,pvar);
    shatNew = (1./pvar).*(zhat-phat);
    svarNew = (1./pvar).*(1-zvar./pvar);
        
    %Enforce step size bounds
    step = min([stepIncr*max([step stepMin]) stepMax]);
    
    % Save results
    if (itSaveHist)
        estHist.Axhat(:,itSaveHist) = AxhatFinal(:);
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
    end
    
    % Create new candidate shat
    if (it==1)
        if any(isnan(svar)), % if user didn't specify svar0
            svar = svarNew; 	% equivalent to making step=1
        end
        if any(isnan(xhatDamp)), % if user didn't specify xhatDampPrev0
            xhatDamp = xhat; % equivalent to making step=1
        end
    end
    shat = (1-step)*shat + step*shatNew;
    svar = (1-step)*svar + step*svarNew;
    xhatDamp = (1-step)*xhatDamp + step*xhat;
    
    % Print results
    if (verbose)
        fprintf(1,'it=%3d  step=%f  |dx|/|x|=%12.4e\n', ...
            it, step, norm(xhatDamp - xhat)/norm(xhat));
    end
    
    % Input linear step
    rvar = 1./A.multSqTr(svar);   % rvar = 1./((A.^2)*svar)
    
    %Rest of input linear step
    rhat = xhatDamp + rvar.*(A.multTr(shat)); % rhat = xhat + rvar.*(A'*shat)
    rvar = max(rvar, rvarMin);
    
    % Input nonlinear step
    [xhat,xvar] = scaEstIn.estim(rhat, rvar);
    
end % main loop

%Save "next" estimates
xhatNext = xhat;
xvarNext = xvar;
shatNext = shat;
svarNext = svar;

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
    % Set other quantities computed in non-simplified gampEst
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
    xhatFinal = A.contract(xhatFinal);
    xhatPrevFinal = A.contract(xhatPrevFinal);
    xvarFinal = A.contract(xvarFinal);
    AxhatFinal = A.contract(AxhatFinal);
    phatFinal = A.contract(phatFinal);
    pvarFinal = A.contract(pvarFinal);
    pvarOpt = A.contract(pvarOpt);
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
estFin.xhatNext = xhatNext;
estFin.xvarNext = xvarNext;
estFin.xhatDamp = xhatDamp;
estFin.xhat = xhatFinal;
estFin.xvar = xvarFinal;
estFin.xhatPrev = xhatPrevFinal;
estFin.Axhat = AxhatFinal;
estFin.phat = phatFinal;
estFin.pvar = pvarFinal;
estFin.pvarOpt = pvarOpt;
estFin.rvarOpt = nan; % needed for compatibility with gampEst
estFin.A2xvarOpt = nan; % needed for compatibility with gampEst
estFin.zhat = zhatFinal;
estFin.zvar = zvarFinal;
estFin.shatNext = shatNext;
estFin.svarNext = svarNext;
estFin.shat = shatFinal;
estFin.svar = svarFinal;
estFin.rhat = rhatFinal;
estFin.rvar = rvarFinal;
estFin.pass = 1; % needed for compatibility with gampEst
estFin.valIn = nan; % needed for compatibility with gampEst
estFin.valOpt = []; % needed for compatibility with gampEst
estFin.step = step;
estFin.scaleFac = 1;
estFin.stepMax = stepMax;
estFin.failCount = 0; % needed for compatibility with gampEst
estFin.nit = it;
optFin = opt;
