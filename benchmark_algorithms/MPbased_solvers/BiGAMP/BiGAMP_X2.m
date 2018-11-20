function [estFin, optFin, estHist, state] = ...
    BiGAMP_X2(gX, gA, gX2, A2, gOut, problem, opt, state)
% BiGAMP_X2:  Bilinear Generalized Approximate Message Passing X2 Variant
%
% The BiG-AMP X2 algorithm is intended for the estimation of
% random matrices A, X, and X2 observed through the Markov Chain
%
%   X,X2,A -> Z = A*X + A2*X2 -> Y,
%
% where the components of X, X2, and A are independent and the mapping Z -> Y is
% separable. X is NxL, A is MxN, and Z,Y are consequently MxL. A2 is
% assumed to be known and of size M by N2. X2 is thus N2 by L. 
%
%
% The function takes seven arguments:
%
% INPUTS:
% -------
% gX:  An input estimator derived from the EstimIn class
%    based on the input distribution p_x_{nl}(x_nl).
% gA:  An input estimator derived from the EstimIn class
%    based on the input distribution p_a_{mn}(a_mn).
% gX2: An input estimator derived from the Estimin class for the
%     entries of X2. 
% A2: A matrix or LinTrans object to implement A2.
% gOut:  An output estimator derived from the EstimOut class
%    based on the output distribution p_{Y|Z}(y_ml|z_ml).
% problem: An objet of the class BiGAMPProblem specifying the problem
%   setup, including the matrix dimensions and observation locations
% opt (optional):  A set of options of the class BiGAMPOpt.
% state (optional): A structure containing all the values needed to warm
%   start BiG-AMP
%
% OUTPUTS:
% --------
% estFin: Structure containing final BiG-AMP outputs
% optFin: The BiGAMPOpt object used
% estHist: Structure containing per iteration metrics about the run
% state: The values of all parameters required to warm start the algorithm

%% Setup

% Get options
if (nargin < 7)
    opt = BiGAMPOpt();
elseif (isempty(opt))
    opt = BiGAMPOpt();
end
nit     = opt.nit;              % number of iterations
nitMin  = opt.nitMin;           % minimum number of iterations
step    = opt.step;             % step size
stepMin = opt.stepMin;          % minimum step size
stepMax = opt.stepMax;          % maximum step size
stepFilter = opt.stepFilter;    % step filter setting, <1 for no effect
adaptStep = opt.adaptStep;      % adaptive step size
stepIncr = opt.stepIncr;        % step inc on succesful step
stepDecr = opt.stepDecr;        % step dec on failed step
stepWindow = opt.stepWindow;    % step size check window size
verbose = opt.verbose;          % Print results in each iteration
diagnostics = opt.diagnostics;  % Save diagnostic information
tol = opt.tol;                  % Convergence tolerance
stepTol = opt.stepTol;          % minimum allowed step size
pvarStep = opt.pvarStep;        % incldue step size in pvar/zvar
uniformVariance =...
    opt.uniformVariance;        % use scalar variances
varNorm = opt.varNorm;          % normalize variances
compVal = adaptStep;            % only compute cost function for adaptive
gainMode = opt.gainMode;  %#ok<NASGU>
maxBadSteps = opt.maxBadSteps;  % maximum number of allowed bad steps
maxStepDecr = opt.maxStepDecr;  % amount to decrease maxStep after failures
zvarToPvarMax = opt.zvarToPvarMax;  % maximum zvar/pvar ratio

%This option was included for testing purposes and should not be modified
gainMode = opt.gainMode;
if gainMode ~= 1
    warning(...
        'Running with gainMode other than 1 is not recommended') %#ok<*WNTAG>
end

%Warn user about diagnostics
if diagnostics
    warning('Using diagnostics will slow performance') 
end

%Determine requested outputs
saveEM = opt.saveEM;
saveHist = (nargout >= 3);
saveState = (nargout >= 4);

%Get problem dimensions
M = problem.M;
L = problem.L;
N = problem.N;

%Check for partial observation of Z
rLoc = problem.rowLocations;
cLoc = problem.columnLocations;
maskFlag = ~isempty(rLoc);

%Check lengths
if length(rLoc) ~= length(cLoc)
    error('rowLocations and columnLocations must be same length')
end

%Check for sparse mode
sparseMode = opt.sparseMode;

%Sparse mode is not yet implemented for X2 variant
if sparseMode
    error('Sparse mode not yet implemented for X2 variant')
end

%Check mask flag
if sparseMode && ~maskFlag
    error('Must set rowLocations and columnLocations for sparse mode');
end

%If in sparse mode, ensure that sparseMult2 has been mexed
if sparseMode
   
    %Check if it is here
    if isempty(which('sparseMult2'))
        try
            %Try to mex the function
            mex sparseMult2.c -largeArrayDims
        catch %#ok<CTCH>
            %Something went wrong
            error('Unable to mex sparseMult2- must run with sparseMode=false')
            
        end
    end
    
end

%Create sparse matrix multiplication operator. This operator multiplies two
%full matrices of sizes MxN and NxL and returns a vector of the observed
%entries in the MxL product specified by rLoc and cLoc.
if sparseMode
    totalDataPoints = length(rLoc);
    sMult = @(arg1,arg2) sparseMult2(arg1.',arg2,rLoc,cLoc);
    
    %Check if a full Z is computed for diagnostic purposes
    if saveHist
        zErrorFlag = nargin(opt.error_function) > 0;
    end
end

%Now require X2
flagX2 = true;

%Handle the A2 matrix if X2 was provided
if flagX2
    
    if isnumeric(A2) %check to see if it is numeric
        A2 = MatrixLinTrans(A2);
    end
    
    %Determine size of X2
    [~,N2] = A2.size();
end


%Setup for masked case
if maskFlag
    %Indices of observed entries
    omega = sub2ind([M L],rLoc,cLoc);
    
    %Create mask matrix if not in sparse mode
    if ~sparseMode
        maskMatrix = zeros(M,L);
        maskMatrix(omega) = 1;
    end
end

%Assign Avar and xvar mins
xvarMin = opt.xvarMin;
Avarmin = opt.AvarMin;

%Preallocate storage for estHist if user requires it
if (saveHist)
    estHist.errZ = zeros(nit,1);
    estHist.errX = zeros(nit,1);
    if flagX2
        estHist.errX2 = zeros(nit,1);
    end
    estHist.errA = zeros(nit,1);
    estHist.val = zeros(nit,1);
    estHist.step = zeros(nit,1);
    estHist.pass = false(nit,1);
    estHist.timing = zeros(nit,1);
    if diagnostics %Only create these if diagnostics are requested
        estHist.pvarMin = zeros(nit,1);
        estHist.pvarMax = zeros(nit,1);
        estHist.pvarMean = zeros(nit,1);
        estHist.zvarMin = zeros(nit,1);
        estHist.zvarMax = zeros(nit,1);
        estHist.zvarMean = zeros(nit,1);
        estHist.AvarMin = zeros(nit,1);
        estHist.AvarMax = zeros(nit,1);
        estHist.AvarMean = zeros(nit,1);
        estHist.xvarMin = zeros(nit,1);
        estHist.xvarMax = zeros(nit,1);
        estHist.xvarMean = zeros(nit,1);
        estHist.svarMin = zeros(nit,1);
        estHist.svarMax = zeros(nit,1);
        estHist.svarMean = zeros(nit,1);
        estHist.qvarMin = zeros(nit,1);
        estHist.qvarMax = zeros(nit,1);
        estHist.qvarMean = zeros(nit,1);
        estHist.rvarMin = zeros(nit,1);
        estHist.rvarMax = zeros(nit,1);
        estHist.rvarMean = zeros(nit,1);
        estHist.normA = zeros(nit,1);
        estHist.normX = zeros(nit,1);
        estHist.normZ = zeros(nit,1);
    end
end


%% Initialization

%Check for provided state
if nargin < 8
    state = [];
end

if isempty(state) %if no state is provided
    
    %Initialize Avar
    [Ahat,Avar] = gA.estimInit();
    
    %Initialize Xvar
    [xhat,xvar] = gX.estimInit();
    
    %Initialize X2
    if flagX2
        [x2hat,x2var] = gX2.estimInit();
        if length(x2hat) == 1
            x2hat = x2hat*ones(N2,L); 
        end
    end
    
    
    %Handle case of scalar input distribution estimator on xhat
    if (length(xhat) == 1)
        xhat = gX.genRand([N L]);
    end
    
    
    %Handle case of scalar input distribution estimator on Ahat
    if (length(Ahat) == 1)
        Ahat = gA.genRand([M N]);
    end
    
    
    %Initialize valIn
    valIn = 0;
    
    %Replace these defaults with the warm start values if provided in the
    %options object
    if ~isempty(opt.xhat0)
        xhat = opt.xhat0;
        %If warm starting, set valIn to be a negative inf to avoid problems
        valIn = -inf;
    end
    if ~isempty(opt.xvar0)
        xvar = opt.xvar0;
    end
    if ~isempty(opt.Ahat0)
        Ahat = opt.Ahat0;
    end
    if ~isempty(opt.Avar0)
        Avar = opt.Avar0;
    end
    
    %Warm starting of X2
    if flagX2
        if ~isempty(opt.x2hat0)
            x2hat = opt.x2hat0;
        end
        if ~isempty(opt.x2var0)
            x2var = opt.x2var0;
        end
    end
    
    %Handle uniform variances
    if (length(Avar) == 1)
        Avar = repmat(Avar,M,N);
    end
    if (length(xvar) == 1)
        xvar = repmat(xvar,N,L);
    end
    if flagX2
        if length(x2var) == 1
            x2var = repmat(x2var,N2,L); 
        end
    end
    
    %Placeholder initializations- values are not used
    xhatBar = 0;
    AhatBar = 0;
    if flagX2
        x2hatBar = 0;
    end
    shat = 0;
    svar = 0;
    pvarOpt = 0;
    zvarOpt = 0;
        
    %Scalar variance
    if uniformVariance
        Avar = mean(Avar(:));
        xvar = mean(xvar(:));
        if flagX2
            x2var = mean(x2var(:));
        end
    end
    
    %Address warm starting of shat0
    if ~isempty(opt.shat0)
        shat = opt.shat0;
        
        %Make sure that size is correct if in sparse mode
        if sparseMode
            if numel(shat) > totalDataPoints
                shat = reshape(shat(omega),1,[]);
            end
        end
    end
    
    %Init valOpt empty
    valOpt = [];
    
    %Set pvar_mean to unity initially
    pvar_mean = 1;
    
else %Use the provided state information
    
    %A variables
    Ahat = state.Ahat;
    Avar = state.Avar;
    AhatBar = state.AhatBar;
    AhatOpt = state.AhatOpt;
    AhatBarOpt = state.AhatBarOpt;
    
    %X Variables
    xhat = state.xhat;
    xvar = state.xvar;
    xhatBar = state.xhatBar;
    xhatOpt = state.xhatOpt;
    xhatBarOpt = state.xhatBarOpt;
    
    %S variables
    shat = state.shat;
    svar = state.svar;
    shatOpt = state.shatOpt;
    svarOpt = state.svarOpt;
    shatNew = state.shatNew;
    svarNew = state.svarNew;
    
    %Cost stuff
    valIn = state.valIn;
    
    %Variance momentum terms
    pvarOpt = state.pvarOpt;
    zvarOpt = state.zvarOpt;
    
    %Step
    step = state.step;
    
    
    %Old cost values
    valOpt = state.valOpt;
    
    
    %Set pvar_mean
    pvar_mean = state.pvar_mean;
end

%Set pvarMin
pvarMin = opt.pvarMin;

%Placeholder initializations
rhat = 0;
rvar = 0;
qhat = 0;
qvar = 0;

%Placeholders for X2
if flagX2
    r2hat = 0;
    r2var = 0;
end

%Cost init
val = zeros(nit,1);
zhatOpt = 0;
testVal = inf;

%% Iterations

%Start timing first iteration
tstart = tic;

%Control variable to end the iterations
stop = false;
it = 0;
failCount = 0;

%Handle first step
if isempty(state)
    step1 = 1;
else
    step1 = step;
end

% Main iteration loop
while ~stop
    
    % Iteration count
    it = it + 1;
    
    % Check for final iteration
    if it >= nit
        stop = true;
    end
    

    if ~uniformVariance
        %Precompute squares quantities for use, these change on every
        %iteration
        Ahat2 = abs(Ahat).^2;
        xhat2 = abs(xhat).^2;
        
        %Compute zvar
        if ~sparseMode
            zvar = Avar*xhat2 + Ahat2*xvar;
            
            %Add X2 component
            if flagX2
                zvar = zvar + A2.multSq(x2var);
            end
        else
            zvar = sMult(Avar,xhat2) + sMult(Ahat2,xvar);
        end
        
        %Compute pvar
        if ~sparseMode
            pvar = zvar + Avar*xvar; %Note that A2var is by design 0, so no X2 term
        else
            pvar = zvar + sMult(Avar,xvar);
        end
        
    else
        
        %Compute useful quantities
        mxhat2 = norm(xhat,'fro').^2/numel(xhat);
        mAhat2 = norm(Ahat,'fro').^2/numel(Ahat);
        mAvar = norm(Avar(:),1)/numel(Avar);
        mxvar = norm(xvar(:),1)/numel(xvar);
        
        %Compute variances
        zvar = (mAvar*mxhat2 + mAhat2*mxvar)*N;
        pvar = zvar + N*mAvar*mxvar;
    end
    
    %Include pvar step
    if pvarStep
        pvar = step1*pvar + (1-step1)*pvarOpt;
        zvar = step1*zvar + (1-step1)*zvarOpt;
    end
    
    %Update zhat
    if ~sparseMode
        zhat = Ahat * xhat;
        if flagX2
            zhat = zhat + A2.mult(x2hat);
        end
    else
        zhat = sMult(Ahat,xhat);
    end
    
    % Compute log likelihood at the output and add it the total negative
    % K-L distance at the input.
    if (compVal)
        if ~maskFlag || sparseMode
            valOut = sum(sum(gOut.logLike(zhat,pvar)));
        else
            valOut = sum(sum(maskMatrix .* gOut.logLike(zhat,pvar)));
        end
        val(it) = valOut + valIn;
    end
    
    % Determine if candidate passed
    if ~isempty(valOpt)
        
        %Check against worst value in last stepWindow good steps
        stopInd = length(valOpt);
        startInd = max(1,stopInd - stepWindow);
        
        %Check the step
        pass = (val(it) > min(valOpt(startInd:stopInd))) ||...
            ~adaptStep || (step <= stepMin);
        
    else
        pass = true;
    end
    
    
    %Save the step size and pass result if history requested
    if saveHist
        estHist.step(it) = step;
        estHist.pass(it) = pass;
    end
    
    
    
    % If pass, set the optimal values and compute a new target shat and
    % snew.
    if (pass)
        
        %Slightly inrease step size after pass if using adaptive steps
        if adaptStep
            step = stepIncr*step;
        end
        
        % Set new optimal values
        shatOpt = shat;
        svarOpt = svar;
        xhatBarOpt = xhatBar;
        xhatOpt = xhat;
        if flagX2
            x2hatBarOpt = x2hatBar;
            x2hatOpt = x2hat;
        end
        AhatBarOpt = AhatBar;
        AhatOpt = Ahat;
        pvarOpt = pvar;
        zvarOpt = zvar;
        
        %Bound pvar
        pvar = max(pvar, pvarMin);
        
        %We keep a record of only the succesful step valOpt values
        valOpt = [valOpt val(it)]; %#ok<AGROW>
        
        % Continued output step
        phat = zhat...
            - shat.*(zvar/pvar_mean);%...
        %+ abs(shat).^2 .* ( (Ahat .* Avar) * (xhat .* xvar)  );
        
        %Compute mean of pvar
        if varNorm
            pvar_mean = mean(pvar(:));
        end
        
        % Output nonlinear step
        [zhat0,zvar0] = gOut.estim(phat,pvar);
        
        %Compute 1/pvar
        pvarInv = pvar_mean ./ pvar;
        
        %Update the shat quantities
        if maskFlag && ~sparseMode
            pvarInvMask = pvarInv .* maskMatrix;
            shatNew = pvarInvMask.*(zhat0-phat);
            svarNew = pvarInvMask.*(1-min(zvar0./pvar,zvarToPvarMax));
        else
            shatNew = pvarInv.*(zhat0-phat);
            svarNew = pvarInv.*(1-min(zvar0./pvar,zvarToPvarMax));
        end
        
        
        %Scalar Variance
        if uniformVariance
            if ~sparseMode
                svarNew = mean(svarNew(:));
            else
                if numel(svarNew) == 1
                    %If this is scalar, then we will assume it is the
                    %average over the observed entries
                    svarNew = svarNew*totalDataPoints/(M*L);
                else
                    %Otherwise, we average it with all the other zero
                    %entries in svar
                    svarNew = sum(svarNew) / (M*L);
                end
            end
        end
        
        %Enforce step size bounds
        step = min([max([step stepMin]) stepMax]);
        
    else
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
    end
    
    
    
    % Save results
    if (saveHist)
        
        %Record timing information
        if it > 1
            estHist.timing(it) = estHist.timing(it-1) + toc(tstart);
        else
            estHist.timing(it) = toc(tstart);
        end
        
        %Compute the Z error only if needed
        if ~sparseMode
            if ~flagX2
                estHist.errZ(it) = opt.error_function(zhat);
            else
                estHist.errZ(it) = opt.error_function(Ahat*xhat);
            end
        else
            if zErrorFlag
                zhatFull = Ahat*xhat;
                estHist.errZ(it) = opt.error_function(zhatFull);
            end
        end
        estHist.errA(it) = opt.error_functionA(Ahat);
        estHist.errX(it) = opt.error_functionX(xhat);
        if flagX2
            estHist.errX2(it) = opt.error_functionX2(x2hat);
        end
        estHist.val(it) = val(it);
        if diagnostics
            estHist.pvarMin(it) = min(pvar(:));
            estHist.pvarMax(it) = max(pvar(:));
            estHist.pvarMean(it) = mean(pvar(:));
            estHist.zvarMin(it) = min(zvar(:));
            estHist.zvarMax(it) = max(zvar(:));
            estHist.zvarMean(it) = mean(zvar(:));
            estHist.AvarMin(it) = min(Avar(:));
            estHist.AvarMax(it) = max(Avar(:));
            estHist.AvarMean(it) = mean(Avar(:));
            estHist.xvarMin(it) = min(xvar(:));
            estHist.xvarMax(it) = max(xvar(:));
            estHist.xvarMean(it) = mean(xvar(:));
            estHist.svarMin(it) = min(svar(:));
            estHist.svarMax(it) = max(svar(:));
            estHist.svarMean(it) = mean(svar(:));
            estHist.qvarMin(it) = min(qvar(:));
            estHist.qvarMax(it) = max(qvar(:));
            estHist.qvarMean(it) = mean(qvar(:));
            estHist.rvarMin(it) = min(rvar(:));
            estHist.rvarMax(it) = max(rvar(:));
            estHist.rvarMean(it) = mean(rvar(:));
            estHist.normA(it) = norm(Ahat,'fro');
            estHist.normX(it) = norm(xhat,'fro');
            if ~sparseMode
                estHist.normZ(it) = norm(zhat,'fro');
            else
                if zErrorFlag
                    estHist.normZ(it) = norm(zhatFull,'fro');
                end
            end
        end
    end
    
  
    % Check for convergence if step was succesful
    if pass
        if any(isnan(zhat(:))) || any(isinf(zhat(:)))
            stop = true;
        else
            testVal = norm(zhat(:) - zhatOpt(:)) / norm(zhat(:));
            if (it > 1) && ...
                    (testVal < tol)
                stop = true;
            end
        end
        
        %Set other optimal values- not actually used by iterations
        AvarOpt = Avar;
        xvarOpt = xvar;
        if flagX2
            x2varOpt = x2var;
        end
        zhatOpt = zhat;
        
        %Save EM variables if requested
        if saveEM
            rhatFinal = rhat;
            rvarFinal = pvar_mean*rvar;
            qhatFinal = qhat;
            qvarFinal = pvar_mean*qvar;
            zvarFinal = zvar0;
            pvarFinal = pvar;
            if flagX2
                r2hatFinal = r2hat;
                r2varFinal = pvar_mean*r2var;
            end
        end
    end
    
    % Print results
    if (verbose)
        if ~saveHist
            fprintf(1,'it=%3d value=%12.4e step=%f\n', it, testVal, step1);
        else
            fprintf(1,...
                'it=%3d value=%12.4e errZ=%f step=%f\n',...
                it, testVal,estHist.errZ(it), step1);
        end
    end
    
    %Start timing next iteration
    tstart = tic;
    
    % Create new candidate shat
    if it > 1 || ~isempty(state)
        step1 = step;
        if stepFilter >= 1
            step1 = step1*it/(it+stepFilter);
        end
    end
    shat = (1-step1)*shatOpt + step1*shatNew;
    svar = (1-step1)*svarOpt + step1*svarNew;
    xhatBar = (1-step1)*xhatBarOpt + step1*xhatOpt;
    AhatBar = (1-step1)*AhatBarOpt + step1*AhatOpt;
    if flagX2
        x2hatBar = (1-step1)*x2hatBarOpt + step1*x2hatOpt;
    end
    
    %For sparse mode, need to construct sparse matrix versions of shat and
    %svar for multiplication
    if sparseMode
        shatMat = sparse(rLoc,cLoc,shat,M,L);
        if ~uniformVariance
            svarMat = sparse(rLoc,cLoc,svar,M,L);
        end
    end
    
    %Compute rvar and correct for infinite variance
    if uniformVariance
        mAhatBar2 = norm(AhatBar,'fro')^2/numel(AhatBar);
        rvar = 1/(mAhatBar2*svar*M);
    else
        if ~sparseMode
            rvar = 1./((abs(AhatBar).^2)'*svar);
            if flagX2
                r2var = 1./ A2.multSqTr(svar);
            end
        else
            rvar = 1./((abs(AhatBar).^2)'*svarMat);
        end
    end
    rvar(rvar > opt.varThresh) = opt.varThresh;
    if flagX2
        r2var(r2var > opt.varThresh) = opt.varThresh; %#ok<AGROW>
    end
    
    %Update rhat
    if ~sparseMode
        switch gainMode
            case 1,
                rGain = (1 - (rvar.*(Avar'*svar)));
            case 2,
                rGain = (1 - (rvar.*(Avar'*shat.^2)));
            case 3,
                rGain = 1;
        end
        rGain = min(1,max(0,rGain));
        rhat = xhatBar.*rGain + rvar.*(AhatBar'*shat);
        if flagX2
            r2hat = x2hatBar + r2var.*(A2.multTr(shat));
        end
    else
        error('Sparse mode not implemented yet for BiG-AMP X2')
    end
    rvar = max(rvar, xvarMin);
    if flagX2
        r2var = max(r2var,xvarMin);
    end
    
    % Input linear step for A
    if uniformVariance
        mxhatBar2 = norm(xhatBar,'fro')^2/numel(xhatBar);
        qvar = 1/(svar*mxhatBar2*L);
    else
        if ~sparseMode
            qvar = 1./(svar*(abs(xhatBar).^2)');
        else
            qvar = 1./(svarMat*(abs(xhatBar).^2)');
        end
    end
    qvar(qvar > opt.varThresh) = opt.varThresh;
    
    
    %Update qhat
    if ~sparseMode
        qhat = AhatBar + qvar.*(shat*xhatBar');
    else
        qhat = AhatBar + qvar.*(shatMat*xhatBar');
    end
    qvar = max(qvar,Avarmin);
    
    % Input nonlinear step
    if compVal
        [xhat,xvar,valInX] = gX.estim(rhat, rvar*pvar_mean);
        [Ahat,Avar,valInA] = gA.estim(qhat, qvar*pvar_mean);
        if flagX2
            [x2hat,x2var,valInX2] = gX2.estim(r2hat, r2var*pvar_mean);
        end
    else %method may avoid computation if the vals are not needed
        [xhat,xvar] = gX.estim(rhat, rvar*pvar_mean);
        [Ahat,Avar] = gA.estim(qhat, qvar*pvar_mean);
        if flagX2
            [x2hat,x2var] = gX2.estim(r2hat, r2var*pvar_mean);
        end
    end
    
    %Scalar variances
    if uniformVariance
        Avar = mean(Avar(:));
        xvar = mean(xvar(:));
        if flagX2
            x2var = mean(x2var(:));
        end
    end
    
    %Update valIn
    if compVal
        valIn = sum( valInX(:) ) + sum ( valInA(:) );
        if flagX2
            valIn = valIn + sum( valInX2(:) );
        end
    end
    
    %Don't stop before minimum iteration count
    if it < nitMin
        stop = false;
    end
    

end



%% Save the final values


%Save the options object that was used
optFin = opt;

%Estimates of the two matrix factors
estFin.xhat = xhatOpt;
estFin.xvar = xvarOpt;
estFin.Ahat = AhatOpt;
estFin.Avar = AvarOpt;
estFin.x2hat = x2hatOpt;
estFin.x2var = x2varOpt;


%Save values useful for EM learning
if saveEM
    estFin.rhat = rhatFinal;
    estFin.rvar = rvarFinal;
    estFin.qhat = qhatFinal;
    estFin.qvar = qvarFinal;
    estFin.r2hat = r2hatFinal;
    estFin.r2var = r2varFinal;
    estFin.zvar = zvarFinal;
    estFin.phat = phat;
    estFin.pvar = pvarFinal;
end

%% Cleanup
%Trim the outputs if early termination occurred
if saveHist && (it < nit)
    estHist.errZ = estHist.errZ(1:it);
    estHist.errA = estHist.errA(1:it);
    estHist.errX = estHist.errX(1:it);
    if flagX2
        estHist.errX2 = estHist.errX2(1:it);
    end
    estHist.val = estHist.val(1:it);
    estHist.step = estHist.step(1:it);
    estHist.pass = estHist.pass(1:it);
    estHist.timing = estHist.timing(1:it);
    if diagnostics
        estHist.pvarMin = estHist.pvarMin(1:it);
        estHist.pvarMax = estHist.pvarMax(1:it);
        estHist.pvarMean = estHist.pvarMean(1:it);
        estHist.zvarMin = estHist.zvarMin(1:it);
        estHist.zvarMax = estHist.zvarMax(1:it);
        estHist.zvarMean = estHist.zvarMean(1:it);
        estHist.AvarMin = estHist.AvarMin(1:it);
        estHist.AvarMax = estHist.AvarMax(1:it);
        estHist.AvarMean = estHist.AvarMean(1:it);
        estHist.xvarMin = estHist.xvarMin(1:it);
        estHist.xvarMax = estHist.xvarMax(1:it);
        estHist.xvarMean = estHist.xvarMean(1:it);
        estHist.svarMin = estHist.svarMin(1:it);
        estHist.svarMax = estHist.svarMax(1:it);
        estHist.svarMean = estHist.svarMean(1:it);
        estHist.qvarMin = estHist.qvarMin(1:it);
        estHist.qvarMax = estHist.qvarMax(1:it);
        estHist.qvarMean = estHist.qvarMean(1:it);
        estHist.rvarMin = estHist.rvarMin(1:it);
        estHist.rvarMax = estHist.rvarMax(1:it);
        estHist.rvarMean = estHist.rvarMean(1:it);
        estHist.normA = estHist.normA(1:it);
        estHist.normX = estHist.normX(1:it);
        estHist.normZ = estHist.normZ(1:it);
    end
end


%% Save the state

if saveState
    
    %A variables
    state.Ahat = Ahat;
    state.Avar = Avar;
    state.AhatBar = AhatBar;
    state.AhatOpt = AhatOpt;
    state.AhatBarOpt = AhatBarOpt;
    
    %X Variables
    state.xhat = xhat;
    state.xvar = xvar;
    state.xhatBar = xhatBar;
    state.xhatBarOpt = xhatBarOpt;
    state.xhatOpt = xhatOpt;
    
    %s variables
    state.shat = shat;
    state.svar = svar;
    state.shatOpt = shatOpt;
    state.shatNew = shatNew;
    state.svarOpt = svarOpt;
    state.svarNew = svarNew;
    
    %Cost stuff
    state.valIn = valIn;
    
    %Variance momentum terms
    state.pvarOpt = pvarOpt;
    state.zvarOpt = zvarOpt;
    
    %Step
    state.step = step;
    
    
    %Old cost values
    state.valOpt = valOpt;
        
    %pvar_mean
    state.pvar_mean = pvar_mean;
end



