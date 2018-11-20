function [estFin, optFin, estHist] = ...
    PBiGAMP(gB, gC, gOut, problem, opt)
% PBiGAMP:  Parametric Bilinear Generalized Approximate Message Passing
%
% The P-BiG-AMP algorithm is intended for the estimation of
% random vectors b and c observed through the Markov Chain
%
%   b,c -> z -> y,
%
% where the mapping from b,c -> z is is known and bilinear in b,c. The
% mapping from z -> y is assumed to be known and separable.
% b is Nb x 1, c is Nc x 1, and both y and z are M x 1.
%
% INPUTS:
% -------
% gB:  An input estimator derived from the EstimIn class
%    based on the input distribution p_b_{i}(b_i). If multiple
%    distributions govern groups of entries in b, construct the estimator 
%    using EstimInConcat for proper interaction with uniform variances. 
% gC:  An input estimator derived from the EstimIn class
%    based on the input distribution p_c_{j}(c_j). If multiple
%    distributions govern groups of entries in c, construct the estimator 
%    using EstimInConcat for proper interaction with uniform variances. 
% gOut:  An output estimator derived from the EstimOut class
%    based on the output distribution p_{Y|Z}(y_m|z_m).
% problem: An object of the class PBiGAMPProblem specifying the problem
%   setup, including the dimensions and the mappings A() and X()
% opt (optional):  A set of options of the class PBiGAMPOpt that inclues
%   the state needed for warm-starting.
%
% OUTPUTS:
% --------
% estFin: Structure containing final P-BiG-AMP outputs
% optFin: The PBiGAMPOpt object used
% estHist: Structure containing per iteration metrics about the run


%% Setup

% Get options
if (nargin < 5)
    opt = PBiGAMPOpt();
elseif (isempty(opt))
    opt = PBiGAMPOpt();
end
nit     = opt.nit;              % maximum number of iterations
nitMin  = opt.nitMin;           % minimum number of iterations
step    = opt.step;             % step size (i.e., damping parameter)
stepMin = opt.stepMin;          % minimum step size
stepMax = opt.stepMax;          % maximum step size
stepFilter = opt.stepFilter;    % step filter setting, <1 for no effect
adaptStep = opt.adaptStep;      % adaptive step size
stepIncr = opt.stepIncr;        % step increment on succesful iteration
stepDecr = opt.stepDecr;        % step decrement on failed iteration
stepWindow = opt.stepWindow;    % adaptive step check-window size
verbose = opt.verbose;          % Print results in each iteration
diagnostics = opt.diagnostics;  % Save diagnostic information
tol = opt.tol;                  % Convergence tolerance on change in zhat
normTol = opt.normTol;          % Convergence tolerance on norm of zhat
stepTol = opt.stepTol;          % minimum allowed step size
pvarStep = opt.pvarStep;        % incldue step size in pvar/zvar
uniformVariance =...
    opt.uniformVariance;        % use scalar variances
compVal = adaptStep;            % only compute cost function for adaptive
maxBadSteps = opt.maxBadSteps;  % maximum number of allowed bad steps
maxStepDecr = opt.maxStepDecr;  % amount to decrease maxStep after failures
zvarToPvarMax = opt.zvarToPvarMax;  % maximum zvar/pvar ratio, inf to disable
errTune = opt.errTune;          % disabled autoTune when error_function > errTune 

%Assign Avar and xvar mins
bvarMin = opt.bvarMin;
cvarMin = opt.cvarMin;

%Warn user about diagnostics
if diagnostics
    warning('Using diagnostics will slow performance')
end

%Determine requested outputs
saveEM = opt.saveEM;
saveState = opt.saveState;
saveHist = (nargout >= 3);

%Get problem dimensions
Nb = problem.Nb;
Nc = problem.Nc;

%Ensure that the problem setup is consistent by calling problem check
%method.
problem.check();


%Preallocate storage for estHist if user requires it
if (saveHist)
    estHist.errZ = zeros(nit,1);
    estHist.errB = zeros(nit,1);
    estHist.errC = zeros(nit,1);
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
        estHist.bvarMin = zeros(nit,1);
        estHist.bvarMax = zeros(nit,1);
        estHist.bvarMean = zeros(nit,1);
        estHist.cvarMin = zeros(nit,1);
        estHist.cvarMax = zeros(nit,1);
        estHist.cvarMean = zeros(nit,1);
        estHist.svarMin = zeros(nit,1);
        estHist.svarMax = zeros(nit,1);
        estHist.svarMean = zeros(nit,1);
        estHist.qvarMin = zeros(nit,1);
        estHist.qvarMax = zeros(nit,1);
        estHist.qvarMean = zeros(nit,1);
        estHist.rvarMin = zeros(nit,1);
        estHist.rvarMax = zeros(nit,1);
        estHist.rvarMean = zeros(nit,1);
        estHist.normB = zeros(nit,1);
        estHist.normC = zeros(nit,1);
        estHist.normZ = zeros(nit,1);
    end
end

%Determine whether estimators are tunable and, if so, temporarily disable 
if any(strcmp('disableTune', properties(gOut)))
  tuneableZ = true; % is tunable
  disableTuneZ = gOut.disableTune;  % remember state
  gOut.disableTune = true; % temporarily disable
else
  tuneableZ = false; % is not tunable
end
if any(strcmp('disableTune', properties(gB)))
  tuneableB = true; % is tunable
  disableTuneB = gB.disableTune; % remember state
  gB.disableTune = true; % temporarily disable
else
  tuneableB = false; % is not tunable
end
if any(strcmp('disableTune', properties(gC)))
  tuneableC = true; % is tunable
  disableTuneC = gC.disableTune; % remember state
  gC.disableTune = true; % temporarily disable
else
  tuneableC = false; % is not tunable
end

%Handle concatenated input estimators
%Handle uniform variance
%if uniformVariance
    if isa(gB,'EstimInConcat')
        %Determine how many distributions are included
        distCountB = length(gB.estimArray);

        % Store change indices for possible use in ParametricZ
        opt.gBind = gB.ind;
    else
        distCountB = 1;

        opt.gBind = [1;Nb+1];
    end

    if isa(gC,'EstimInConcat')
        %Determine how many distributions are included
        distCountC = length(gC.estimArray);

        % Store change indices for possible use in ParametricZ
        opt.gCind = gC.ind;
    else
        distCountC = 1;

        opt.gCind = [1;Nc+1];
    end
%end

%% Initialization

if isempty(opt.state) %if no state is provided
    
    %Initialize Avar
    [bhat,bvar] = gB.estimInit();
    
    %Initialize Xvar
    [chat,cvar] = gC.estimInit();
    
    %Handle case of scalar input distribution 
    if (length(bhat) == 1)
        bhat = gB.genRand([Nb,1]);
        bvar = repmat(bvar,Nb,1);
    end
    if (length(chat) == 1)
        chat = gC.genRand([Nc,1]);
        cvar = repmat(cvar,Nc,1);
    end
    
    %Initialize valIn
    valIn = 0;
    
    %Replace these defaults with values in the options object, if they exist
    if ~isempty(opt.bhat0)
        bhat = opt.bhat0;
        %If warm starting, set valIn to be a negative inf to avoid problems
        valIn = -inf;
    end
    if ~isempty(opt.bvar0)
        bvar = opt.bvar0;
    end
    if ~isempty(opt.chat0)
        chat = opt.chat0;
    end
    if ~isempty(opt.cvar0)
        cvar = opt.cvar0;
    end
    
    %Placeholder initializations- values are not used
    bhatBar = 0;
    chatBar = 0;
    shat = 0;
    svar = 0;
    pvarOpt = 0;
    zvarOpt = 0;
    
    %Scalar variance
    if uniformVariance
        
        %Handle multiple concatenated estimators for b
        if distCountB>1
            
            %Check if bvar is already the right size
            if numel(bvar) == distCountB
                bvar = bvar(:);
                %Check if it is the size of the total B vector
            elseif numel(bvar) == Nb
                %If so, average over the relevant groups
                bvarOld = bvar(:);
                bvar = zeros(distCountB,1);
                for kk = 1:length(bvar)
                    I = (gB.ind(kk):gB.ind(kk+1)-1);
                    bvar(kk) = mean(bvarOld(I));
                end
            else %Otherwise, multiply across the whole length
                bvar = ones(distCountB,1)*mean(bvar(:));
            end

        else
            %just average 
            bvar = mean(bvar(:));
        end
        
        %Handle concatenated estimators for c
        if distCountC>1
            
            %Check if cvar is already the right size
            if numel(cvar) == distCountC
                cvar = cvar(:);
                %Check if it is the size of the total C vector
            elseif numel(cvar) == Nc
                %If so, average over the relevant groups
                cvarOld = cvar(:);
                cvar = zeros(distCountC,1);
                for kk = 1:length(cvar)
                    I = (gC.ind(kk):gC.ind(kk+1)-1);
                    cvar(kk) = mean(cvarOld(I));
                end
            else %Otherwise, multiply across the whole length
                cvar = ones(distCountC,1)*mean(cvar(:));
            end

        else
            %just average 
            cvar = mean(cvar(:));
        end
    
    else % not uniformVariance

        %Check to make sure that variances are not scalars!
        if (length(bvar) == 1)
           bvar = repmat(bvar,Nb,1);
        end

        if (length(cvar) == 1)
           cvar = repmat(cvar,Nc,1);
        end

    end
    
    %Address warm starting of shat0
    if ~isempty(opt.shat0)
        shat = opt.shat0;
    end
    
    %Init valOpt empty
    valOpt = [];
    
else %Use the provided state information
    
    %B variables
    bhat = opt.state.bhat;
    bvar = opt.state.bvar;
    bhatBar = opt.state.bhatBar;
    bhatOpt = opt.state.bhatOpt;
    bhatBarOpt = opt.state.bhatBarOpt;
    
    %C Variables
    chat = opt.state.chat;
    cvar = opt.state.cvar;
    chatBar = opt.state.chatBar;
    chatOpt = opt.state.chatOpt;
    chatBarOpt = opt.state.chatBarOpt;
    
    %S variables
    shat = opt.state.shat;
    svar = opt.state.svar;
    shatOpt = opt.state.shatOpt;
    svarOpt = opt.state.svarOpt;
    shatNew = opt.state.shatNew;
    svarNew = opt.state.svarNew;
    
    %Cost stuff
    valIn = opt.state.valIn;
    
    %Variance momentum terms
    pvarOpt = opt.state.pvarOpt;
    zvarOpt = opt.state.zvarOpt;
    
    %Step
    step = opt.state.step;
    
    %Old cost values
    valOpt = opt.state.valOpt;
    
end

%Specify minimum variances
pvarMin = opt.pvarMin;

%Placeholder initializations
rhat = 0;
rvar = 0;
qhat = 0;
qvar = 0;

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
if isempty(opt.state)
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
    
    %Compute new P values (zhat = A(bhat)*X(chat) and zvar is pvarBar in
    %the P-BiG-AMP derivation. Using these names for consistency with the
    %BiG-AMP code.
    [zhat,zvar,pvar] = problem.zObject.pComputation(opt,bhat,bvar,chat,cvar);
    
    %Include pvar step
    if pvarStep
        pvar = step1*pvar + (1-step1)*pvarOpt;
        zvar = step1*zvar + (1-step1)*zvarOpt;
    end
    
    % Compute log likelihood at the output and add it the total negative
    % K-L distance at the input.
    if (compVal)
        valOut = sum(sum(gOut.logLike(zhat,pvar)));
        val(it) = valOut + valIn;
    end
   
    % Compute the errors
    errZ = opt.error_function(zhat);
    errB = opt.error_functionB(bhat);
    errC = opt.error_functionC(chat);

    % An iteration "passes" if any of below is true:
    % 1. Adaptive stepsizing is turned off
    % 2. Current stepsize is so small it can't be reduced
    % 3. The current utility at least as large as the worst in the stepWindow
    stopInd = length(valOpt);
    startInd = max(1,stopInd - stepWindow);
    valMin = min(valOpt(startInd:stopInd));
    pass = (~adaptStep) || (step <= stepMin) || isempty(valMin) || (val(it) >= valMin);
    
    %Save the step size and pass result if history requested
    if saveHist
        estHist.step(it) = step;
        estHist.pass(it) = pass;
    end
    
    % If pass, set the optimal values and compute a new target shat and
    % snew.
    if (pass)
        
        %Slightly inrease step size after pass if using adaptive steps
        step = stepIncr*step;
        
        % Set new optimal values
        shatOpt = shat;
        svarOpt = svar;
        chatBarOpt = chatBar;
        chatOpt = chat;
        bhatBarOpt = bhatBar;
        bhatOpt = bhat;
        pvarOpt = pvar;
        zvarOpt = zvar;
        
        %Bound pvar
        pvar = max(pvar, pvarMin);
        
        %We keep a record of only the succesful step valOpt values
        valOpt = [valOpt val(it)]; %#ok<AGROW>
        
        %Continued output step
        phat = zhat - shat.*zvar;
        
        %Restore original disableTune settings (nominally =false)
        if errZ <= errTune
            if tuneableZ, gOut.disableTune = disableTuneZ; end;
            if tuneableB, gB.disableTune = disableTuneB; end;
            if tuneableC, gC.disableTune = disableTuneC; end;
        end

        % Output nonlinear step
        [zhat0,zvar0] = gOut.estim(phat,pvar);
        
        %Compute 1/pvar
        pvarInv = 1 ./ pvar;
        
        %Update the shat quantities
        shatNew = pvarInv.*(zhat0-phat);
        svarNew = pvarInv.*(1-min(zvar0./pvar,zvarToPvarMax));
        
        
        %Scalar Variance
        if uniformVariance
            svarNew = mean(svarNew(:));
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

        %Disable autotuning
        if tuneableZ, gOut.disableTune = true; end;
        if tuneableB, gB.disableTune = true; end;
        if tuneableC, gC.disableTune = true; end;
    end
    
    
    
    % Save results
    if (saveHist)
        
        %Record timing information
        if it > 1
            estHist.timing(it) = estHist.timing(it-1) + toc(tstart);
        else
            estHist.timing(it) = toc(tstart);
        end
        
        %Record errors and value
        estHist.errZ(it) = errZ;
        estHist.errB(it) = errB;
        estHist.errC(it) = errC;
        estHist.val(it) = val(it);
        
        %Save diagnostics if requested
        if diagnostics
            estHist.pvarMin(it) = min(pvar(:));
            estHist.pvarMax(it) = max(pvar(:));
            estHist.pvarMean(it) = mean(pvar(:));
            estHist.zvarMin(it) = min(zvar(:));
            estHist.zvarMax(it) = max(zvar(:));
            estHist.zvarMean(it) = mean(zvar(:));
            estHist.bvarMin(it) = min(bvar(:));
            estHist.bvarMax(it) = max(bvar(:));
            estHist.bvarMean(it) = mean(bvar(:));
            estHist.cvarMin(it) = min(cvar(:));
            estHist.cvarMax(it) = max(cvar(:));
            estHist.cvarMean(it) = mean(cvar(:));
            estHist.svarMin(it) = min(svar(:));
            estHist.svarMax(it) = max(svar(:));
            estHist.svarMean(it) = mean(svar(:));
            estHist.qvarMin(it) = min(qvar(:));
            estHist.qvarMax(it) = max(qvar(:));
            estHist.qvarMean(it) = mean(qvar(:));
            estHist.rvarMin(it) = min(rvar(:));
            estHist.rvarMax(it) = max(rvar(:));
            estHist.rvarMean(it) = mean(rvar(:));
            estHist.normB(it) = norm(bhat,'fro');
            estHist.normC(it) = norm(chat,'fro');
            estHist.normZ(it) = norm(zhat,'fro');
            
            
        end
    end
    
    
    % Check for convergence if step was succesful
    if pass
        if any(isnan(zhat(:))) || any(isinf(zhat(:))) || norm(zhat(:))<normTol
            stop = true;
        else
            testVal = norm(zhat(:) - zhatOpt(:)) / norm(zhat(:));
            if (it > 1) && (testVal < tol)
                stop = true;
            end
        end
        
        %Set other optimal values- not actually used by iterations
        bvarOpt = bvar;
        cvarOpt = cvar;
        zhatOpt = zhat;
        
        %Save EM variables if requested
        if saveEM
            rhatFinal = rhat;
            rvarFinal = rvar;
            qhatFinal = qhat;
            qvarFinal = qvar;
            zvarFinal = zvar0; % should probably be called "zvar0Final"
            %zhat0Final = zhat0;
            %zvar0Final = zvar0;
            pvarFinal = pvar;
        end
    end
    
    % Print results
    if (verbose)
        fprintf(1,...
                'it=%3d errZ=%6.2e normZ=%6.2e val=%6.2e step=%5.2e\n',...
                it, errZ, norm(zhat), testVal, step1);
    end
    
    %Start timing next iteration
    tstart = tic;
    
    % Create new candidate shat
    if it > 1 || ~isempty(opt.state)
        step1 = step;
        if stepFilter >= 1
            step1 = step1*it/(it+stepFilter);
        end
    end
    shat = (1-step1)*shatOpt + step1*shatNew;
    svar = (1-step1)*svarOpt + step1*svarNew;
    chatBar = (1-step1)*chatBarOpt + step1*chatOpt;
    bhatBar = (1-step1)*bhatBarOpt + step1*bhatOpt;
    
    %Compute R and Q updates
    [rhat,rvar,qhat,qvar] = problem.zObject.rqComputation(...
        opt,bhatBar,bvar,chatBar,cvar,shat,svar);
    
    %Check limits. zObject is supposed to enforce upper bounds on the
    %variance as specified in the options object opt
    if any(rvar > opt.varThresh)
        warning('problem object did not implement rvar limit')
    end
    if any(qvar > opt.varThresh)
        warning('problem object did not implement qvar limit')
    end
    
    %Enforce limits
    rvar = max(rvar,cvarMin);
    qvar = max(qvar,bvarMin);
    
    % Input nonlinear step
    if compVal
        [chat,cvar,valInC] = gC.estim(rhat, rvar);
        [bhat,bvar,valInB] = gB.estim(qhat, qvar);
    else %method may avoid computation if the vals are not needed
        [chat,cvar] = gC.estim(rhat, rvar);
        [bhat,bvar] = gB.estim(qhat, qvar);
    end
    
    %Scalar variances
    if uniformVariance
        
        %Handle B
        if distCountB == 1
            bvar = mean(bvar(:));
        else
            %Check if bvar is already the right size
            if numel(bvar) == distCountB
                bvar = bvar(:);
                %Check if it is the size of the total B vector
            elseif numel(bvar) == Nb
                %If so, average over the relevant groups
                bvarOld = bvar(:);
                bvar = zeros(distCountB,1);
                for kk = 1:length(bvar)
                    I = (gB.ind(kk):gB.ind(kk+1)-1);
                    bvar(kk) = mean(bvarOld(I));
                end
            else %Otherwise, multiply across the whole length
                bvar = ones(distCountB,1)*mean(bvar(:));
            end
        end
        
        %Handle C
        if distCountC == 1
            cvar = mean(cvar(:));
        else
            %Check if bvar is already the right size
            if numel(cvar) == distCountC
                cvar = cvar(:);
                %Check if it is the size of the total B vector
            elseif numel(cvar) == Nc
                %If so, average over the relevant groups
                cvarOld = cvar(:);
                cvar = zeros(distCountC,1);
                for kk = 1:length(cvar)
                    I = (gC.ind(kk):gC.ind(kk+1)-1);
                    cvar(kk) = mean(cvarOld(I));
                end
            else %Otherwise, multiply across the whole length
                cvar = ones(distCountC,1)*mean(cvar(:));
            end
        end
        
        
    end
    
    %Update valIn
    if compVal
        valIn = sum( valInC(:) ) + sum ( valInB(:) );
    end
    
    %Don't stop before minimum iteration count
    if it < nitMin
        stop = false;
    end
    
end

%% Save the final values

%Save the options object that was used
optFin = opt;

%Estimates of the two signal vectors and their bilinear output
estFin.chat = chatOpt;
estFin.cvar = cvarOpt;
estFin.bhat = bhatOpt;
estFin.bvar = bvarOpt;
estFin.zhat = zhatOpt;
estFin.nit = it;

%Save values useful for EM learning
if saveEM
    estFin.rhat = rhatFinal;
    estFin.rvar = rvarFinal;
    estFin.qhat = qhatFinal;
    estFin.qvar = qvarFinal;
    estFin.zvar = zvarFinal; % should probably be called "zvar0, zvar0Final"
    %estFin.zhat0 = zhat0Final;
    %estFin.zvar0 = zvar0Final;
    estFin.phat = phat;
    estFin.pvar = pvarFinal;
end

%% Cleanup estHist

%Trim the outputs if early termination occurred
if saveHist && (it < nit)
    estHist.errZ = estHist.errZ(1:it);
    estHist.errB = estHist.errB(1:it);
    estHist.errC = estHist.errC(1:it);
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
        estHist.bvarMin = estHist.bvarMin(1:it);
        estHist.bvarMax = estHist.bvarMax(1:it);
        estHist.bvarMean = estHist.bvarMean(1:it);
        estHist.cvarMin = estHist.cvarMin(1:it);
        estHist.cvarMax = estHist.cvarMax(1:it);
        estHist.cvarMean = estHist.cvarMean(1:it);
        estHist.svarMin = estHist.svarMin(1:it);
        estHist.svarMax = estHist.svarMax(1:it);
        estHist.svarMean = estHist.svarMean(1:it);
        estHist.qvarMin = estHist.qvarMin(1:it);
        estHist.qvarMax = estHist.qvarMax(1:it);
        estHist.qvarMean = estHist.qvarMean(1:it);
        estHist.rvarMin = estHist.rvarMin(1:it);
        estHist.rvarMax = estHist.rvarMax(1:it);
        estHist.rvarMean = estHist.rvarMean(1:it);
        estHist.normB = estHist.normB(1:it);
        estHist.normC = estHist.normC(1:it);
        estHist.normZ = estHist.normZ(1:it);
    end
end


%% Save the state

if saveState
    
    %B variables
    estFin.state.bhat = bhat;
    estFin.state.bvar = bvar;
    estFin.state.bhatBar = bhatBar;
    estFin.state.bhatOpt = bhatOpt;
    estFin.state.bhatBarOpt = bhatBarOpt;
    
    %C Variables
    estFin.state.chat = chat;
    estFin.state.cvar = cvar;
    estFin.state.chatBar = chatBar;
    estFin.state.chatBarOpt = chatBarOpt;
    estFin.state.chatOpt = chatOpt;
    
    %s variables
    estFin.state.shat = shat;
    estFin.state.svar = svar;
    estFin.state.shatOpt = shatOpt;
    estFin.state.shatNew = shatNew;
    estFin.state.svarOpt = svarOpt;
    estFin.state.svarNew = svarNew;
    
    %Cost stuff
    estFin.state.valIn = valIn;
    
    %Variance momentum terms
    estFin.state.pvarOpt = pvarOpt;
    estFin.state.zvarOpt = zvarOpt;
    
    %Step
    estFin.state.step = step;
    
    %Old cost values
    estFin.state.valOpt = valOpt;
    
end



