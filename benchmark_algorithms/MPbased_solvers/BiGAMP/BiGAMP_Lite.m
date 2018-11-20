function [estFin,optFin,estHist] = ...
    BiGAMP_Lite(Y,nux,nuA,nuw,problem,opt)
% BiGAMP_Lite: Implements a simplified version of BiG-AMP. This code
% assumes AWGN output channel, i.i.d. zero-mean Gaussian input channels for
% A and X, and scalar variances. The resulting simplications offer
% significant computational cost savings at the expense of generality. 
%
%
% INPUTS:
% -------
% Y: the noisy data matrix. May be passed as a full matrix, a sparse matrix
%   as in sparse(Y), or as a vector containing the observed entries. The
%   locations of these entries are defined in the rowLocations and
%   columnLocations fields of the options structure. 
% nux, nuA, nuw: Prior variances on X and A entries. nuw is the AWGN noise 
%   variance
% problem: An objet of the class BiGAMPProblem specifying the problem
%   setup, including the matrix dimensions and observation locations
% opt (optional):  A set of options of the class BiGAMPOpt. 
%
% OUTPUTS:
% --------
% estFin: Structure containing final BiG-AMP outputs
% optFin: The BiGAMPOpt object used
% estHist: Structure containing per iteration metrics about the run



%% Setup

% Get options
if (nargin < 6)
    opt = BiGAMPOpt();
elseif (isempty(opt))
    opt = BiGAMPOpt();
end
nit     = opt.nit;              % number of iterations
nitMin  = opt.nitMin;           % minimum number of iterations
step    = min(opt.step,opt.stepMax);         % step size
verbose = opt.verbose;                       % Print results in each iteration
tol = opt.tol;                               % Convergence tolerance
stepMax = opt.stepMax;
stepMin = opt.stepMin;
stepWindow = opt.stepWindow;    % step size check window size
stepIncr = opt.stepIncr;
stepDecr = opt.stepDecr;
stepTol = opt.stepTol;
stepFilter = opt.stepFilter;    % step filter setting, <1 for no effect
adaptStep = opt.adaptStep;      % adaptive step size
gainMode = opt.gainMode;

%Save checks
saveEM = opt.saveEM;
saveHist = nargout >= 2;


%Get problem dimensions
M = problem.M;
L = problem.L;
N = problem.N;

%Check for partial observation of Z
rLoc = problem.rowLocations;
cLoc = problem.columnLocations;

%Verify that locations are provided
if isempty(rLoc) || isempty(cLoc) || length(rLoc) ~= length(cLoc)
    error('Column and row locations of observed entries must be provided')
end

%Indices of observed entries
omega = sub2ind([M L],rLoc,cLoc);

%Determine sampling rate
p1 = length(rLoc) / (M*L);

%Check for sparse mode
sparseMode = opt.sparseMode;

%If in sparse mode, ensure that sparseMult2 has been mexed
if sparseMode

    %Check if it is here
    if isempty(which('sparseMult2'))
        %Switch to this directory
        curLoc = pwd;
        cd(fileparts(which(mfilename)));
        try
            %Try to mex the function
            mex sparseMult2.c -largeArrayDims
        catch %#ok<CTCH>
            %Something went wrong
            cd(curLoc);
            error('Unable to mex sparseMult2- must run with sparseMode=false')
            
        end
        %Return to working directory
        cd(curLoc);
    end
    
end

%Create sparse matrix multiplication operator. This operator multiplies two
%full matrices of sizes MxN and NxL and returns a vector of the observed
%entries in the MxL product specified by rLoc and cLoc.
if sparseMode
    totalDataPoints = length(rLoc);
    sMult = @(arg1,arg2) sparseMult2(arg1.',arg2,rLoc,cLoc);
    
    %Ensure that Y contains only the observed entries
    if numel(Y) > totalDataPoints
        Y = Y(omega).'; %assume it is a full matrix
    end
    %Check if a full Z is computed for diagnostic purposes
    if saveHist
        zErrorFlag = nargin(opt.error_function) > 0;
    end
end

%Create mask matrix if not in sparse mode
if ~sparseMode
    maskMatrix = zeros(M,L);
    maskMatrix(omega) = 1;
end

%Initialize A
if ~isempty(opt.Ahat0)
    Ahat = opt.Ahat0;
else
    Ahat = sqrt(nuA)*randn(M,N);
end
if isempty(opt.Avar0)
    Avar = nuA;
else
    Avar = mean(opt.Avar0(:));
end

%Initialize X
if ~isempty(opt.xhat0)
    xhat = opt.xhat0;
else
    xhat = sqrt(nux)*randn(N,L);
end
if isempty(opt.xvar0)
    xvar = nux;
else
    xvar = mean(opt.xvar0(:));
end

%Preallocate storage for estHist if user requires it
if (saveHist)
    estHist.errZ = zeros(nit,1);
    estHist.timing = zeros(nit,1);
    estHist.step = zeros(nit,1);
    estHist.pass = zeros(nit,1);
end

%Compute variance limits
pvarMin = opt.pvarMin;

%Control variable to end the iterations
stop = false;
it = 0;


%% Iterations

%Init pvar
pvarOpt = pvarMin;

%optimal values
valOpt = [];

%Start timing first iteration
tstart = tic;

%Initial step
step1 = 1;

%Cost values
val = zeros(nit,1);
testVal = inf;

%Initialize valIn
valIn = 0;

%Placeholder inits
Vhat = 0;
xBar = 0;
ABar = 0;
zhatOpt = 0;

%Use Vhat
if ~isempty(opt.shat0)
    Vhat = opt.shat0;
end

% Main iteration loop
while ~stop
    
    % Iteration count
    it = it + 1;
    
    %Set step1 to step on second iteration to avoid transients
    if it == 2
        step1 = step;
    end
    
    % Check for final iteration
    if it >= nit
        stop = true;
    end
    
    %% Initial computations
    
    %Update zhat
    if ~sparseMode
        zhatFull = Ahat * xhat;
        zhat = maskMatrix .* zhatFull;
    else
        zhat = sMult(Ahat,xhat);
    end
    
    %Compute new V
    holder = (Y - zhat);
    
    %Compute Frobenius norms
    Xf2 = norm(xhat,'fro')^2;
    Af2 = norm(Ahat,'fro')^2;
    
    %Compute pvar
    pvar = step1*(Avar*Xf2 / L + xvar*Af2/M + N*Avar*xvar) +...
        (1-step1)*pvarOpt;
    pvar = max(pvar, pvarMin);
    
    %Set pvarOpt
    if it == 1
        pvarOpt = pvar;
    end
    
    %Compute cost function
    if adaptStep
        valY = (abs(holder).^2 + pvar);
        if ~sparseMode
            valY = valY .* maskMatrix;
        end
        valY = -0.5*sum(sum(valY));
        
        if nuw > 0
            %Scale with noise variance
            valY = valY / nuw;
            
            val(it) = valY + valIn;
        else
            val(it) = valY;
        end
        
    end
    
    %% Test for step acceptance
    
    if it > 1
        
        %Check against worst value in last stepWindow good steps
        stopInd = length(valOpt);
        startInd = max(1,stopInd - stepWindow);
        
        pass = ~adaptStep ||...
            (val(it) > min(valOpt(startInd:stopInd))) ||...
            (step <= stepMin);
    else
        pass = true;
    end
    
    %Save the step size and pass result if history requested
    if saveHist
        estHist.step(it) = step;
        estHist.pass(it) = pass;
        estHist.val(it) = val(it);
    end
    
    % If pass, set the optimal values
    if (pass)
        
        %Compute V gain
        Vgain = (Avar*Xf2 / L + xvar*Af2/M ) ...
            / (pvarOpt + nuw);
        
        %Slightly inrease step size after pass if using adaptive steps
        if adaptStep
            step = stepIncr*step;
        end
        
        %Enforce step size bounds
        step = min([max([step stepMin]) stepMax]);
        
        %Save Optimal values
        valOpt = [valOpt val(it)]; %#ok<AGROW>
        xhatOpt = xhat;
        AhatOpt = Ahat;
        xvarOpt = xvar;
        AvarOpt = Avar;
        VhatOpt = Vhat;
        pvarOpt = pvar;
        xBarOpt = xBar;
        ABarOpt = ABar;
        holderOpt = holder;
    else
        % Decrease step size
        step = max(stepMin, stepDecr*step);
        
        %Check for minimum step size
        if step < stepTol
            stop = true;
        end
    end
    
    %% Timing and storage
    
    %Save timing information
    if saveHist
        if it > 1
            estHist.timing(it) = estHist.timing(it-1) + toc(tstart);
        else
            estHist.timing(it) = toc(tstart);
        end
        
        %Compute the Z error only if needed
        if ~sparseMode
            estHist.errZ(it) = opt.error_function(zhatFull);
        else
            if zErrorFlag
                zhatFull = Ahat*xhat;
                estHist.errZ(it) = opt.error_function(zhatFull);
            end
        end
        
    end
    
    %Check for convergence
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
        
        %Update zhatOpt
        zhatOpt = zhat;
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
    
    
    
    %Check stopping criteria
    if (it > 1) && ...
            (testVal < tol)
        stop = true;
    end
    
    %Check for divergence
    if any(isnan(xhat))
        stop = true;
    end
    
    %Start timing next iteration
    tstart = tic;
    
    %% Update X and A
    
    %Determine step size
    if it > 1
        step1 = step;
        if stepFilter >= 1
            step1 = step1*it/(it+stepFilter);
        end
    end
    
    %Compute new bars
    xBar = step1*xhatOpt + (1-step1)*xBarOpt;
    ABar = step1*AhatOpt + (1-step1)*ABarOpt;
    Vhat = step1*holderOpt + (1 + step1*Vgain - step1)*VhatOpt; %faster
    
    %Update the optimum values for first iteration
    if it == 1
        xBarOpt = xBar;
        ABarOpt = ABar;
        VhatOpt = Vhat;
    end
    
    %Compute norms
    Xbarf2 = norm(xBar,'fro')^2;
    Abarf2 = norm(ABar,'fro')^2;
    
    %Compute Xgain
    Xgain = nux /...
        (nux + N*(nuw + pvarOpt)/Abarf2/p1);
    
    %Compute Again
    Again = nuA /...
        (nuA + N*(nuw+pvarOpt)/Xbarf2/p1);
    
    %Update xhat and Ahat
    if sparseMode
        
        %Need a matrix version of Vhat
        VhatMat = sparse(rLoc,cLoc,Vhat,M,L);
        Vf2 = norm(VhatMat,'fro')^2;
        
        %Compute gains
        switch gainMode
            case 1,
                rGain = 1 - Avar*Vf2*N/Abarf2/(nuw + pvarOpt)/p1/L;
                qGain = 1 - xvar*Vf2*N/Xbarf2/(nuw + pvarOpt)/p1/M;
            case 2,
                rGain = 1 - Avar*M*N/Abarf2;
                qGain = 1 - xvar*N*L/Xbarf2;
            case 3,
                rGain = 1;
                qGain = 1;
        end
        qGain = min(1,max(0,qGain));
        rGain = min(1,max(0,rGain));
        
        % Input nonlinear step for X
        xhat = Xgain * (xBar.*rGain + N/p1/Abarf2*(ABar'*VhatMat));
        
        %Input nonlinear step for A
        Ahat = Again * (ABar.*qGain + N/p1/Xbarf2*(VhatMat*xBar'));
    else
        
        %Compute gains
        Vf2 = norm(Vhat,'fro')^2;
        switch gainMode
            case 1,
                rGain = 1 - Avar*Vf2*N/Abarf2/(nuw + pvarOpt)/p1/L;
                qGain = 1 - xvar*Vf2*N/Xbarf2/(nuw + pvarOpt)/p1/M;
            case 2,
                rGain = 1 - Avar*M*N/Abarf2;
                qGain = 1 - xvar*N*L/Xbarf2;
            case 3,
                rGain = 1;
                qGain = 1;
        end
        qGain = min(1,max(0,qGain));
        rGain = min(1,max(0,rGain));
        
        % Input nonlinear step for X
        xhat = Xgain * (xBar.*rGain + N/p1/Abarf2*(ABar'*Vhat));
        
        %Input nonlinear step for A
        Ahat = Again * (ABar.*qGain + N/p1/Xbarf2*(Vhat*xBar'));
        
    end
    
    %% Update variances
    
    %Compute new xvar
    xvar = nux - nux*Xgain;
    Avar = nuA - nuA*Again;
    
    
    %Update valIn
    if adaptStep && nuw > 0
        valX = 0.5*(log(xvar ./ nux) + (1 - xvar./nux) - xhat.^2/nux);
        valA = 0.5*(log(Avar ./ nuA) + (1 - Avar./nuA) - Ahat.^2/nuA);
        valIn = sum( valX(:) ) + sum ( valA(:) );
    else
        valIn = 0;
    end
    
    %Don't stop before minimum iteration count
    if it < nitMin
        stop = false;
    end
end


%% Cleanup

%Trim the outputs if early termination occurred
if saveHist && (it < nit)
    estHist.errZ = estHist.errZ(1:it);
    estHist.timing = estHist.timing(1:it);
    estHist.step = estHist.step(1:it);
    estHist.pass = estHist.pass(1:it);
end

%Save options
optFin = opt;

%Save result
estFin.xhat = xhatOpt;
estFin.xvar = xvarOpt;
estFin.Ahat = AhatOpt;
estFin.Avar = AvarOpt;

%Provide the EM variables if requested. These values may be slightly
%out-of-sync with the saved values of A and X
%This functionality has not been thoroughly tested with BiG-AMP Lite
if saveEM
    estFin.rhat = xhat / Xgain;
    estFin.rvar = N/p1/Abarf2*(nuw + pvar);
    estFin.qhat = Ahat / Again;
    estFin.qvar = N/p1/Xbarf2*(nuw + pvar);
    estFin.Vhat = Vhat;
    estFin.zvar = nuw*pvar / (pvar + nuw);
end




