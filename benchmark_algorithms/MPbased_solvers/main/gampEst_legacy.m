function [xhatFinal, xvarFinal, rhatFinal, rvarFinal,...
    shatFinal, svarFinal, zhatFinal,zvarFinal,estHist] = ...
    gampEst(scaEstIn, scaEstOut, A, opt)
% gampEst:  G-AMP estimation algorithm
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
adaptStep = opt.adaptStep;      % adaptive step size
stepWindow = opt.stepWindow;    % step size check window size
bbStep = opt.bbStep;            % Barzilai Borwein step size
verbose = opt.verbose;          % Print results in each iteration
tol = opt.tol;                  % Convergence tolerance
stepTol = opt.stepTol;          % minimum allowed step size
compVal = true;
saveHist = (nargout >= 9);

% If A is a double matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

% Get dimensions
[m,n] = A.size();
S = scaEstOut.numColumns; %check for matrix valued signal

% Get initial mean and variance
% For a scalar output, the same distribution is applied to
% all components
[xhat0,xvar0] = scaEstIn.estimInit();
if (length(xhat0) == 1)
    xhat0 = repmat(xhat0,n,S);
    xvar0 = repmat(xvar0,n,S);
end
valIn = 0;

%Replace these defaults with the warm start values if provided in the
%options object
if ~isempty(opt.xhat0)
    xhat0 = opt.xhat0;
    %If warm starting, set valIn to be a negative inf to avoid problems
    valIn = -inf;
end
if ~isempty(opt.xvar0)
    xvar0 = opt.xvar0;
end

% If the mean removal option is set, create an augmented linear
% transform class with the mean removed.  See the LinTransDemean
% class for more details
if (opt.removeMean)
    Ad = LinTransDemean(A,xhat0);
    xhat = zeros(n+1,1);
    xvar0Mean = mean(xvar0);
    xvar = [xvar0; xvar0Mean];
else
    Ad = A;
    xhat = xhat0;
    xvar = xvar0;
end
[m1,n1] = Ad.size();

rhat = xhat;
rvar = 100*xvar;
xhatPrev = zeros(n1,S);
xhatFinal  = zeros(n1,S);

% Initialization of the algorithm
shat = zeros(m1,S);
svar = zeros(m1,S);
val = zeros(nit,1);
valOpt = [];
if (saveHist)
    estHist.xhat = zeros(n*S,nit);
    estHist.xvar = zeros(n*S,nit);
    estHist.rhat = zeros(n*S,nit);
    estHist.rvar = zeros(n*S,nit);
    estHist.phat = zeros(m,nit);
    estHist.pvar = zeros(m,nit);
    estHist.val = zeros(nit,1);
    estHist.step = zeros(nit,1);
    estHist.pass = false(nit,1);
end

% Check for the presence of two methods within the LinTrans and EstimIn
% objects and set flags accordingly
MtxUncertaintyFlag = ismethod(Ad,'includeMatrixUncertainty');
MsgUpdateFlag = ismethod(scaEstIn, 'msgUpdate');

% Compute min pvar
pvarMin = opt.pvarMin*(A.multSq(xvar0)); %Asq(1:m,1:n)*xvar0; ---
xvarMin = opt.xvarMin*xvar;

%Protect from zero valued pvarMin
pvarMin(pvarMin == 0) = opt.pvarMin;


%If computing BB steps, compute column norms for use in scaling
if bbStep
    columnNorms = A.multSqTr(ones(m,1)).^0.5;
    columnNorms = repmat(columnNorms,1,S);
end

%Control variable to end the iterations
stop = false;
it = 0;

%Address warm starting of shat0
if ~isempty(opt.shat0)
    shat = opt.shat0;
end

% Main iteration loop
while ~stop
    
    % Iteration count
    it = it + 1;
    
    % Check for final iteration
    if it >= nit
        stop = true;
    end
    
    % Output linear step with no A uncertainty
    pvarBar = Ad.multSq(xvar);
    
    %Incorporate A uncertainty
    if MtxUncertaintyFlag
        pvar = Ad.includeMatrixUncertainty(pvarBar,xhat,xvar);
    else
        pvar = pvarBar;
    end
    
   
    %Update zhat
    zhat = Ad.mult(xhat);
    
    % Compute log likelihood at the output and add it the total negative
    % K-L distance at the input.
    if (compVal)
        valOut = sum(sum(scaEstOut.logLike(zhat(1:m,:),pvar(1:m,:))));
        val(it) = valOut + valIn;
    end
    
    % Determine if candidate passed
    if (it > 1)
        
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
        
        % Set new optimal values
        shatOpt = shat;
        svarOpt = svar;
        xhatPrevOpt = xhatPrev;
        xhatOpt = xhat;
        
        %Set zhat outputs
        zhatFinal = zhat;
        zvarFinal = pvarBar;
        
        %We keep a record of only the succesful step valOpt values
        valOpt = [valOpt val(it)]; %#ok<AGROW>
        
        %Store previous optimal solution
        xhatPrevFinal = xhatFinal;
        
        % Save current optimal solution
        if (opt.removeMean)
            [xhatFinal, rhatFinal, rvarFinal] = Ad.getEst(xhat,rhat,rvar);
            xvarFinal = xvar(1:n);
        else
            xhatFinal = xhatOpt;
            xvarFinal = xvar;
            rhatFinal = rhat;
            rvarFinal = rvar;
            
        end
        
        %Set shat and svar outputs
        shatFinal = shatOpt;
        svarFinal = svarOpt;
        
        % Check for convergence
        if (it > 1) && ...
                (norm(xhatPrevFinal - xhatFinal) / norm(xhatFinal) < tol)
            stop = true;
        end
        
        % Continued output step
        phat = zhat - pvarBar.*shat; %uses pvarBar rather than pvar
        pvar(1:m,:) = max(pvar(1:m,:), pvarMin);
        
        % Output nonlinear step -- regular components
        shatNew = zeros(m1,S);
        svarNew = zeros(m1,S);
        I = (1:m)';
        [zhat0,zvar0] = scaEstOut.estim(phat(I,:),pvar(I,:));
        shatNew(I,:) = (1./pvar(I,:)).*(zhat0-phat(I,:));
        svarNew(I,:) = (1./pvar(I,:)).*(1-zvar0./pvar(I,:));
        
        % Output nonlinear step for the demean output component
        if (opt.removeMean)
            shatNew(m+1) = (-phat(m+1))/(pvar(m+1));
            svarNew(m+1) = 1/pvar(m+1);
        end
        
        %Compute new BB Step size if requested
        if bbStep && it > 2
            
            %Compute previous step direction/size weighted with column
            %norms
            sBB = (xhatOpt(1:n,:) - xhatPrevOpt(1:n,:));
            
            %Compute new step size using columnNorms weighting
            %Select the smallest step over all the columns for a matrix
            %valud signal
            values = sum(abs(sBB .* columnNorms).^2,1) ./...
                sum(abs(A.mult(sBB).^2),1);
            step = min(values);

            
            
        end
        
        %Enforce step size bounds
        step = min([max([step stepMin]) stepMax]);
        
    else
        % Decrease step size
        step = max(stepMin, 0.5*step);
        
        %Check for minimum step size
        if step < stepTol
            stop = true;
        end
    end
    
    % Save results
    if (saveHist)
        estHist.xhat(:,it) = xhatFinal(:);
        estHist.xvar(:,it) = xvarFinal(:);
        estHist.rhat(:,it) = rhatFinal(:);
        estHist.rvar(:,it) = rvarFinal(:);
        estHist.phat(:,it) = phat(1:m);
        estHist.pvar(:,it) = pvar(1:m);
        estHist.val(it) = val(it);
    end
    
    % Create new candidate shat
    if (it==1)
        step1 = 1;
    else
        step1 = step;
    end
    shat = (1-step1)*shatOpt + step1*shatNew;
    svar = (1-step1)*svarOpt + step1*svarNew;
    xhatPrev = (1-step1)*xhatPrevOpt + step1*xhatOpt;
    
    % Print results
    if (verbose)
        fprintf(1,'it=%3d value=%12.4e step=%f\n', it, val(it), step);
    end
    
    % Input linear step
    rvar = 1./Ad.multSqTr(svar);               % rvar = 1./((A.^2)*svar)
    rhat = xhatPrev + rvar.*(Ad.multTr(shat)); % rhat = xhatPrev + rvar.*(A*shat)
    rvar = max(rvar, xvarMin);
    
    % Send messages to input estimation function.
    if MsgUpdateFlag
        valMsg = scaEstIn.msgUpdate(it, rhat, rvar);
    else
        valMsg = 0;
    end
    
    % Input nonlinear step
    if (opt.removeMean)
        
        % Regular components
        I = (1:n)';
        [xhat(I),xvar(I),valIn] = scaEstIn.estim(rhat(I)+xhat0, rvar(I));
        xhat(I) = xhat(I) - xhat0;
        
        % Mean component
        xhat(n+1) = xvar0Mean/(xvar0Mean+rvar(n+1))*rhat(n+1);
        xvar(n+1) = xvar0Mean*rvar(n+1)/(xvar0Mean+rvar(n+1));
        
    else
        % Call input scalar estimator
        [xhat,xvar,valIn] = scaEstIn.estim(rhat, rvar);
        
    end
    valIn = sum( valIn(:) ) + valMsg;
    
end

%Trim the outputs if early termination occurred
if saveHist && (it < nit)
    estHist.xhat = estHist.xhat(:,1:it);
    estHist.rhat = estHist.rhat(:,1:it);
    estHist.rvar = estHist.rvar(:,1:it);
    estHist.val = estHist.val(1:it);
    estHist.step = estHist.step(1:it);
    estHist.pass = estHist.pass(1:it);
end

