% Set path
addpath('../main/');

% Select if parameters are set by gamOptGroupLasso
batchRun = true;

% Simulation parameters
ngrp = 100;
grpsize = 4;
nx = ngrp * grpsize;
sparseProb = 0.1;
xvar1 = 1;
xmean1 = 0;
snr = 20;
newdat = 1;
numiter = 20;

% Methods to test
GAMP_METH = 1;      % Generalized approx message passing
LS_METH = 2;        % Least squares
GLASSO_METH = 3;    % Group lasso
GOMP_METH = 4;      % Group OMP
methStr = {'gamp', 'ls', 'glasso', 'gomp'};

% Select parameters if not set by the batch run
if ~batchRun
    nz = 100;
    %methTest = [LS_METH GAMP_METH GLASSO_METH];
    methTest = [GAMP_METH GOMP_METH];
    ntest = 20;
    gam = 0.1;
end
nmeth = length(methTest);

% Create the input estimator function for the case when U=1.
% In this example, this is just a Gaussian
xmean1 = repmat(xmean1, nx,1);
xvar1 = repmat(xvar1, nx,1);
inputEst1 = AwgnEstimIn(xmean1, xvar1);   % Estimator when U=1

% Create the group index vector:
% grpInd(j) = the group index for the component x(j).
% In this simple example, group k consists of the components 
% j \in [(k-1)*grpSize+1:k*grpSize].
grpInd = repmat((1:ngrp), grpsize, 1);
grpInd = grpInd(:);

% Create the group sparsity estimator
p1Nom = repmat(sparseProb,ngrp,1);
inputEst = GrpSparseEstim(inputEst1, p1Nom, grpInd);

% Get mean and variance
[xmean0, xvar0] = inputEst.estimInit;

% Output distribution
wmean = 0;                            % Output mean
wvar = 10.^(-0.1*snr)*mean(xvar0);    % Output variance

% G-AMP parameters
opt = GampOpt();
opt.step = 1;           % step size
opt.nit = numiter;      % number of iterations
opt.removeMean = true;  % remove mean
opt.pvarMin = 0.0001;
opt.xvarMin = 0.0001;
opt.adaptStep = true;
opt.verbose = false;

% Lasso options
lassoOpt.nit = 25;  % number of iterations
lassoOpt.prt = false; % verbose

% Initialize vectors
mseMeth = zeros(ntest, nmeth);
mseGAMP = zeros(numiter, ntest);
val = zeros(numiter, ntest);

for itest = 1:ntest
    
    if (newdat)
        % Generate random input vector
        x = inputEst.genRand(nx);
        
        % Generate random matrix and transform output z        
        A = 1/sqrt(nx).*(randn(nz,nx));
        Aop = MatrixLinTrans(A);
        z = A*x;
        
        % Generate output
        w = normrnd(wmean, sqrt(wvar), nz, 1);
        y = z + w;
        outputEst = AwgnEstimOut(y, wvar);
    end
    
    % Loop over methods
    for imeth = 1:nmeth
        
        meth = methTest(imeth);
        if (meth == GAMP_METH)
            
            % Call the G-AMP algorithm
            % The function returns xhatTot(i,t) = estimate of x(i) on iteration t.
            % So, xhatTot(:,end) is the final estimate
            [xhatGAMP,rhat,rvar,estHist] = gampEst(inputEst, outputEst, Aop, opt);
            xhatTot = estHist.xhat;
            xhat = xhatGAMP;
            
        elseif (meth == LS_METH)
            
            % Compute LS solution
            Dx = repmat(xvar0,1,nx);
            Dx1 = repmat(xvar0,1,nz);
            xhat = (wvar*eye(nx) + Dx.*(A'*A)) \ ((Dx1.* A')*(y-A*xmean0)) ...
                + xmean0;
            
        elseif (meth == GLASSO_METH)
            xhat = grpLasso(y,A,gam,grpInd,lassoOpt);
            
        elseif (meth == GOMP_METH)
            
            % Find number of groups that are non-zero
            k = 0;
            for igrp = 1:ngrp
                if (sum(abs(x).*(grpInd == igrp) > 1e-6))
                    k = k+1;
                end
            end
            xhat = grpOmp(y,A,grpInd,k);
        end

        % Measure MSE
        mseMeth(itest,imeth) = 10*log10( mean((xhat-x).^2) );
        if (meth == GAMP_METH)
            dx = xhatTot - repmat(x,1,opt.nit);
            mseGAMP(:,itest) = 10*log10( mean( abs(dx).^2 )' );
        end
        fprintf(1,'it=%d %s mse=%f\n', itest, methStr{meth}, mseMeth(itest,imeth));
        
        if (meth == GAMP_METH) && (mseMeth(itest,imeth) > -15)
            %return;
        end
        
        
        
    end
    if ~newdat
        return
    end
    
end

% Compute mean error by iteration
for imeth = 1:nmeth
    if (methTest(imeth) == GAMP_METH)
        mseGAMPmean = mean( mseGAMP, 2 );
        plot((1:numiter), mseGAMPmean, '-');
    else
        mseMean = mean( mseMeth(:,imeth) );
        plot([1 numiter], mseMean*[1 1], '-');
    end
    hold on;
end
hold off;
grid on;
xlabel('Iteration');
ylabel('MSE (dB)');


