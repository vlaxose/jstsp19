% neuralConnSim:  Simulation of neural connectivity
%------------------------------------------------------

% Set path
addpath('../../main/');

% Simulation parameters
% ---------------------
ntest = 10;      % Number of Monte-Carlo tests
numiter = 20;    % Number of iterations
nx = 500;        % Number of input components
nz = 300;        % Number of output components
newdat = 1;
npoly = 4;       % Number of polynomial terms in fit
savedat = 1;
datStr = '';

% Output parameters for single neuron excitation
% see NeuralConnOut
tProb = 0.2;   
outSnr = 20;
inSnr = 100;
outSnrMax = 25;
nexcite = 40;    % number of neurons that are excited per measurements

% Input distribution
GAUSS_IN = 1;         % Gaussian distribution
WEIBUL_IN = 2;        % Weibul distribution
BPSK_IN = 3;          % BPSK distribution
CGAUSS_IN = 4;         % Circular Gaussian distribution
sparseRat = 0.06;      % Sparsity ratio on input
inDist = WEIBUL_IN;    % Input distribution


% CoSAMP k levels to test
%kTest = round([0.5 1 2]*sparseRat*nx)';
%kTest = round(0.5*sparseRat*nx)';
kTest = 0;
nkTest = length(kTest);

% Methods to test
GAMP_METH = 1;
LS_METH = 2;
COSAMP_METH = 3;
methStr = {'gamp', 'ls', 'cosamp'};
methTest = [LS_METH COSAMP_METH*ones(nkTest,1) GAMP_METH];
nmeth = length(methTest);


% GAMP parameters.  See class GampOpt
gampOpt = GampOpt();        % default parameters
gampOpt.step = 0.5;           % step size
gampOpt.nit = 10;      % number of iterations
gampOpt.removeMean = true;  % remove mean
gampOpt.pvarMin = 0.0001;
gampOpt.xvarMin = 0.0001;
gampOpt.adaptStep = true;
gampOpt.verbose = true;
gampOpt.stepMin = 0.1;      % minimum step size
gampOpt.tol = -1;           % do not allow early termination
gampOpt.stepTol = -1;       % do not allow early termination

iterOpt.npoly = 4;  % number of terms in poly approx of nonlinearity
iterOpt.useLS = 1;  % use LS fit for polynomial coefficients
iterOpt.nitTot = 2; % number of iterations
iterOpt.pcoeffInit = []; % Initial polynomial coefficient estiamte
iterOpt.verbose = true;  % Print progress

% Generate input distribution
% ----------------------------
switch (inDist)
    case GAUSS_IN
        xmean0 = 0;
        xvar0 = 1;
        estimWt1 = AwgnEstimIn(xmean0, xvar0);
    case BPSK_IN
        xmax = 1;       % Max value
        x0 = [-xmax xmax]';
        px0 = 0.5*[1 1]';
        estimWt1 = DisScaEstim(x0, px0);
    case WEIBUL_IN
        kx = 1;       % Stretch parameter
        lambdax = 1;    % Shape parameter
        xmax = 4;
        nx0 = 400;
        [x0,px0] = Weibull(kx, lambdax, xmax, nx0);
        estimWt1 = DisScaEstim(x0,px0);
end

% Create sparse distributin for weights
if (sparseRat < 1)
    estimWt = SparseScaEstim(estimWt1, sparseRat, 0);
else
    estimWt = estimWt1;
end
[xmean0,xvar0]= estimWt.estimInit();
           
% Create output function
nxpts = 1000;
xp = estimWt1.genRand(nxpts);
outFn = NeuralConnOut(xp, tProb, inSnr, outSnr, outSnrMax);

% Initialize vectors
xhatTot = zeros(nx,ntest,nmeth);
xTot = zeros(nx,ntest);
mseTot = zeros(ntest,nmeth);
pcoeffTot = zeros(npoly+1,ntest);

for itest = 1:ntest
    
    if (newdat)
        % Generate random input vector
        x = estimWt.genRand(nx);
        
        % Generate random 0-1 stimulation matrix with nexcite elements on
        % per row
        A0 = zeros(nz,nx);
        for iz = 1:nz
            I = randperm(nx);
            A0(iz, I(1:nexcite)) = 1;
        end        
                
        % Generate linear output and count
        z = A0*x;
        cnt = outFn.genRandCnt(z);
        
        % Save true value
        xTot(:,itest) = x;
                
    end
    
    % Loop over methods
    icosamp = 0;
    for imeth = 1:nmeth
        
        meth = methTest(imeth);
        if (meth == COSAMP_METH)            
            A = [A0 ones(nz,1)];
            cosampOpt.tol1 = 1e-3;
            cosampOpt.prt = 0;
            
            % Find k value for this test of cosamp
            icosamp = icosamp + 1;
            k = kTest(icosamp);
            if (k==0)
                k = sum(abs(x) > 1e-3);
            end
            uhat = cosamp(A,cnt,k,cosampOpt);            
            xhat = uhat(1:nx);  
            xhatCS = xhat;
            
        elseif (meth == GAMP_METH)
            
           %iterOpt.pcoeffInit = [outFn.scale 0]';
           iterOpt.scale = outFn;
            
           [xhat,pcoeff,plike] = gampEstPoly(cnt,A0,estimWt,iterOpt,gampOpt);
           xhatGAMP = xhat;
           
           % Save the poly coefficient
           np1 = length(pcoeff);
           pcoeffTot(npoly-np1+2:npoly+1,itest) = pcoeff;
            
        elseif (meth == LS_METH)
            
            % Compute LS solution         
            zmean = A0*repmat(xmean0,nx,1);
            scale = outFn.scale;
            xhat = xvar0*A0'* ((diag(zmean) + scale*xvar0.*A0*A0') \ (cnt-scale*zmean)) ...
                + xmean0;
        end
        
        % Normalize solution
        xhat = xhat*norm(x)/norm(xhat);
        xhatTot(:,itest,imeth) = xhat;
        
        % MSE
        if (meth == COSAMP_METH)
            kstr = sprintf(' k=%d', k);
        else
            kstr = '';
        end
        mseTot(itest,imeth) = norm(xhatTot(:,itest,imeth)-x)^2;
        fprintf(1,'it=%d meth=%s %s MSE=%f\n', itest, methStr{meth}, kstr, ...
            10*log10(mseTot(itest,imeth)/sum(abs(x).^2)) );
                          
    end        
    
     if ~newdat
            return
     end
    
end

if (savedat)
    cmd = sprintf('save data/connSim%s_nz%d methTest kTest xhatTot xTot nz nx ntest', datStr, nz);
    eval(cmd);
end
    

xTot1 = xTot(:);
mseAvg = zeros(nmeth,1);

plotStr = {'b-', 'g-', 'r', 'c'};
for imeth = 1:nmeth
    xhat1 = xhatTot(:,:,imeth);
    xhat1 = xhat1(:);
    
    mseAvg(imeth) = 10*log10( sum(abs(xhat1-xTot1).^2) / sum(abs(xTot1).^2));
    [perr, topt, pfa, pmd] = perrThresh(xTot(:), xhat1);            
    semilogx(pfa,pmd, plotStr{imeth});
    hold on;
end
hold off;