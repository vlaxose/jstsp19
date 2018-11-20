% Test new auto tune method for a 2 component gaussian mixture using the
% mixScaEstim function.  
%
%Coded by Jeremy Vila
%10-21-14

clear all

M = 1000; %number of measurements
N = 500;  %length of signal

T = 1; %number of columns

%Set prior means and variance
mn0 = -1;
mn1 = 2;
vr0 = 1e-8;
vr1 = 1;
lambda = 0.4;  %Define ratio of elements in gaussian component 1

% Generate signal distribution
gXtmp0 = AwgnEstimIn(mn0,vr0);
gXtmp1 = AwgnEstimIn(mn1,vr1);
gXtmp = MixScaEstimIn(gXtmp1, lambda, gXtmp0);
x = gXtmp.genRand([N T]);

%Generate iid zero-mean Gaussian matrix;
Amat = randn(M,N);
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns
A = MatrixLinTrans(Amat);

% Generate noiseless and noisy measurements
z = A.mult(x);

SNRdB = 30;
% Find appropriate noise variance
muw = norm(z,'fro')^2/M/T*10^(-SNRdB/10);

% Compute noisy output
y = z + sqrt(muw)*(randn(M,T));


%% Set up and run GAMP
%Set GAMP options
optGAMP = GampOpt();
optGAMP.xvarMin = 0;
optGAMP.pvarMin = 0;
optGAMP.legacyOut = false;
optGAMP.verbose = true;
optGAMP.stepWindow = 0;
optGAMP.tol = 1e-8;
optGAMP.adaptStep = true;
optGAMP.adaptStepBethe = true;

%Define postulated prior
dim = 'joint';
gX0 = AwgnEstimIn(0.5,1, 0, 'autoTune', true);
gX1 = AwgnEstimIn(-0.5,0.5, 0, 'autoTune', true);
gX = MixScaEstimIn(gX1, 0.5, gX0, 'autoTune', true);

%Define true output channel
gOut = AwgnEstimOut(y,muw);


%% Run GAMP
tic
estFin = gampEst(gX, gOut, A, optGAMP);
time = toc
xhat = estFin.xhat;

%Compare results
NMSEdB = 10*log10(norm(x-xhat,'fro')^2/norm(x,'fro')^2)