% Test new auto tune method for Bernoulli (complex) Gaussian Mixture signals, 
% using the GMEstimIn or CGMEstimIn functions
%
%Coded by Jeremy Vila
%10-21-14

clear all

%Define dimensions
N = 1000;   % number of signal elements
rho = 0.2;  % normalized sparsity ratio
del = 0.6;  % measurement ratio

M = ceil(del*N);  % number of measurements
K = ceil(rho*M);  % Sparsity of signal

T = 10; %number of columns

%Set prior mean and variance
theta = [-2 1]; theta = permute(theta, [3, 1, 2]);  % true means of components
phi = [1e-8 1e-3]; phi = permute(phi , [3, 1, 2]);  % true variance of components
omega = [0.3 0.7]; omega = omega/sum(omega);
omega = permute(omega , [3, 1, 2]);                 %True weights of components

% Generate signal distribution
complex = false;  % Flag to set whether signal is real or complex
if complex
    gXtmp1 = CGMEstimIn(omega, theta, phi);  
    Amat = randn(M,N) + 1i*randn(M,N);
else
    gXtmp1 = GMEstimIn(omega, theta, phi);
    Amat = randn(M,N);
end
gXtmp = SparseScaEstim(gXtmp1, K/N);  % Impose sparsity in signal
x = gXtmp.genRand([N T]);             % randomly draw signal

%Generate iid zero-mean Gaussian matrix;
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns
A = MatrixLinTrans(Amat);

% Generate noiseless and noisy measurements
z = A.mult(x);

SNRdB = 20;
% Find appropriate noise variance
muw = norm(z,'fro')^2/M/T*10^(-SNRdB/10);    

if complex
    gOutTmp = CAwgnEstimOut(z,muw);
else
    gOutTmp = AwgnEstimOut(z,muw);
end

% Compute noisy output
y = gOutTmp.genRand(z);   % generate noisy measurements


%% Set up and run GAMP
%Set GAMP options
optGAMP = GampOpt();
optGAMP.xvarMin = 0;
optGAMP.pvarMin = 0;
optGAMP.legacyOut = false;
optGAMP.verbose = true;
optGAMP.stepWindow = 0;
optGAMP.tol = 1e-5;
optGAMP.adaptStep = false;
optGAMP.adaptStepBethe = true;

%Define postulated prior
theta = [-1 0]; theta = permute(theta, [3, 1, 2]);
phi = [1e-1 1e-1]; phi = permute(phi , [3, 1, 2]);
omega = [0.5 0.5]; omega = omega/sum(omega);
omega = permute(omega , [3, 1, 2]);
dim = 'joint';
if complex
    gX1 = CGMEstimIn(omega, theta, phi, 'autoTune', true, 'tuneDim', dim);
    gOut = CAwgnEstimOut(y,muw);
else
    gX1 = GMEstimIn(omega, theta, phi, 'autoTune', true, 'tuneDim', dim);
    gOut = AwgnEstimOut(y,muw);
    Amat = randn(M,N);
end
gX = SparseScaEstim(gX1, K/N, 0, 'autoTune', true, 'tuneDim', dim);

%% Run GAMP
tic
estFin = gampEst(gX, gOut, A, optGAMP);
time = toc
xhat = estFin.xhat;

%Compare results
NMSEdB = 10*log10(norm(x-xhat,'fro')^2/norm(x,'fro')^2)