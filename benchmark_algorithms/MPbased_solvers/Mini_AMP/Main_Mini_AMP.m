% sparseAWGN:  Example of estimating a sparse vector with Gaussian noise.
%
% In this problem, x is a Bernoulli-Gaussian random vector, that we want to
% estimate from measurements of the form
%
%   y = A*x + w,
%
% where A is a random matrix and w is Gaussian noise.  This is a classical
% compressed sensing problem of estimating a sparse vector for random
% linear measurements.

% Set path to the main directory
addpath('../../main/');

%Handle random seed
defaultStream = RandStream.getGlobalStream;

savedState = defaultStream.State;
save random_state.mat savedState;

defaultStream.State = savedState;

% Parameters
nx = 1000;         % Number of input components (dimension of x)
nz = 500;         % Number of output components (dimension of y)
sparseRat = 0.1;    % fraction of components of x that are non-zero
SNRdB = 20;           % SNR in dB.

% Create a random sparse vector
xmean0 = 0;
xvar0 = 1;
x0 = normrnd(xmean0, sqrt(xvar0),nx,1); % a dense Gaussian vector
x = x0.*(rand(nx,1) < sparseRat);       % insert zeros

% Create a random measurement matrix
A = (1/sqrt(nz))*randn(nz,nx);

% optionally flatten the singular values
if 1
  nr = min(nx,nz);
  [~,~,V] = svd(A,'econ');
  A = sqrt(nx/nr)*V'; % ensures that nx=norm(A,'fro')^2
end

% Compute the noise level based on the specified SNR. Since the components
% of A have power 1/nz, the E|y(i)|^2 = (nx/nz)*E|x(j)|^2 
wvar = 10^(-SNRdB/10)*(nx/nz)*(xmean0^2+xvar0)*sparseRat;
w = normrnd(0, sqrt(wvar), nz, 1);
z = A*x;
y = z + w;

% Generate input estimation class
% First, create an estimator for a Gaussian random variable (with no
% sparsity)
inputEst0 = AwgnEstimIn(xmean0, xvar0);

% Then, create an input estimator from inputEst0 corresponding to a random
% variable x that is zero with probability 1-sparseRat and has the
% distribution of x in inputEst0 with probability sparseRat.
inputEst = SparseScaEstim( inputEst0, sparseRat );

% Monitor progress
%inputEst = TruthReporter( inputEst, x);

% Output estimation class:  Use the one for AWGN.
outputEst = AwgnEstimOut(y, wvar);

% Set options
opt = GampOpt();
  opt.legacyOut = false;
  opt.tol = 1e-4; % decrease for high SNR
optA = AmpOpt();
  optA.tol = opt.tol;
  optA.rvarMethod = 'wvar'; optA.wvar = wvar; % optional 
  %optA.rvarMethod = 'mean'; % optional

% Optional: run S-AMP from [Cakmak,Winther,Fleury] instead of standard AMP
runSAMP = false;
if runSAMP
  optA.Stransform = true;
  optA.wvar = wvar; % need to specify wvar when running S-AMP
  AMPstr = 'SAMP';
else
  AMPstr = 'AMP';
end

% Run the GAMP algorithm
[estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, opt);
xhat = estFin.xhat;

% Run the AMP algorithm
[estFinA,optFinA,estHistA] = Mini_AMP(inputEst, y, A, optA);
xhatA = estFinA.xhat;

% Plot the results
figure(1); clf; set(gcf,'Name',[AMPstr,' and GAMP'])
subplot(211)
  h = stem([x, xhatA, xhat]);
  set(h(2),'Marker','x')
  set(h(3),'Marker','+')
  grid on;
  legend('True', [AMPstr,' Estimate'], 'GAMP Estimate');
subplot(212)
  h = stem([xhatA-x, xhat-x]);
  set(h(1),'Marker','x')
  set(h(2),'Marker','+')
  grid on;
  legend([AMPstr,' Error'], 'GAMP Error');

figure(2); clf; set(gcf,'Name','GAMP')
gampShowHist(estHist,optFin,x);

figure(3); clf; set(gcf,'Name',AMPstr)
gampShowHist(estHistA,optFinA,x);

% Display the MSE
nmseGAMP = 20*log10( norm(x-xhat)/norm(x));
fprintf(1,'GAMP: NMSE = %5.2f dB\n', nmseGAMP);
nmseAMP = 20*log10( norm(x-xhatA)/norm(x));
fprintf(1,'%s: NMSE = %5.2f dB\n', AMPstr,nmseAMP);

% Demonstrate automatic selection of xvar0 
if 0
  opt2 = opt;
  opt2.xvar0auto = true;
  opt2.xhat0 = estFin.xhat + 0.1*randn(nx,1)*norm(x)/sqrt(nx); % start close to final solution
  [estFin2,optFin2,estHist2] = gampEst(inputEst, outputEst, A, opt2);
  figure(4); clf; set(gcf,'Name','GAMP from xhat0')
  gampShowHist(estHist2,optFin2,x,z);

  opt2A = AmpOpt();
  opt2A.xhat0 = opt2.xhat0;
  [estFin2A,optFin2A,estHist2A] = ampEst(inputEst, y, A, opt2A);
  figure(5); clf; set(gcf,'Name',[AMPstr,' from xhat0']); 
  gampShowHist(estHist2A,optFin2A,x);
end


