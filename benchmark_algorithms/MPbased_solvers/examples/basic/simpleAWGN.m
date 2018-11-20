% simpleAWGN:  Simple example of estimating a Gaussian vector.
%
% In this problem, x is a Gaussian random vector that we want to
% estimate from measurements of the form
%
%   y = A*x + w,
%
% where A is a random matrix and w is Gaussian noise.  This is a classical
% LMMSE estimation problem and can be easily performed in MATLAB
% without the GAMP method.  But, you can look at this example, just to
% understand the syntax of the gampEst function.

% Set path to the main directory
addpath('../../main/');

% Handle random seed
if verLessThan('matlab','7.14')
  defaultStream = RandStream.getDefaultStream;
else
  defaultStream = RandStream.getGlobalStream;
end;
if 1 % use new random seed
    savedState = defaultStream.State;
    save random_state.mat savedState;
else % reuse old random seed
    load random_state.mat
end
defaultStream.State = savedState;

% Parameters
nx = 100;         % Number of input components (dimension of x)
nz = 200;         % Number of output components (dimension of y)
snr = 100;        % SNR in dB.

% Create a random Gaussian vector
xmean0 = 0;
xvar0 = 1;
x = xmean0*ones(nx,1) + sqrt(xvar0)*randn(nx,1);

% Create a random measurement matrix
A = (1/sqrt(nx))*randn(nz,nx);

% Compute the noise level based on the specified SNR. Since the components
% of A have power 1/nx, the E|y(i)|^2 = E|x(j)|^2 = xmean^2+xvar.
wvar = 10^(-0.1*snr)*(xmean0^2+xvar0);

% Generate the noise 
w = sqrt(wvar)*randn(nz,1);
y = A*x + w;

% Decide on MAP or MMSE GAMP
map = 0;

% Create an input estimation class corresponding to a Gaussian vector
inputEst = AwgnEstimIn(xmean0, xvar0, map);

% Create an output estimation class corresponding to the Gaussian noise.
% Note that the observation vector is passed to the class constructor, not
% the gampEst function.
outputEst = AwgnEstimOut(y, wvar, map);

% Set the default options
opt = GampOpt();
opt.nit = 10000;
opt.tol = max(min(1e-3, 10^(-snr/10)),1e-15);
opt.uniformVariance=0;
opt.pvarMin=0;
opt.xvarMin=0;
opt.adaptStep=true;
opt.adaptStepBethe=true;
opt.legacyOut=false;

% Demonstrate automatic selection of xvar0 
if 0
  opt.xvar0auto = true;
  opt.xhat0 = x + 0.01*randn(nx,1)*norm(x)/sqrt(nx); 
end;

% Run the GAMP algorithm
tic
[estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, opt);
xhat = estFin.xhat;
timeGAMP = toc;

% Now perform the exact LMMSE solution
tic
xhatLMMSE = xmean0 + (A'*A + wvar/xvar0*eye(nx))\(A'*(y-A*ones(nx,1)*xmean0));
timeLMMSE = toc;

% Plot the results
figure(1); clf;
[xsort,I] = sort(x);
handy = plot(xsort, xsort,'-', xsort,xhat(I),'g.',xsort,xhatLMMSE(I), 'r.');
%set(handy(2),'MarkerSize',8);
%set(handy(3),'MarkerSize',8);
set(gca,'FontSize',16);
grid on;
legend('True', 'GAMP estimate', 'LMMSE estimate');
xlabel('True value of x');
ylabel('Estimate of x');

figure(2); clf;
 subplot(311)
  plot(10*log10(sum(abs( estHist.xhat - x*ones(1,size(estHist.xhat,2)) ).^2,1)/norm(x)^2))
  ylabel('NMSE [dB]')
  grid on
 subplot(312)
  plot(estHist.step)
  ylabel('step')
  grid on
 subplot(313)
  plot(estHist.val)
  ylabel('val')
  xlabel('iteration')
  grid on

% Display the MSE
mseGAMP = 20*log10( norm(x-xhat)/norm(x));
mseLMMSE = 20*log10( norm(x-xhatLMMSE)/norm(x));
fprintf(1,'GAMP: MSE = %5.5f dB\n', mseGAMP);
fprintf(1,'LMMSE:   MSE = %5.5f dB\n', mseLMMSE);

