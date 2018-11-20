% simpleLS:  Simple example of least-squares estimation 
%
% In this problem, we want to solve for the least-squares solution
%
%   x = pinv(A)*y,
%
% where A is a random matrix and y is a given vector.  This is a classical
% problem and can be easily performed in MATLAB without the GAMP method.  
% But it helps to demonstrate how to apply GAMP.

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
nx = 200;         % Number of input components (dimension of x)
ny = 100;         % Number of output components (dimension of y)

% Create a random linear system and outputs 
A = (1/sqrt(nx))*randn(ny,nx);
y = randn(ny,1);

% Decide on MAP or MMSE GAMP
map = 0;

% Create a GAMP output estimation class corresponding to the noiseless observations.
wvar = 1e-5; % chosen small to avoid bias, but must be positive
outputEst = AwgnEstimOut(y, wvar, map);

% Create a GAMP input estimation class corresponding to the trivial prior
xvar0 = 1e5; % chosen large to avoid bias, where large is relative to wvar
inputEst = AwgnEstimIn(0, xvar0, map);
%inputEst = NullEstimIn(0, xvar0); % another way, often unstable

% Set the default options
opt = GampOpt();
opt.legacyOut=false;
opt.nit = 1000;
opt.tol = 1e-10;
opt.uniformVariance=false;
opt.pvarMin=0;
opt.xvarMin=0;
opt.adaptStep=true;
opt.adaptStepBethe=true;
opt.stepWindow = inf; % disables stepsize adaptation while still plotting val
opt.xvar0auto=true;

% Run the GAMP algorithm
tic
[estFin,optFin,estHist] = gampEst(inputEst, outputEst, A, opt);
timeGAMP = toc;

% Compute the exact LS solution
tic
x = pinv(A)*y;
timeLS = toc;

% Compute the regularized LS solution that GAMP is computing
tic
xhatLMMSE = (A'*A + wvar/xvar0*eye(nx))\(A'*y);
timeLMMSE = toc;

% Plot the results
figure(1); clf;
[xsort,I] = sort(x);
handy = plot(xsort, xsort,'-', xsort,estFin.xhat(I),'g.', xsort,xhatLMMSE(I),'r.');
%set(handy(2),'MarkerSize',8);
%set(handy(3),'MarkerSize',8);
set(gca,'FontSize',16);
grid on;
legend('Exact LS', 'GAMP estimate', 'Regularized LS');
xlabel('True value of x');
ylabel('Estimate of x');

figure(2); clf;
if 1
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
else
  gampShowHist(estHist,optFin,x,y)
end

% Display the MSE
mseGAMP = 20*log10( norm(x-estFin.xhat)/norm(x));
mseLMMSE = 20*log10( norm(x-xhatLMMSE)/norm(x));
fprintf(1,'GAMP: MSE = %5.5f dB\n', mseGAMP);
fprintf(1,'regLS: MSE = %5.5f dB\n', mseLMMSE);

