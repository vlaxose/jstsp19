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
if verLessThan('matlab','7.14')
  defaultStream = RandStream.getDefaultStream;
else
  defaultStream = RandStream.getGlobalStream;
end;
if 1
    savedState = defaultStream.State;
    save random_state.mat savedState;
else
    load random_state.mat
end
defaultStream.State = savedState;

% Parameters
nx = 1000;         % Number of input components (dimension of x)
nz = 500;         % Number of output components (dimension of y)
sparseRat = 0.2;  % fraction of components of x that are non-zero
snr = 60;         % SNR in dB.

% Create a random sparse vector
xmean0 = 0;        
xvar0 = 1;
x0 = normrnd(xmean0, sqrt(xvar0),nx,1); % a dense Gaussian vector
x = x0.*(rand(nx,1) < sparseRat);       % insert zeros

% Create a random measurement matrix
%A = (1/sqrt(nx))*randn(nz,nx);		% iid Gaussian, zero-mean
A = (1/sqrt(nx))*rand(nz,nx) + 0.021; % iid uniform, positive
if 0,
  A = A-mean(A,2)*ones(1,nx);		% make exactly zero row-mean
  mu = rand(nz,1); 			% desired row-mean 
  A = A + mu*ones(1,nx);		% force row-mean
elseif 0
  A = A-ones(nz,1)*mean(A,1);		% make exactly zero column-mean
  mu = 0.05*rand(1,nx);   		% desired column-mean 
  A = A + ones(nz,1)*mu;		% force column-mean
end;

% Compute the noise level based on the specified SNR. Since the components
% of A have power 1/nx, the E|y(i)|^2 = E|x(j)|^2 = sparseRat.  
%wvar = 10^(-0.1*snr)*xvar0*sparseRat;
wvar = 10^(-0.1*snr)*(norm(A*x)^2)/nz;
w = normrnd(0, sqrt(wvar), nz,1); % AWGN
z = A*x;
y = z + w;

% Test MMV version
%y = [y,y];

% Generate input estimation class
% First, create an estimator for a Gaussian random variable (with no
% sparsity)
inputEst0 = AwgnEstimIn(xmean0, xvar0);

% Then, create an input estimator from inputEst0 corresponding to a random
% variable x that is zero with probability 1-sparseRat and has the
% distribution of x in inputEst0 with probability sparseRat.
inputEst = SparseScaEstim( inputEst0, sparseRat );

% Output estimation class:  Use the 
outputEst = AwgnEstimOut(y, wvar);

% Set GAMP options
opt = GampOpt();
opt.legacyOut = false;
opt.uniformVariance = 0; 
opt.tol = 1e-5;
opt.nit = 2000;
opt.adaptStep = 1;	
opt.adaptStepBethe = 1;	
opt.stepWindow = 0;	
opt.stepMax = 1.0;	
opt.stepMin = 0.1;	% seems important to be >0
opt.step = opt.stepMax;	
%opt.stepIncr = 1.0;
%opt.stepWindow = 10;
%opt.maxBadSteps = 5;
opt.varNorm = 0;	
opt.xvarMin = 0;	
opt.pvarMin = 0;	
opt.pvarStep = 1;	
opt.rvarStep = 1;	

% Run GAMP without mean removal
opt.removeMean = 0;
estFin = gampEst(inputEst, outputEst, A, opt);
xhat_naive = estFin.xhat(:,1);% for MMV version
nmseGAMP_naive = 20*log10( norm(x-xhat_naive)/norm(x) )

% Run GAMP with internal row/column-mean removal
opt.removeMean = 1;
opt.removeMeanExplicit = 1;
tstart1 = tic; estFin1 = gampEst(inputEst, outputEst, A, opt); toc(tstart1)
[estFin1,optFin1,estHist1] = gampEst(inputEst, outputEst, A, opt);
xhat_demean = estFin1.xhat(:,1);% for MMV version
nmseGAMP_demean = 20*log10( norm(x-xhat_demean)/norm(x) ) 

% Run GAMP with internal row/column-mean removal
opt.removeMean = 1;
opt.removeMeanExplicit = 0;
tstart2 = tic; estFin2 = gampEst(inputEst, outputEst, A, opt); toc(tstart2)
xhat_demean2 = estFin2.xhat(:,1);% for MMV version
nmseGAMP_demean2 = 20*log10( norm(x-xhat_demean2)/norm(x) ) 

% support oracle 
supp = find(x~=0);
xhat_genie = zeros(nx,1);
xhat_genie(supp) =( A(:,supp)'*A(:,supp) + (wvar/xvar0)*eye(length(supp)) )\(A(:,supp)'*y(:,1));
nmseGENIE = 20*log10( norm(x-xhat_genie)/norm(x) ) 

% Plot the results
figure(1)
stem(x,'rx')
axe = axis;
hold on;
  stem(xhat_naive,'gs');
  stem(xhat_demean,'--b');
hold off;
axis(axe);
grid on;
legend('True',...
       ['GAMP no de-mean, NMSE=',num2str(nmseGAMP_naive,3),'dB'],...
       ['GAMP int de-mean, NMSE=',num2str(nmseGAMP_demean,3),'dB']);

% plot step and val trajectories 
figure(2)
subplot(311)
 if 1
   plot(estHist1.val)
   ylabel('val')
 else
   maxVal = max(estHist1.val);
   semilogy(maxVal-estHist1.val+eps);
   ylabel('max(val)-val')
 end
 grid on
 title('mean-removal')
subplot(312)
 plot(estHist1.step)
 grid on
 ylabel('step')
subplot(313)
 plot(10*log10(sum(abs(estHist1.xhat(1:nx,:)-x*ones(1,estFin1.nit)).^2,1)./norm(x).^2))
 grid on
 ylabel('NMSE [dB]')
%  semilogy([abs(estHist1.shat(nz+1:nz+2,:))]')
%  grid on
%  ylabel('augmented shat variables')

% Demonstrate automatic selection of xvar0 
if 0
  opt2 = opt;
  opt2.xvar0auto = true;
  opt2.xhat0 = xhat_demean + 0.01*randn(nx,1)*norm(x)/sqrt(nx); % start close to final solution
  [estFin2,optFin2,estHist2] = gampEst(inputEst, outputEst, A, opt2);
  figure(3); clf;
  gampShowHist(estHist2,optFin2,x,z);
end

