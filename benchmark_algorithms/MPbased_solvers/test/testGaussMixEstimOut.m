%Simple code to test GaussMixEstimOut Class

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


%Note to the user: We still need to implement the log-likelihood
%calculation so we can enable adaptive step sizes for the Gaussian Mixture
%model. This will likely eliminate the transients observed on some
%realizations. 

% Set path
addpath('../main/');

%% Setup and global options
%Specify problem size
N = 1000;
del = 0.5; %ratio of m/n
rho = 0.3; %ratio of sparsity to number of measurements

%Specify noise model.
%Model is (1-lambda) Nor(0,nu0) + lambda Nor(0,nu1)
nu0 = 1e-3;
nu1 = 500e-3;
lambda = 0.1; % 0.01


%Derive sizes
%Set M and K
M = ceil(del*N);
K = floor(rho*M);


%Set options for GAMP
GAMP_options = GampOpt; %initialize the options object
GAMP_options.nit = 200;
GAMP_options.verbose = 0;
GAMP_options.removeMean = 0;
GAMP_options.pvarMin = 0;
GAMP_options.xvarMin = 0;
GAMP_options.tol = 1e-4;

GAMP_options.stepWindow = 3;
GAMP_options.adaptStep = 1;
GAMP_options.adaptStepBethe = 1;


%% Generate the forward operator

%Avoid zero K
if K == 0
    K = 1;
end

%Compute column normalized A matrix
A = randn(M,N);
A = A*diag(1 ./ sqrt(diag(A'*A)));



%% Generate the true signal

%Determine true bits
truebits = false(N,1);
truebits(1:K) = true; %which bits are on

%Generate the signal
x = randn(N,1) .* truebits;



%Generate the uncorrupted measurements
z = A*x;

%% Generate noisy signal

%Noisy output channel
inds = rand(size(z)) < lambda;
y = z + sqrt(nu0)*randn(size(z)).*(1-inds)...
    + sqrt(nu1)*(randn(size(z))).*inds;



%% GAMP with Gaussian noise model

%Input channel
inputEst = AwgnEstimIn(0, 1);
inputEst = SparseScaEstim(inputEst,K/N);

%Output channel
outputEst = AwgnEstimOut(y, (1-lambda)*nu0 + lambda*nu1);

%Run GAMP
[resGAMP,~,~,~,~,~,~,~, estHistGAMP] = ...
    gampEst(inputEst, outputEst, A, GAMP_options);

%Compute error values
errGAMP = zeros(size(estHistGAMP.xhat,2),1);
for kk = 1:length(errGAMP)
    errGAMP(kk) = norm(x - estHistGAMP.xhat(:,kk)) / norm(x);
end

%% GAMP with Gaussian Mixture noise model

%Output channel
outputEst2 = GaussMixEstimOut(y,nu0,nu1,lambda);

%Run GAMP
[resGAMP2,~,~,~,~,~,~,~, estHistGAMP2] = ...
    gampEst(inputEst, outputEst2, A, GAMP_options);

%Compute error values
errGAMP2 = zeros(size(estHistGAMP2.xhat,2),1);
for kk = 1:length(errGAMP2)
    errGAMP2(kk) = norm(x - estHistGAMP2.xhat(:,kk)) / norm(x);
end



%% Plot results

%Show the results
figure(1)
clf
plot(abs(x),'ko')
hold on
plot(abs(resGAMP),'bx')
plot(abs(resGAMP2),'r+')
legend('Truth','GAMP Gaussian Noise Model','GAMP Gaussian Mixture Noise Model')
title(['GAMP AWGN NMSE: ' num2str(20*log10(errGAMP(end)))...
    ' dB;   GAMP Mix  NMSE: ' num2str(20*log10(errGAMP2(end)))...
    ]);
axis([0 N -.2 5])
grid

%Show convergence history
figure(2)
clf
subplot(311)
  plot(20*log10(abs(errGAMP)),'b-')
  hold on; plot(20*log10(abs(errGAMP2)),'r--'); hold off
  grid
  xlabel('iteration')
  ylabel('NMSE (dB)')
  legend('GAMP AwgnEstimOut','GAMP GaussMixEstimOut','location','best')
subplot(312)
  plot(estHistGAMP.val,'b-')
  hold on; plot(estHistGAMP2.val,'r--'); hold off
  grid
  ylabel('val')
  legend('GAMP AwgnEstimOut','GAMP GaussMixEstimOut','location','best')
subplot(313)
  plot(estHistGAMP.step,'b-')
  hold on; plot(estHistGAMP2.step,'r--'); hold off
  grid
  ylabel('step')
  legend('GAMP AwgnEstimOut','GAMP GaussMixEstimOut','location','best')
