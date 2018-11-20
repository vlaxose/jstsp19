%Simple code to test LaplaceEstimOut Class
clear all
clc

%Note to the user: We still need to implement the log-likelihood
%calculation so we can enable adaptive step sizes for the Gaussian Mixture
%model. This will likely eliminate the transients observed on some
%realizations. 

% Set path
addpath('../main/');

%% Setup and global options
%Specify problem size
N = 500;
del = 0.6; %ratio of m/n
rho = 0.2; %ratio of sparisty to number of measurements

%Specify noise model.
%Model is y = z + L(0,lambda)
%   where L(x;0,lambda) = lambda/2*exp(-1*lambda*abs(x))
lambda = 20;


%Derive sizes
%Set M and K
M = ceil(del*N);
K = floor(rho*M);



%Set options for GAMP
GAMP_options = GampOpt; %initialize the options object
GAMP_options.nit = 200;
GAMP_options.step = 0.25;
GAMP_options.adaptStep = 0;
GAMP_options.verbose = 0;
GAMP_options.removeMean = 0;
GAMP_options.pvarMin = 0;
GAMP_options.xvarMin = 0;
GAMP_options.tol = 1e-4;
GAMP_options.stepTol = -1; %don't use for now



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

%% Channel objects

%Input channel
inputEst = AwgnEstimIn(0, 1);
inputEst = SparseScaEstim(inputEst,K/N);


%Output channel
outputEst = LaplaceEstimOut(0,lambda);

%% Generate noisy signal

%Noisy output channel
y = outputEst.genRand(z);

%Set measurements
outputEst.y = y;

%Compute SNR
SNR = 20*log10(norm(z) / norm(y-z));

%Build an output estimator with Gaussian noise with matching variance
outputEstGaussian = AwgnEstimOut(y,2/lambda^2);

%% GAMP with Gaussian noise model



%Run GAMP
[resGAMP,~,~,~,~,~,~,~, estHistGAMP] = ...
    gampEst(inputEst, outputEstGaussian, A, GAMP_options);



%Compute error values
errGAMP = zeros(size(estHistGAMP.xhat,2),1);
for kk = 1:length(errGAMP)
    errGAMP(kk) = norm(x - estHistGAMP.xhat(:,kk)) / norm(x);
end


%% GAMP with Laplacian Noise Model


%Run GAMP
[resGAMP2,~,~,~,~,~,~,~, estHistGAMP2] = ...
    gampEst(inputEst, outputEst, A, GAMP_options);



%Compute error values
errGAMP2 = zeros(size(estHistGAMP2.xhat,2),1);
for kk = 1:length(errGAMP2)
    errGAMP2(kk) = norm(x - estHistGAMP2.xhat(:,kk)) / norm(x);
end

%Test EM code- not running EM in an iterative fashion, just checking that
%the estimate returned is reasonable
lam_estimate =...
    outputEst.updateRate(estHistGAMP2.phat(:,end),estHistGAMP2.pvar(:,end));

disp(['True lambda = ' num2str(lambda) ', EM update returned ' ...
    num2str(lam_estimate)])



%% Plot results

%Show the results
figure(1)
clf
plot(abs(x),'ko')
hold on
plot(abs(resGAMP),'bx')
plot(abs(resGAMP2),'r+')
legend('Truth','GAMP Gaussian Noise Model','GAMP Laplacian Noise Model')
title(['SNR = ' num2str(SNR) ' dB;  GAMP AWGN NMSE: ' num2str(20*log10(errGAMP(end)))...
    ' dB;   Laplacian  NMSE: ' num2str(20*log10(errGAMP2(end)))...
    ]);
axis([0 N -.2 5])
grid

%Show convergence history
figure(2)
clf
plot(20*log10(abs(errGAMP)),'b-')
hold on
plot(20*log10(abs(errGAMP2)),'r--')
xlabel('iteration')
ylabel('NMSE (dB)')
legend('GAMP AWGN','GAMP Laplacian Noise Model','location','best')
grid





