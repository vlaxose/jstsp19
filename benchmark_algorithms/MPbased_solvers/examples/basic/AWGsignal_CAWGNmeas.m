% AWGsignal_CAWGNmeas: a real-Gaussian vector in complex-Gaussian observations.
%
% In this problem, x is a real-Gaussian random vector, that we want to
% estimate from complex measurements of the form
%
%   y = A*x + w,
%
% where A is a random complex matrix and w is complex-Gaussian noise.  
% This can be rewritten and solved using a classical least squares 
% formulation and can be easily performed in MATLAB
% without the GAMP method.  But, you can look at this example, just to
% understand the syntax of the gampEst function.

% Set path to the main directory
addpath('../../main/');

%Handle random seed
defaultStream = RandStream.getDefaultStream;
if 1
    savedState = defaultStream.State;
    save random_state.mat savedState;
else
    load random_state.mat
end
defaultStream.State = savedState;

% Parameters
nx = 1000;         % Number of input components (dimension of x)
nz = 2000;         % Number of output components (dimension of y)
snr = 100;          % SNR in dB. 

% Create a random Gaussian vector
xmean0 = zeros(nx,1);
xvar0 = ones(nx,1);
x = xmean0 + xvar0.*randn(nx,1);

% Create a random measurement matrix
A = (1/sqrt(2*nx))*(randn(nz,nx)+1i*randn(nz,nx));

% Compute the noise level based on the specified SNR. Since the components
% of A have power 1/nx, the E|y(i)|^2 = E|x(j)|^2 = |xmean|^2+xvar.
wvar = 10^(-0.1*snr)*mean(abs(xmean0).^2+xvar0)*ones(nz,1);

% Generate the noise 
w = sqrt(wvar/2).*(randn(nz,2)*[1;1i]);
y = A*x + w;

% Decide on MAP or MMSE GAMP
map = 0;

% Create an input estimation class corresponding to a real-Gaussian vector
inputEst = AwgnEstimIn(xmean0, xvar0, map);

% Create an output estimation class corresponding to complex-Gaussian noise.
% Note that the observation vector is passed to the class constructor, not
% the gampEst function.
outputEst = CAwgnEstimOut(y, wvar, map);

% Set the default options
opt = GampOpt();
opt.nit = 1000;	
opt.tol = 1e-8;	
opt.uniformVariance=1;
%opt.stepIncr=1.1;
%opt.varNorm=1;
%opt.pvarStep=1;

% Run the GAMP algorithm
tic
[xhat,~,~,~,~,~,~,~,estHist] = gampEst(inputEst, outputEst, A, opt);
timeGAMP = toc;

% Now perform the exact LMMSE solution
tic
Ari = [real(A);imag(A)];
wri = [real(w);imag(w)];
yri = Ari*x + wri;
wvarri = [wvar/2;wvar/2];
xhatLMMSE = xmean0 + xvar0.*(Ari'*((Ari*diag(xvar0)*Ari'+diag(wvarri))\(yri-Ari*xmean0)));
timeLMMSE = toc;

% Plot the results
figure(1); clf;
[xsort,I] = sort(real(x));
plot(xsort, xsort,'-', xsort,[real(xhat(I)) real(xhatLMMSE(I))], '.');
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
  semilogy(estHist.val)
  ylabel('val')
  xlabel('iteration')
  grid on

% Display the MSE
mseGAMP = 20*log10( norm(x-xhat)/norm(x));
mseLMMSE = 20*log10( norm(x-xhatLMMSE)/norm(x));
fprintf(1,'GAMP:  MSE = %5.1f dB\n', mseGAMP);
fprintf(1,'LMMSE: MSE = %5.1f dB\n', mseLMMSE);

