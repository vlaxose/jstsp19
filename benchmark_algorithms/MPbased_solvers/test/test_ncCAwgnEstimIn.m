% test phase-retrieval GAMP
%
% In this problem, x is a Gaussian random vector, that we want to
% estimate from measurements of the form
%
%   y = A*x + w,
%
% where A is a random matrix and w is Gaussian noise, and given knowledge
% of abs(x).

% Set path to the main directory
%addpath('../../main/');

% set randn and rand seeds
reset(RandStream.getDefaultStream);        % uncomment to start at reference
%defaultStream.State = savedState;          % uncomment to use previous state
defaultStream = RandStream.getDefaultStream;% leave uncommented!
savedState = defaultStream.State;           % leave uncommented!

% Parameters
nx = 1000;         % Number of input components (dimension of x)
nz = 600;         % Number of output components (dimension of y) [0.58*nz]
snr = 30;         % SNR in dB. [keep <= 30 to avoid GAMP stepsize problems]
if nx>1000, testLMMSE = 0; else testLMMSE = 1; end;

% Create a random Gaussian vector
xmean0 = 0*ones(nx,1);
xvar0 = 1*ones(nx,1);
x = xmean0 + sqrt(xvar0/2).*(randn(nx,2)*[1;1i]);
if 0,
  x = x./abs(x);%optionally make constant modulus
end;

% Create a random measurement matrix
A = (1/sqrt(2*nx))*(randn(nz,nx)+1i*randn(nz,nx));

% Compute the noise level based on the specified SNR. 
wvar = 10^(-0.1*snr)*mean(abs(x).^2)*ones(nz,1);

% Generate the noise 
w = sqrt(wvar/2).*(randn(nz,2)*[1;1i]);
y = A*x + w;

% Create an input estimation class corresponding to a noncoherent Gaussian 
inputEst = ncCAwgnEstimIn(abs(x),(abs(x).^2)*1e-2);% [dflt=(|x|^2)*1e-2]
%inputEst = CAwgnEstimIn(xmean0,xvar0);
%inputEst = PhaseEstimIn; % jason's: works only with unit-modulus x

% Create an output estimation class corresponding to the Gaussian noise.
outputEst = CAwgnEstimOut(y, wvar);

% Set the default options
opt = GampOpt();
%opt.step = 0.1;
%opt.adaptStep = 0; 	%[dflt=1]
%opt.stepMax = .1;	%[dflt=0.1??]
opt.stepWindow = 20;	%[dflt=20]
opt.tol = 1e-4;		%[dflt=1e-4]
opt.nit = 500;		%[dflt=500]

% Optional initialization
%opt.xhat0 = x.*exp(1i*2*pi*randn(nx,1)*0);

% Run the GAMP algorithm
tic
[xhat, xvar] = gampEst(inputEst, outputEst, A, opt);
timeGAMP = toc 

% Display the MSE
mseGAMP = 20*log10( norm(x-xhat)/norm(x));
fprintf(1,'ncGAMP: MSE = %5.1f dB\n', mseGAMP);

% Now compute the LMMSE estimate
if testLMMSE,
tic
xhatLMMSE = xmean0 + xvar0.*(A'*((A*diag(xvar0)*A'+diag(wvar))\(y-A*xmean0)));
timeLMMSE = toc;
mseLMMSE = 20*log10( norm(x-xhatLMMSE)/norm(x));
fprintf(1,'LMMSE:  MSE = %5.1f dB\n', mseLMMSE);
else 
timeLMMSE = NaN;
end;

% Plot the results
x_ = [real(x);imag(x)];
xhat_ = [real(xhat);imag(xhat)];
[xsort_,I] = sort(x_);
jim = plot(xsort_, xsort_,'b.',xsort_,xhat_(I),'g.');
leg_str = strvcat('True', 'GAMP estimate');
if testLMMSE,
  xhatLMMSE_ = [real(xhatLMMSE);imag(xhatLMMSE)];
  hold on; jim = [jim;plot(xsort_,xhatLMMSE_(I), 'r.')]; hold off;
  leg_str = strvcat(leg_str,'LMMSE estimate');
end;
legend(jim,leg_str,'Location','Best');
xlabel('True value of x');
ylabel('Estimate of x');

