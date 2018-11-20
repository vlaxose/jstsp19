% testSNIPE:  tests SNIPE regularization at output 
%
% In this problem, we want to solve for x such that
%
%   y = B*x where nx > ny
%     & D*x contains zeros at unknown locations
%
% where B and D are random matrices and y is a given vector.  

% Set path to the main directory
addpath('../main/');

% Handle random seed
if exist('seed')
    rngseed = seed;
else
    rngseed = round(rand*1e8);
    disp(['seed=' num2str(rngseed)])
end
rng(rngseed);

% Problem Parameters
nx = 500;         % Number of input components (dimension of x)
delta = 0.7;      % Oversampling ratio on observations y
sigma = 2;        % Oversampling ratio on dictionary D
rho = .1;         %difficulty level (0<rho<1 , rho = (nx - numzeros(D*x))/ny
SNRdB = 40;      % Observation SNR

% GAMP Parameters
omega = 3;      % important to optimize this!!
xvar0 = 1e5;      % input-channel regularization: make large to avoid bias
cpx = true;

if cpx
    randmat=@(nrows,ncols) (randn(nrows,ncols)+1j*randn(nrows,ncols))*sqrt(.5);
else
    randmat=@(nrows,ncols) randn(nrows,ncols);
end

% Create measurement matrix 
ny = round(delta*nx);  % Number of output components (dimension of y)
B = (1/sqrt(nx))*randmat(ny,nx); % measurement matrix (approximately unit row-norm)

% Create analysis dictionary
n1 = nx - round(rho*ny);
n2 = round(sigma*nx) - n1;
D1 = (1/sqrt(nx))*randmat(n1,nx);  % random rows to which our signal will be orthogonal
D2 = (1/sqrt(nx))*randmat(n2,nx);  % other rows, not orthogonal to our signal
D = [D1;D2]; % could shuffle the rows

% Create signal in the nullspace of D1
N = null(D1);
x = N*randmat(nx-n1,1);
x = x/norm(x)*sqrt(nx); % make it xvar=1
z = B*x;

% Create noisy observations
wvar = (1/ny)*10^(-SNRdB/10)*norm(z)^2; % noise variance
y = z + randmat(ny,1)*sqrt(wvar); % noisy measurements
SNRdB_test = 10*log10(norm(z)^2/norm(y-z)^2);

% Create GAMP matrix
if 1
  A = [B;D]; % one way
else
  A = LinTransConcat({MatrixLinTrans(B);MatrixLinTrans(D)}); % another way
end

% Create a GAMP input estimation class corresponding to a (nearly) trivial prior
if 0
    if cpx
        inputEst = CAwgnEstimIn(0, xvar0); % one way
    else
        inputEst = AwgnEstimIn(0, xvar0); % one way
    end
else
    inputEst = NullEstimIn(0, xvar0,'maxSumVal',false,'isCmplx',cpx); % another way (may diverge if omega too low!)
end

% Create a GAMP output estimation class corresponding to the noiseless observations.
wvar0 = max(wvar,1e-5);     % output-channel regularization: make small to avoid bias
if cpx
    outputEst1 = CAwgnEstimOut(y, wvar0); % measurement likelihood 
else
    outputEst1 = AwgnEstimOut(y, wvar0); % measurement likelihood 
end
outputEst2 = SNIPEstim(omega,'isCmplx',cpx); % dictionary output likelihood
outputEst = EstimOutConcat({outputEst1,outputEst2},[ny,n1+n2]);

% Set the default options
opt = GampOpt();
opt.nit = 1000;
opt.tol = 1e-10;
opt.uniformVariance=0;
opt.pvarMin=0;
opt.xvarMin=0;
opt.adaptStep=1;
opt.adaptStepBethe=1;
%opt.step = 0.1; opt.stepMax = 0.5;
opt.stepWindow = 20; %inf disables adaptive stepsize while still plotting val

% Run the GAMP algorithm
tic
[xhat,~,~,~,~,~,~,~,estHist] = gampEst(inputEst, outputEst, A, opt);
timeGAMP = toc;

% Plot the results
figure(1); clf;
[xsort,I] = sort(x);
handy = plot(real(xsort), real(xsort),'-', real(xsort),real(xhat(I)),'g.');
%set(handy(2),'MarkerSize',8);
%set(handy(3),'MarkerSize',8);
set(gca,'FontSize',16);
grid on;
legend('X', 'GAMP estimate');
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

% how well would the cosupport genie do?
mseGenie =  20*log10( norm( x - N*(B*N\y) )/norm(x));

fprintf(1,'GAMP: MSE = %5.5f dB  (cosupport genie MSE = %5.5f dB)\n', mseGAMP,mseGenie);
