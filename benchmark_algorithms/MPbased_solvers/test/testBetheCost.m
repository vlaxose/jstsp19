% Jeremy Vila
% This script demonstrates how to invoke the new cost computing method and
% demonstrates performance for various output channels p_{Y|Z}(y|z).
% 
% 9-05-14

clear all; clf;

% Define problem dimensions
M = 1000;   %number of measurements
N = 500;    %number of signal elements
K = 200;    %signal sparsity level

%% Define signal prior to be Bernoulli Gaussian

%Define Bernoulli-Gaussian signal
gX = AwgnEstimIn(0,1);
gX = SparseScaEstim(gX,K/N);
%Generate the random BG signal
x = gX.genRand([N,1]);

%% Generate iid Gaussian sensing matrix with normalized columns

%Generate matrix
Amat = randn(M,N);
%Normalize the columns
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns

% Make matrix A a LinTrans class for GAMP;
A = MatrixLinTrans(Amat);

%Generate noiseless measurements
z = A.mult(x);

%% Generate noisy measurements based on desired output channel
% pYZtype must belong to the set {'AWGN','AWBGN','AWLN','Dirac','AWGMN'}
% See case statement comments for description

pYZtype = 'Dirac'; 
switch pYZtype
    case 'AWGN' %Additive white Gaussian noise
        SNRdB = 60;                                     %Define SNR
        muw = norm(z)^2/M*10^(-SNRdB/10);               %Calculate noise variance
        gOut = AwgnEstimOut(z,muw);                     %Output channel                                                   
    case 'AWBGN' %Additive white Bernoulli-Gaussian noise
        SNRdB = 10;                                     %Define SNR
        del = 0.4;                                      %Define ratio of noisy measurements to total
        P = ceil(del*N);                                %Set number of corrupted measurements P
        muw = norm(z)^2/M*10^(-SNRdB/10)/del;           %Calculate "active" noise variance
        gOut = AwbgnEstimOut(z,muw,del);                %Output channel
    case 'AWLN' %Additive white Laplacian noise
        rate = 1e2;                                     %Laplacian Rate parameter
        gOut = LaplaceEstimOut(z,rate);                 %Output channel
    case 'Dirac' %Noiseless measurement via Dirac Delta
        gOut = DiracEstimOut(z);
    case 'AWGMN' %Additive white Gaussian Mixture noise
        lambda = 0.2;                                   %ratio of large var elements to small var elements
        nu0 = 1e-6;                                     %small variance                             
        nu1 = 1e-2;                                     %large variance
        gOut = GaussMixEstimOut(z,nu0,nu1,lambda);
end
        
%Generate noisy outputs
y = gOut.genRand(z);

%% Define output channel p_{Y|Z}(y|z)

switch pYZtype
    case 'AWGN' %Additive white Gaussian noise
        gOut = AwgnEstimOut(y,muw);                     %Output channel                                                   
    case 'AWBGN' %Additive white Bernoulli-Gaussian noise
        gOut = AwbgnEstimOut(y,muw,del);             %Output channel
    case 'AWLN' %Additive white Laplacian noise
        gOut = LaplaceEstimOut(y,rate);                 %Output channel
    case 'Dirac' %Noiseless measurement via Dirac Delta
        gOut = DiracEstimOut(y);
    case 'AWGMN' %Additive white Gaussian Mixture noise
        gOut = GaussMixEstimOut(y,nu0,nu1,lambda);
end

%% Set GAMP options
optGAMP = GampOpt();
optGAMP.adaptStep = true;       %allow for adaptive stepsize
optGAMP.stepMin = 0.05;         %set minimum stepsize
optGAMP.stepMax = 1;            %set maximum step size
optGAMP.legacyOut = false;      %turn off legacy settings
optGAMP.tol = 1e-8;             %set GAMP tolerance
optGAMP.stepWindow = 0;          %set GAMP step window to 0

%% Run GAMP with Bethe cost
optGAMP.adaptStepBethe = true;  %turn on new cost function
[estFin,~,estHist] = ...
    gampEst(gX, gOut, A, optGAMP);
xhat = estFin.xhat;

%% Run GAMP with log likelihood cost
optGAMP.adaptStepBethe = false;  %turn on new cost function
[estFin2,optFin,estHist2] = ...
    gampEst(gX, gOut, A, optGAMP);
xhat2 = estFin2.xhat;

%% Compute errors and plot information
nmsedBbethe = 10*log10(norm(x-xhat)^2/norm(x)^2)
nmsedBLogLike = 10*log10(norm(x-xhat2)^2/norm(x)^2)

xMat = repmat(x,1,estHist.it(end));
nmseBetheHist = 10*log10(sum((xMat - estHist.xhat).^2)./sum(xMat.^2));

xMat = repmat(x,1,estHist2.it(end));
nmseBetheHist2 = 10*log10(sum((xMat - estHist2.xhat).^2)./sum(xMat.^2));

%Display GAMP outputs
figure(50)
suptitle(strcat('GAMP info under',{' '},pYZtype,' output channel'))
subplot(2,2,2);
% Plot NMSE [dB] over iterations
plot(nmseBetheHist,'r-o','linewidth',2);
hold on
plot(nmseBetheHist2,'b--+','linewidth',2);
xlabel('iterations');
ylabel('NMSE dB');
%Plot negative cost over iterations
subplot(2,2,1);
plot(estHist.val,'r-o','linewidth',2);
hold on
plot(estHist2.val,'b--+','linewidth',2);
legend({'Bethe','LogLike'},'location','SouthEast')
xlabel('iterations');
ylabel('Negative Cost');
subplot(2,2,3)
%plot step size over iterations
plot(estHist.step,'r-o','linewidth',2);
hold on
plot(estHist2.step,'b--+','linewidth',2);
axis([0 max(estHist.it(end),estHist2.it(end)) 0 1.1])
xlabel('iterations');
ylabel('Step Size');
subplot(2,2,4);
%plot step pass over iterations
plot(estHist.pass,'ro','linewidth',2);
hold on
plot(estHist2.pass,'b+','linewidth',2);
axis([0 max(estHist.it(end),estHist2.it(end)) 0 1.1])
xlabel('iterations');
ylabel('Step Pass');

