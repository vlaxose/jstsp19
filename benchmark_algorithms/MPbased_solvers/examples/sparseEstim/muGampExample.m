%Example demonstrating Matrix Uncertain GAMP

%Clear
clear all
clc


%% Problem Setup

%Specify problem size
N = 256;
M = 100;
K = 20;

%Specify total SNR, including both Matrix and additive uncertainty
SNR = 20;

%Matrix errors are mostly zeros, this determines what fraction are non-zero
onFraction = 0.01; %should be less than 1


%Set options for GAMP
opt = GampOpt; %initialize the options object
opt.nit = 40;
opt.removeMean = false;
opt.tol = 1e-4;
opt.stepTol = 1e-4;
opt.bbStep = true;
opt.adaptStep = true;
opt.stepWindow = 20;

%% Form x

%Determine true bits
truebits = false(N,1);
truebits(1:K) = true; %which bits are on

%Compute true input vector as Bernoulli-Radamacher
x = sign(randn(N,1)) .* truebits;



%% Form A and measured data


%Build the nominal A matrix
Amat = randn(M,N);

%Normalize the columns
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns


%Rename A
Amean = MatrixLinTrans(Amat);


%Determine desired noise level
muDesired = norm(Amat*x)^2/M*10^(-SNR/10);

%Compute Avar
Avar = muDesired / (K + 1);
muw = Avar;

%Make Avar mostly zeros
Avar = (1/ onFraction) .* Avar.*(rand(M,N) < onFraction);


%Compute the matrix errors as Gaussian with element-wise variances given by
%Avar
E = randn(M,N);
E = sqrt(Avar).*E;


%Store the true, unknown to GAMP, A
A = MatrixLinTrans(Amat + E);
ztrue = A.mult(x);


%Compute the measured data
w = sqrt(muw)*(randn(M,1));
y = ztrue + w;

%Compute effective SNR
SNR_effective =...
    20*log10(norm(Amat*x) / norm(E*x + w));


%% Compute GENIE Estimators

%Compute
fhand = @(x) pcgHelper(A,muw/1,truebits,x);
[yada,~] = pcg(fhand,y,1e-10,1e6);
x_mmsegenie = A.multTr(yada);
x_mmsegenie(~truebits) = 0;

%Compute NMSE
NMSE_GENIE = 20*log10(norm(x_mmsegenie - x)/norm(x));

%Compute
fhand = @(x) pcgHelper(Amean,(muw+K*mean(Avar(:)))/1,truebits,x);
[yada,~] = pcg(fhand,y,1e-10,1e6);
x_mmsegenie_Amean = Amean.multTr(yada);
x_mmsegenie_Amean(~truebits) = 0;

%Compute NMSE
NMSE_GENIE_Amean = 20*log10(norm(x_mmsegenie_Amean - x)/norm(x));


%% Establish the input/output channel objects

%Input channel
xmean0 = 0;
xvar0 = 1;
inputEst = AwgnEstimIn(xmean0, xvar0);
inputEst = SparseScaEstim(inputEst,K/N);

%Output channel with corrected noise level for matrix uncertainty
outputEst = AwgnEstimOut(y, muw + K*mean(Avar(:)));

%% Use GAMP

disp('Running GAMP')
%First, use GAMP without matrix uncertainty
Amean.Avar = 0;
tic
[xhat1, ~, ~,~,~,~,~,~, estHist1] =...
    gampEst(inputEst, outputEst, Amean, opt);
t1 = toc;


%% Use MU-GAMP

%Now use GAMP with matrix uncertainty and knowledge of the entry-wise
%variances Avar


%Output channel with power set based on additive noise only
outputEst = AwgnEstimOut(y, muw);

%Now try with MU
disp('Running MU-GAMP')
Amean.Avar = Avar;
tic
[xhat2, ~, ~, ~,~,~,~,~, estHist2] =...
    gampEst(inputEst, outputEst, Amean, opt);
t2 = toc;





%% Show results



%Compute errors
err1 = zeros(length(estHist1.step),1);
for kk = 1:length(err1)
    err1(kk) = 20*log10(norm(estHist1.xhat(:,kk) - x) / norm(x));
end
err2 = zeros(length(estHist2.step),1);
for kk = 1:length(err2)
    err2(kk) = 20*log10(norm(estHist2.xhat(:,kk) - x) / norm(x));
end


%Determine run length
run_length = max([length(err1) length(err2)]) * 1.2;

figure(1)
clf
plot(err1,'b-+')
hold on
plot(err2,'r-x')
plot([0 run_length],NMSE_GENIE*[1 1],'k-')
plot([0 run_length],NMSE_GENIE_Amean*[1 1],'k--')
legend('GAMP','MU-GAMP',...
    'support & A -GENIE','support-GENIE','location','best')
ylabel('NMSE (dB)')
xlabel('iteration')
xlim([0 run_length])
title(['Optimization Results, Effective SNR='...
    num2str(SNR_effective) ' dB'])

figure(2)
clf
plot((x),'ko')
hold on
plot((xhat1),'b+')
plot((xhat2),'rx')
legend('True','GAMP','MU-GAMP',...
    'location','best')
xlabel('index')
title('Signal Estimates')

