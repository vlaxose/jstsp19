%Jeremy Vila
%GM GAMP
%6-5-13

%clc
clear all;
clf

N = 1000;

del = 0.5; %del = M/N
rho = 0.3; %rho = K/M
%Set M and K
M = ceil(del*N);
K = floor(rho*M);

T = 1;

%Avoid zero K
if K == 0
    K = 1;
end

SNRdB = 30;

%Determine true bits
truebits = false(N,T);
truebits(1:K,:) = true; %which bits are on
% 
%Compute true input vector
% BG signal
x = randn(N,T).* truebits;
for t = 1:T
    x(:,t) = x(randperm(N),t);
end

%%%%%%%%%%%%%%%%%%%%%%%
%A matrix, Gaussian
 Amat = randn(M,N);

%Normalize the columns
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns

%Rename A
A = MatrixLinTrans(Amat);

%Get sizes
[M,N] = A.size();

ztrue = A.mult(x);

lambda = 0.1;
nu0 = 0;
%Output channel- we assume AWGN with mean thetahat and variance muw
nu1 = (norm(A.mult(x))^2/M/T*10^(-SNRdB/10) -(1-lambda)*nu0)/lambda;

%Generate noisy y
gOut = GaussMixEstimOut(ztrue,nu0,nu1,lambda);
y = gOut.genRand(ztrue);

%Set GAMP options
optGAMP = GampOpt();
optGAMP.legacyOut = false;
optGAMP.stepMin = 1;
optGAMP.stepMax = 1;
optGAMP.adaptStep = true;
optGAMP.stepWindow = -1;
optGAMP.pvarStep = false;


%% Traditional genie GAMP

gX = AwgnEstimIn(0,1);
gX = SparseScaEstim(gX,K/N);
if lambda ~= 1
    gOut = GaussMixEstimOut(y,nu0,nu1,lambda);
else
    gOut = AwgnEstimOut(y,nu1);
end

optGAMP.adaptStepBethe = false;
[estFin, ~, estHist] = gampEst(gX, gOut, A, optGAMP);
xhat = estFin.xhat;

%% genie GAMP with new cost function

optGAMP.adaptStepBethe = true;
gX = AwgnEstimIn(0,1);
gX = SparseScaEstim(gX,K/N);
gOut2 = GaussMixEstimOut(y,nu0,nu1,lambda);

[estFin2, ~, estHist2] = gampEst(gX, gOut2, A, optGAMP);
xhat2 = estFin2.xhat;

%% 
nmse_loglik = 10*log10((norm(xhat-x)^2)/(norm(x)^2))
nmse_Bethe = 10*log10((norm(xhat2-x)^2)/(norm(x)^2))

plot(estHist.val,'r-','linewidth',1);
hold on
plot(estHist2.val,'b-+','linewidth',1);
set(gca,'FontSize',18)
hold off
axis tight
grid on


list = {'BGAMPloglik','BGAMP Bethe'};
legend(list,'fontsize',15,'Location','SouthEast')
xlabel('iter')
ylabel('cost')
