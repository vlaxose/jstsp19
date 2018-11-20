% EMNNAMPdemo      This file will provide an illustrative demo of how to run
% EMNNAMP, run on either under non-negative leqast squares AMP (NNLSAMP), 
% expectation maximization non-negative LASSO AMP (EMNNLAMP), and 
% expectation maximization non-negative Gaussian mixture AMP (EMNNGMAMP).
%
% We give two examples: A non-negative signal with M > N noisy
% measurements, and a a noisy compressed sensing case, where the signal is
% drawn from a K-sparse symmetric Dirichlet distribution.  In this case, we
% allow all algorithms to enforce the linear equality constraints
% (sum-to-one) on the signal.  
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 4/4/12
% Change summary: 
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Added MMV example with complex FFT operator.
%
% Version 2.0

clc
clear all;
if isdir('../../EMNNAMP')
    addpath('../../EMNNAMP')
end

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



%% Case 1 Non-negative Uniform Signal M > N noisy linear measurements
fprintf('*******************************************************\n');
fprintf('Case 1: Non-Negative Uniform Signal, Low SNR \n')

%Specify problem size and SNR
M = 2000;
N = 1000;
SNR = 15;

% Form A matrix
%Generate normalized random sensing matrix
Amat = randn(M,N);
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns
A = MatrixLinTrans(Amat);

x = rand(N,1);

%Compute true noiseless measurement
ztrue = A.mult(x);

%Output channel- Calculate noise level
muw = norm(ztrue)^2/M*10^(-SNR/10);

%Compute noisy output
y = ztrue + sqrt(muw)*(randn(M,1));

fprintf('SNR is %2.2f dB. A matrix is Gaussian with dimensions %d by %d \n', ...
    SNR,M, N)
fprintf('*******************************************************\n');

%Perform NNLSAMP
time = tic;
optALG.alg_type = 'NNLSAMP';
xhat_NNLS = EMNNAMP(y, Amat,optALG);
time = toc(time);

% Report NNLSAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat_NNLS)^2/norm(x)^2);
fprintf('NNLSAMP runtime: %g sec\n', time);
fprintf('NNLSAMP NMSE: %g dB\n', nmse);
fprintf('*******************************************************\n');

%Perform EMNNLAMP
time = tic;
optALG.alg_type = 'NNLAMP';
xhat_NNL = EMNNAMP(y, Amat,optALG);
time = toc(time);

% Report EMNNLAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat_NNL)^2/norm(x)^2);
fprintf('EMNNLAMP runtime: %g sec\n', time);
fprintf('EMNNLAMP NMSE: %g dB\n', nmse);
fprintf('*******************************************************\n');

%Perform EMNNGMAMP
time = tic;
optALG.alg_type = 'NNGMAMP';
[xhat_NNGM, stateFin] = EMNNAMP(y, Amat, optALG);
time = toc(time);

% Report EMNNGMAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat_NNGM)^2/norm(x)^2);
fprintf('EMNNGMAMP runtime: %g sec\n', time);
fprintf('EMNNGMAMP NMSE: %g dB\n', nmse);
fprintf('*******************************************************\n');

figure(1)
clf
[tag1, tag2] = plot_NNGM(stateFin,'b','-');
xlabel('x');
title('Estimated apdfs/pmfs for Bernoulli NNGM Signal')
legend([tag1, tag2], 'Estimated pdf', 'Estimated pmf')
hold off

fprintf('Hit any key to continue to the next case.\n')
pause

%% Case 2 Symmetric Dirichlet signal with M < N noisy linear measurements
fprintf('*******************************************************\n');
fprintf('Case 2: Symmetric Dirichlet Signal, High SNR \n')

%Specify problem size and SNR
del = 0.5;
rho = 0.2;

N = 500;
M = ceil(del*N);
K = floor(rho*M);

if K == 0
    K = 1;
end
SNR = 40;

% Form A matrix
%Generate normalized random sensing matrix
Amat = randn(M,N);
columnNorms = sqrt(diag(Amat'*Amat));
Amat = Amat*diag(1 ./ columnNorms); %unit norm columns
A = MatrixLinTrans(Amat);

%Generate signal
x = dirrnd(10*ones(1,K),1)';
x(end+1:N) = zeros(N-K,1);
x = x(randperm(N));

%Compute true noiseless measurement
ztrue = A.mult(x);

%Output channel- Calculate noise level
muw = norm(ztrue)^2/M*10^(-SNR/10);

%Compute noisy output
y = ztrue + sqrt(muw)*(randn(M,1));

fprintf('SNR is %2.2f dB. A matrix is Gaussian with dimensions %d by %d \n', ...
    SNR,M, N)
fprintf('*******************************************************\n');

%Perform NNLSAMP
time = tic;
optALG.alg_type = 'NNLSAMP';
optALG.linEqMat = ones(1,N);
optALG.linEqMeas = 1;
xhat_NNLS = EMNNAMP(y, Amat, optALG);
time = toc(time);

% Report NNLSAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat_NNLS)^2/norm(x)^2);
fprintf('NNLSAMP runtime: %g sec\n', time);
fprintf('NNLSAMP NMSE: %g dB\n', nmse);
disp(['Linear Equality NMSE: ' ...
    num2str(10*log10(norm(1-sum(xhat_NNLS))^2),'%1.5f') ' dB'])
fprintf('*******************************************************\n');

%Perform EMNNLAMP
time = tic;
optALG.alg_type = 'NNLAMP';
optALG.linEqMat = ones(1,N);
optALG.linEqMeas = 1;
xhat_NNL = EMNNAMP(y, Amat, optALG);
time = toc(time);

% Report EMNNLAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat_NNL)^2/norm(x)^2);
fprintf('EMNNLAMP runtime: %g sec\n', time);
fprintf('EMNNLAMP NMSE: %g dB\n', nmse);
disp(['Linear Equality NMSE: ' ...
    num2str(10*log10(norm(ones(1,1)-sum(xhat_NNL))^2/norm(ones(1,1))^2),'%1.5f') ' dB'])
fprintf('*******************************************************\n');

%Perform EMNNGMAMP
time = tic;
optALG.alg_type = 'NNGMAMP';
optALG.linEqMat = ones(1,N);
optALG.linEqMeas = 1;
[xhat_NNGM, stateFin] = EMNNAMP(y, Amat, optALG);
time = toc(time);

% Report EMNNGMAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat_NNGM)^2/norm(x)^2);
fprintf('EMNNGMAMP runtime: %g sec\n', time);
fprintf('EMNNGMAMP NMSE: %g dB\n', nmse);
disp(['Linear Equality NMSE: ' ...
    num2str(10*log10(norm(ones(1,1)-sum(xhat_NNGM))^2/norm(ones(1,1))^2),'%1.5f') ' dB'])
fprintf('*******************************************************\n');

%Plot Recovered NNGM distribution
figure(1)
clf
[tag1, tag2] = plot_NNGM(stateFin,'b','-');
xlabel('x');
title('Estimated apdfs/pmfs for Bernoulli NNGM Signal')
legend([tag1, tag2], 'Estimated pdf', 'Estimated pmf')
hold off
