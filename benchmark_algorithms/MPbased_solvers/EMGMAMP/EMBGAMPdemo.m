% EMBGAMPdemo      This file will provide an illustrative demo of how to run
% EMBGAMP, on data generated according to EMBGAMP's assumed
% signal model, in addition to several mismatched generative models,and 
% interpret the outputs.
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
%if isdir('../../EMGMAMP')
%    addpath('../../EMGMAMP')
%end

warning('This runs a legacy version of EM-BG-AMP!  The current version can be found in ./EMGMAMPnew/')

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

%% Case 1 Real Bernoulli-Gaussian Signal
fprintf('*******************************************************\n');
fprintf('Case 1: Real Bernoulli-Gaussian Signal, Med Sparsity, High SNR \n')

%Specify problem size and SNR
N = 1000;
del = 0.4;
rho = 0.4;
SNR = 30;

%Set M and K
M = ceil(del*N);
K = floor(rho*M);

% Form A matrix
Params.M = M; Params.N = N; Params.type =1; Params.realmat = true;
Amat = generate_Amat(Params);

%Determine true bits
support = false(N,1);
support(1:K) = true; %which bits are on

active_mean = 1;
active_var = 1;
%Compute true input vector
x = (sqrt(active_var)*randn(N,1)+active_mean).* support;
x = x(randperm(N));

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

%Compute true noiseless measurement
ztrue = Amat*x;

%Output channel- Calculate noise level
muw = norm(ztrue)^2/M*10^(-SNR/10);

%Compute noisy output
y = ztrue + sqrt(muw)*(randn(M,1));

if (Params.type == 1) type = 'iid Gaussian'; elseif (Params.type == 2) ...
        type = 'Rademacher'; else type = 'DFT'; end
fprintf('Pre-runtime: SNR is %2.2f dB. A matrix is %s with dimensions %d by %d \n', ...
    SNR,type,M, N)
fprintf('*******************************************************\n');

% -----------------------------------------------------------------
%Perform EMBGAMP
time = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM);
time = toc(time);
% -----------------------------------------------------------------

% Report EMBGAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat)^2/norm(x)^2);
fprintf('EMBGAMP runtime: %g sec\n', time);
fprintf('EMBGAMP NMSE: %g dB\n', nmse);
fprintf('*******************************************************\n');

%plot signal and the estimates  (Note: K = expected # of non-zero coeffs)
figure(1)
clf
plot(x,'b+');hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title('Case 1: Real Bernoulli-Gaussian Signal, Med Sparsity, High SNR')
legend('True signal', 'Est Signal')

figure(2)
clf
[tag1, tag2] = plot_GM(EMfin,'b','-');
xlabel('x');
title('Estimated and True pdfs/pms for Bernoulli-Gaussian Signal')
legend([tag1, tag2], 'Estimated pdf', 'Estimated pmf')
hold off

fprintf('Hit any key to continue to the next case.\n')
pause
%% Case 2 Real Bernoulli-Rademacher
fprintf('\n\n*******************************************************\n');
fprintf('Case 2: Real Bernoulli-Rademacher Signal, High Sparsity, Low SNR \n')

%Specify problem size and SNR
N = 2000;
del = 0.1;
rho = 0.15;
SNR = 15;

%Set M and K
M = ceil(del*N);
K = floor(rho*M);

% Form A matrix
Params.M = M; Params.N = N; Params.type = 2; %Set to Rademacher matrix
Amat = generate_Amat(Params);

%Determine true bits
support = false(N,1);
support(1:K) = true; %which bits are on

%Compute true input vector
x = sign(randn(N,1)).* support;
x = x(randperm(N));

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

%Compute true noiseless measurement
ztrue = Amat*x;

%Output channel- Calculate noise level
muw = norm(ztrue)^2/M*10^(-SNR/10);

%Compute noisy output
y = ztrue + sqrt(muw)*randn(M,1);

%Set GAMP and EM options

if (Params.type == 1) type = 'iid Gaussian'; elseif (Params.type == 2) ...
        type = 'Rademacher'; else type = 'FFT'; end
fprintf('Pre-runtime: SNR is %2.2f dB. A matrix is %s with dimensions %d by %d \n', ...
    SNR,type,M, N)
fprintf('*******************************************************\n');

% -----------------------------------------------------------------
%Perform EMBGAMP
time = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM);
time = toc(time);
% -----------------------------------------------------------------

% Report EMBGAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat)^2/norm(x)^2);
fprintf('EMBGAMP runtime: %g sec\n', time);
fprintf('EMBGAMP NMSE: %g dB\n', nmse);
fprintf('*******************************************************\n');

%plot signal and the estimates  (Note: K = expected # of non-zero coeffs)
figure(1)
clf
plot(x,'b+');hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 2: Real Bernoulli-Rademacher Signal, High Sparsity, Low SNR'))
legend('True signal', 'Est Signal')
axis tight

figure(2)
clf
[tag1, tag2, AX] = plot_GM(EMfin,'b','-');
axes(AX(2));
hold on
tag3 = stem(0,1-K/N,'r','linewidth',2,'LineStyle','--');
stem(-1,K/N/2,'r','linewidth',2,'LineStyle','--'); 
stem(1,K/N/2,'r','linewidth',2,'LineStyle','--');
xlabel('x'); 
title('Estimated and True pdfs/pmfs for Bernoulli-Rademacher Signal')
legend([tag1, tag2, tag3], 'Estimated pdf', 'Estimated pmf', 'True pmf')
hold off

fprintf('Hit any key to continue to the next case.\n')
pause
%% Case 3 Real Bernoulli
fprintf('\n\n*******************************************************\n');
fprintf('Case 3: Real Bernoulli Signal, Low Sparsity, Inf SNR \n')

%Specify problem size and SNR
N = 1000;
del = 0.6;
rho = 1;
SNR = Inf;

%Set M and K
M = ceil(del*N);
K = floor(rho*M);

% Form A matrix
Params.M = M; Params.N = N; Params.type =1; Params.realmat = true;
Amat = generate_Amat(Params);

%Determine true bits
support = false(N,1);
support(1:K) = true; %which bits are on

%Compute true input vector
x = ones(N,1).* support;
x = x(randperm(N));

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

%Compute true noiseless measurements
ztrue = Amat*x;

%Output channel- Calculate noise level
muw = norm(ztrue)^2/M*10^(-SNR/10);

%Compute noisy output
y = ztrue + sqrt(muw)*(randn(M,1));

if (Params.type == 1) type = 'iid Gaussian'; elseif (Params.type == 2) ...
        type = 'iid Rademacher'; else type = 'FFT'; end
fprintf('Pre-runtime: SNR is %2.2f dB. A matrix is %s with dimensions %d by %d \n', ...
    SNR,type,M, N)
fprintf('*******************************************************\n');

% -----------------------------------------------------------------
%Perform EMBGAMP
time = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM);
time = toc(time);
% -----------------------------------------------------------------

% Report EMBGAMP's NMSE (in dB)
nmse = 10*log10(norm(x-xhat)^2/norm(x)^2);
fprintf('EMBGAMP runtime: %g sec\n', time);
fprintf('EMBGAMP NMSE: %g dB\n', nmse);
fprintf('*******************************************************\n');

%plot signal and the estimates  (Note: K = expected # of non-zero coeffs)
figure(1)
clf
plot(x,'b+'); hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 3: Real Bernoulli Signal, Low Sparsity, Inf SNR'))
legend('True signal', 'Est Signal')

figure(2)
clf
[tag1, tag2, AX] = plot_GM(EMfin,'b','-');
axes(AX(2));
hold on
tag3 = stem(0,1-K/N,'r','linewidth',2,'LineStyle','--'); axes(AX(2));
stem(1,K/N,'r','linewidth',2,'LineStyle','--');
xlabel('x');
title('Estimated and True pdfs/pmfs for Bernoulli Signal')
legend([tag1, tag2, tag3], 'Estimated pdf', 'Estimated pmf', 'True pmf',...
    'Location','NorthWest')
axes(AX(2)); ylim([0 1]);
hold off

fprintf('Hit any key to continue to the next case.\n')
pause

%% Case 4 Multiple Measurement Vector Model with Complex fft
% Low Noise High Sparsity
fprintf('\n\n*******************************************************\n');
fprintf('Case 4 Multiple Measurement Vector Model with Complex fft\n')

%Specify problem size and SNR
N = 2^13;
del = 0.5;
rho = 0.2;
SNR = 25;

%Set M and K
M = ceil(del*N);
K = floor(rho*M);

% Form A fft operator
Params.M = M; Params.N = N; Params.type =3;
Amat = generate_Amat(Params);

%Determine true bits
support = false(N,1);
support(1:K) = true; %which bits are on

%Compute true input vector
x1 = (sqrt(1/2)*(randn(N,1)+1i*randn(N,1)) + 1 + 1i).*support;
x2 = sign(randn(N,1)).*support;
x3 = ones(N,1).*support;

%Form MMV inputs
X = [x1 x2 x3];
X = X(randperm(size(X,1)),:);
T = size(X,2);

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;
optEM.noise_dim = 'col';

%Computer true noiseless measurements
Ztrue = Amat.mult(X);

%Output channel- we assume AWGN on each column
muw = sum(abs(Ztrue).^2,1)/M*10^(-SNR/10);

%Compute noisy output
Y = Ztrue + (ones(M,1)*sqrt(muw/2)).*(randn(M,T)+1i*randn(M,T));

if (Params.type == 1) type = 'iid Gaussian'; elseif (Params.type == 2) ...
        type = 'iid Rademacher'; else type = 'FFT'; end
fprintf('Pre-runtime: SNR is %2.2f dB. A matrix is %s with dimensions %d by %d \n', ...
    SNR,type,M, N)
fprintf('*******************************************************\n');

% -----------------------------------------------------------------
%Perform EMBGAMP
time = tic;
[Xhat, EMfin] = EMBGAMP(Y, Amat, optEM);
time = toc(time);
% -----------------------------------------------------------------

fprintf('EMBGAMP runtime: %g sec\n', time);
% Report EMBGAMP's NMSE (in dB)
for i = 1:T
    nmse = 10*log10((norm(Xhat(:,i)-X(:,i))^2)/(norm(X(:,i))^2));
    fprintf('EMBGAMP NMSE of input vector %d: %g dB\n',i, nmse);
end
fprintf('*******************************************************\n');

%plot signal and the estimates of only the first column (complex Gaussian)
figure(1)
clf
subplot(2,1,1)
plot(real(X(:,1)),'b+'); hold on
plot(real(Xhat(:,1)),'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 4: Complex Gaussian (real coefficients) using fft'))
legend('True signal (real)', 'Est Signal (real)')
axis tight
subplot(2,1,2)
plot(imag(X(:,1)),'b+'); hold on
plot(imag(Xhat(:,1)),'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 4: Complex Gaussian (imag coefficients) using fft'))
axis tight
legend('True signal (imag)', 'Est Signal (imag)')

figure(2)
clf
tag1 = plot_GM(EMfin,'b');
hold off
