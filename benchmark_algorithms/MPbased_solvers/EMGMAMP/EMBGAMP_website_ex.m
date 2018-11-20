%% EM-BG-AMP Algorithm: A succinct overview
%
% <http://ece.osu.edu/~vilaj Jeremy Vila>, Nov 2011
%
% <http://ece.osu.edu/~vilaj/code EM-BG-AMP> attempts to recover a sparse 
% signal of length N through (possibly) noisy measurements of length M  
% (with perhaps M < N).  We model this through the linear system
%
% $\bf{y} = \bf{Ax} + \bf{w}$.
%
% Refer to  the paper
% <http://ece.osu.edu/~vilaj/papers/EMBGAMP.pdf EM-BG-AMP> by Jeremy Vila 
% Philip Schniter for a detailed description of the algorithm.  This manual 
% will show how to use the EM-BG-AMP MATLAB code and how to intepret the 
% results.
%
%% Generating Data
% The following examples will use randomly generated data.  First, we must
% create a K-sparse signal and a M-by-N mixing matrix.  Lastly, by 
% specifying an SNR, we will create our random measurements.

clear all
close all

warning('This runs a legacy version of EM-BG-AMP!  The current version can be found in ./EMGMAMPnew/')

N = 1000; %Fix signal size
del = 0.4; %del = M/N
rho = 0.4; %rho = K/M
SNR = 30; %Signal to noise ration (dB)

M = ceil(del*N); %Find appropriate measurement size
K = floor(rho*M); %Find appropriate sparsity
if K == 0
    K = 1; %Ensure at least one active coefficient.
end

Params.M = M; 
Params.N = N; 
Params.type = 1; %Specify i.i.d. Gaussian Matrix
Params.realmat = true; %Specify Real Matrix
Amat = generate_Amat(Params); %Generate random matrix

support = false(N,1); %Initialize support vector.
support(1:K) = true; %Which bits are on

active_mean = 1; active_var = 1;
x = (sqrt(active_var)*randn(N,1)+active_mean).* support;
x = x(randperm(N)); %Form sparse Bernoulli-Gaussian signal with the mean
% and variance of Gaussian component = 1.
support = (x ~= 0); %Calculate new support

ztrue = Amat*x; %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw)*(randn(M,1)); %Compute noisy output

%% Using EM-BG-AMP
% Using the EM-BG-AMP algorithm is extremeley easy. You simply need the
% measurements and mixing matrix.  The algorithm returns signal estimates, 
% the variance on each signal element, and the learned parameters of the 
% assumed Bernoulli-Gaussian signal.  

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM); %Perform EMBGAMP
time = toc(time) %Calculate total time

%%
% 
% Verify accuracy of signal recovery with NMSE and plots


nmse = 10*log10(norm(x-xhat)^2/norm(x)^2) % Report EMBGAMP's NMSE (in dB)

figure(1)
plot(x,'b+');hold on %Plot true signal
plot(xhat,'ro'); hold off %Plot signal estimates
xlabel('Signal Index'); ylabel('Value')
title('Case 1: Real Bernoulli-Gaussian Signal, Med Sparsity, High SNR')
legend('True signal', 'Est Signal')

%% EM-BG-AMP Optional Inputs
% Additionally, the user may wish to define some additional EM-BG-AMP 
% options if they:
% 
% # want to decrease computation time
% # have some prior knowledge of signal
% 
% To do so, we define some GAMP and EM options.

%%
% (1)Decrease Computation time

optGAMP.nit = 3; %Decrease maximum GAMP iterations
optEM.maxEMiter = 10; %Decrease maximum GAMP iterations
%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time2 = tic;
[xhat, xvar, param] = EMBGAMP(y, Amat,optEM, optGAMP); %Perform EMBGAMP
time2 = toc(time2) %Calculate total time

nmse2 = 10*log10(norm(x-xhat)^2/norm(x)^2)

%%
% (2) Have knowledge of signal prior

optEM.lambda = K/N; %Initialize signal sparsity level
optEM.active_mean = 1; %Initialize active mean
optEM.active_var = 1; %Initialize active variance
optEM.noise_var = muw; %Initialize noise variance
%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time4 = tic;
[xhat, xvar, param] = EMBGAMP(y, Amat,optEM); %Perform EMBGAMP
time4 = toc(time4) %Calculate total time

nmse4 = 10*log10(norm(x-xhat)^2/norm(x)^2)

clear optEM

%% Example 2: Complex Bernoulli Gaussian Signal
% EM-BG-AMP can also handle complex models. Let's perform EM-BG-AMP on
% complex measurements.  

N = 1000; %Fix signal size
del = 0.4; %del = M/N
rho = 0.4; %rho = K/M
SNR = 30; %Signal to noise ration (dB)

M = ceil(del*N); %Find appropriate measurement size
K = floor(rho*M); %Find appropriate sparsity
if K == 0
    K = 1; %Ensure at least one active coefficient.
end

% Form A matrix set matrix to complex Gaussian type
Params.M = M; Params.N = N; Params.type =1; Params.realmat = false;
Amat = generate_Amat(Params);

%Determine true bits
support = false(N,1);
support(1:K) = true; %which bits are on

active_mean = 1+1i; active_var = 1;
x = (sqrt(active_var/2)*(randn(N,1)+1i*randn(N,1))+active_mean).* support;
x = x(randperm(N)); %Form sparse complex Bernoulli-Gaussian signal with the 
% mean = 1 + 1i and variance of Gaussian component = 1.
support = (x ~= 0); %Find new support

ztrue = Amat*x; %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw/2)*(randn(M,1)+1i*randn(M,1)); %Compute noisy output

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time5 = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM); %Perform EMBGAMP
time5 = toc(time5) %Calculate total time

nmse5 = 10*log10(norm(x-xhat)^2/norm(x)^2)

%plot signal and the estimates
figure(2)
subplot(2,1,1)
plot(real(x),'b+'); hold on
plot(real(xhat),'ro'); hold off
xlabel('Signal Index'); ylabel('Real Values')
title('Case 2: Complex Bernoulli-Gaussian Signal, Med Sparsity, High SNR')
legend('True signal (Real)', 'Est Signal (Real)')
subplot(2,1,2)
plot(imag(x),'b+'); hold on
plot(imag(xhat),'ro'); hold off
xlabel('Signal Index'); ylabel('Imaginary Values')
legend('True signal (Imag)', 'Est Signal (Imag)')

%% Example 3: Real Bernoulli-Rademacher Signal
% EM-BG-AMP can also be used on any generic sparse signal.  Here, we
% investigate its performance on a extremely sparse Bernoulli-Rademacher 
% signal with low SNR. We also use the oversampled DFT operator.

N = 2^15; %Ensure that N is a power of 2
del = 0.1; %del = M/N
rho = 0.15; %rho = K/M
SNR = 15;

M = ceil(del*N);
K = floor(rho*M);
if K == 0
    K = 1; %Ensure at least one active coefficient.
end

Params.M = M; Params.N = N; 
Params.type =3; %Form a DFT operator with random "rows"
Params.realmat = true;
Amat = generate_Amat(Params);

support = false(N,1);
support(1:K) = true;

x = sign(randn(N,1)).* support;
x = x(randperm(N)); %Compute true signal
support = (x ~= 0); %Compute new support

ztrue = Amat.mult(x); %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw/2)*(randn(M,1)+1i*randn(M,1)); %Compute noisy output

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time6 = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM); %Perform EMBGAMP
time6 = toc(time6)

nmse6 = 10*log10(norm(x-xhat)^2/norm(x)^2) % Report EMBGAMP's NMSE (in dB)

%plot signal and the estimates  (Note: K = expected # of non-zero coeffs)
figure(3)
plot(x,'b+');hold on
plot(real(xhat),'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 3: Real Bernoulli-Rademacher Signal, High Sparsity, Low SNR'))
legend('True signal', 'Est Signal')

%% Example 4: Real Bernoulli Signal
% Lets try EM-BG-AMP on a Bernoulli signal with Infinite SNR and M=K
% (Low sparsity level) using a Rademacher matrix

N = 1000;
del = 0.65; %Specify problem size and SNR
rho = 1; 
SNR = Inf;

M = ceil(del*N);
K = floor(rho*M);
if K == 0
    K = 1; %Ensure at least one active coefficient.
end

Params.M = M; Params.N = N; 
Params.type =2; % Form Rademacher A matrix
Params.realmat = true;
Amat = generate_Amat(Params);

support = false(N,1);
support(1:K) = true; 

x = ones(N,1).* support;
x = x(randperm(N)); %Compute true signal
support = (x ~= 0);

ztrue = Amat*x; %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw)*(randn(M,1)); %Compute noisy output

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time7 = tic;
[xhat, EMfin] = EMBGAMP(y, Amat, optEM); %Perform EMBGAMP
time7 = toc(time7)

nmse7 = 10*log10(norm(x-xhat)^2/norm(x)^2) % Report EMBGAMP's NMSE (in dB)

%Plot signal and the estimates
figure(4)
plot(x,'b+'); hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 4: Real Bernoulli Signal, Low Sparsity, Inf SNR'))
legend('True signal', 'Est Signal')
