%% EM-GM-AMP Algorithm: A succinct overview
%
% <http://ece.osu.edu/~vilaj Jeremy Vila>, Mar. 2012
%
% <http://ece.osu.edu/~vilaj/code EM-GM-AMP> attempts to recover a sparse 
% signal of length N through (possibly) noisy measurements of length M  
% (with perhaps M < N).  We model this through the linear system
%
% $\bf{y} = \bf{Ax} + \bf{w}$.
%
% Refer to  the paper
% <http://ece.osu.edu/~vilaj/papers/EMGMAMP.pdf EM-GM-AMP> by Jeremy Vila 
% Philip Schniter for a detailed description of the algorithm.  This manual 
% will show how to use the EM-GM-AMP MATLAB code and how to intepret the 
% results.
%
%% Generating Data
% The following examples will use randomly generated data.  First, we must
% create a K-sparse signal and a M-by-N mixing matrix.  Lastly, by 
% specifying an SNR, we will create our random measurements.

close all
clear all

warning('This runs a legacy version of EM-GM-AMP!  The current version can be found in ./EMGMAMPnew/')

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

%% Using EM-GM-AMP
% Using the EM-GM-AMP algorithm is extremeley easy. You simply need the
% measurements and mixing matrix.  The algorithm returns signal estimates, 
% the variance on each signal element, and the learned parameters of the 
% assumed Bernoulli-Gaussian signal.  

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time = tic;
[xhat, EMfin] = EMGMAMP(y, Amat, optEM); %Perform EMGMAMP
time = toc(time) %Calculate total time

%plot the estimated Gaussian mixture distribution
figure(1)
plot_GM(EMfin);

%%
% 
% Verify accuracy of signal recovery with NMSE and plots

nmse = 10*log10(norm(x-xhat)^2/norm(x)^2) % Report EMGMAMP's NMSE (in dB)

figure(2)
plot(x,'b+');hold on %Plot true signal
plot(xhat,'ro'); hold off %Plot signal estimates
xlabel('Signal Index'); ylabel('Value')
title('Case 1: Real Bernoulli-Gaussian Signal, Med Sparsity, High SNR')
legend('True signal', 'Est Signal')

%% EM-GM-AMP Optional Inputs
% Additionally, the user may wish to define some additional EM-GM-AMP 
% options if they:
% 
% # want to decrease computation time
% # have some prior knowledge of signal
% 
% To do so, we define some GAMP and EM options.

%%
% (1)Decrease Computation time
optEM.maxEMiter = 10; %Decrease maximum GAMP iterations
optGAMP.nit = 3;
%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time2 = tic;
xhat = EMGMAMP(y, Amat,optEM, optGAMP); %Perform EMGMAMP
time2 = toc(time2) %Calculate total time

nmse2 = 10*log10(norm(x-xhat)^2/norm(x)^2)

clear optEM
clear optGAMP

%%
% (2) Have knowledge of signal prior

%A BG signal has the following Gaussian mixture parameters
optEM.lambda = K/N; %Initialize signal sparsity level
optEM.active_mean = 1; %Initialize active mean
optEM.active_weights = 1;
optEM.active_var = 1; %Initialize active variance
optEM.noise_var = muw; %Initialize noise variance

%Turn of learning of all parameters
optEM.learn_lambda= false;
optEM.learn_mean = false;
optEM.learn_var = false;
optEM.learn_weights = false;
optEM.learn_noisevar = false;

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time4 = tic;
xhat = EMGMAMP(y, Amat,optEM); %Perform EMGMAMP
time4 = toc(time4) %Calculate total time

nmse4 = 10*log10(norm(x-xhat)^2/norm(x)^2)

clear optEM

%% Example 2: Real Bernoulli Signal
% EM-GM-AMP can also handle other signal types. 

N = 1000; %Fix signal size
del = 0.4; %del = M/N
rho = 0.4; %rho = K/M
SNR = 30; %Signal to noise ration (dB)

M = ceil(del*N); %Find appropriate measurement size
K = floor(rho*M); %Find appropriate sparsity
if K == 0
    K = 1; %Ensure at least one active coefficient.
end

% Form A matrix
Params.M = M; Params.N = N; Params.type =1;
Amat = generate_Amat(Params);

%Determine true bits
support = false(N,1);
support(1:K) = true; %which bits are on

x = ones(N,1).*support;
x = x(randperm(N)); %Form Bernoulli signal
support = (x ~= 0); %Find new support

ztrue = Amat*x; %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw)*randn(M,1); %Compute noisy output

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

time5 = tic;
xhat = EMGMAMP(y, Amat, optEM); %Perform EMGMAMP
time5 = toc(time5) %Calculate total time

nmse5 = 10*log10(norm(x-xhat)^2/norm(x)^2)

%plot signal and the estimates
figure(3)
plot(x,'b+'); hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Values')
title('Case 2: Bernoulli Signal, Med Sparsity, High SNR')

%% Example 3: Real Bernoulli-Rademacher Signal
% EM-GM-AMP can also be used on any generic sparse signal.  Here, we
% investigate its performance on a extremely sparse Bernoulli-Rademacher 
% signal with low SNR. 

clear('optEM') %clear previous options
N = 2000; 
del = 0.1; %del = M/N
rho = 0.15; %rho = K/M
SNR = 15;

M = ceil(del*N);
K = floor(rho*M);
if K == 0
    K = 1; %Ensure at least one active coefficient.
end

Params.M = M; Params.N = N; 
Params.type =1; %Form a Gaussian matrix
Amat = generate_Amat(Params);

support = false(N,1);
support(1:K) = true;

x = sign(randn(N,1)).* support;
x = x(randperm(N)); %Compute true signal
support = (x ~= 0); %Compute new support

ztrue = Amat*x; %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw)*randn(M,1); %Compute noisy output

%set heavy_tailed option to false to operate on sparse signal
optEM.heavy_tailed = false;

optEM.L = 3; %Set the number of mixture components (Still default)

time6 = tic;
xhat = EMGMAMP(y, Amat, optEM); %Perform EMGMAMP
time6 = toc(time6)

nmse6 = 10*log10(norm(x-xhat)^2/norm(x)^2) % Report EMGMAMP's NMSE (in dB)

%plot signal and the estimates  (Note: K = expected # of non-zero coeffs)
figure(4)
plot(x,'b+');hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 3: Real Bernoulli-Rademacher Signal, High Sparsity, Low SNR'))
legend('True signal', 'Est Signal')

%% Example 4: Heavy-tailed signal
% Lets try EM-GM-AMP on a heavy tailed signal (non-compressible Student's
% T) with Infinite SNR using a Random Rademacher matrix.

N = 1000;
del = 0.65; %Specify problem size and SNR
SNR = Inf;

M = ceil(del*N);

Params.M = M; Params.N = N; 
Params.type = 2; % Form Rademacher A matrix
Params.realmat = true;
Amat = generate_Amat(Params);

x = random('t',1.5,N,1);

ztrue = Amat*x; %Compute true input vector

muw = norm(ztrue)^2/M*10^(-SNR/10); %Calculate noise level

y = ztrue + sqrt(muw)*(randn(M,1)); %Compute noisy output

time7 = tic;
xhat = EMGMAMP(y, Amat); %Perform EMGMAMP
time7 = toc(time7)

nmse7 = 10*log10(norm(x-xhat)^2/norm(x)^2) % Report EMGMAMP's NMSE (in dB)

%Plot signal and the estimates
figure(5)
plot(x,'b+'); hold on
plot(xhat,'ro'); hold off
xlabel('Signal Index'); ylabel('Value')
title(sprintf('Case 4: Students T, Inf SNR'))
legend('True signal', 'Est Signal')
