clc;clear all;
%%
% % % % % % % % % % % % % % % % % % % % % %
% Sample code to run GROUSE: Stochastic Matrix Completion
% % % % % % % % % % % % % % % % % % % % % %

% We have provided a function that implements the matrix completion version
% of GROUSE, a generic subspace estimation algorithm.
%
% This file shows how to set up a matrix completion problem and how to call
% grouse for matrix completion. If you would like code that runs to a
% particular residual error, or code that has diagnostics built in, please
% email Laura Balzano: sunbeam@ece.wisc.edu.
%
% Ben Recht and Laura Balzano, February 2010


%% First we generate the data matrix and incomplete sample vector.

% Number of rows and columns
numr = 6000;
numc = 10000;
% numr = 700;
% numc = 700;
probSize = [numr,numc];
% Rank of the underlying matrix.
truerank = 10;

% Size of vectorized matrix
N = numr*numc;
% Number of samples that we will reveal.
oversample = 6;
M = oversample*truerank*(numr+numc-truerank);

% The left and right factors which make up our true data matrix Y.
YL = randn(numr,truerank);
YR = randn(numc,truerank);

% Select a random set of M entries of Y.
idx = unique(ceil(N*rand(1,(10*M))));
idx = idx(randperm(length(idx)));

[I,J] = ind2sub([numr,numc],idx(1:M));
[J, inxs]=sort(J'); I=I(inxs)';

% Values of Y at the locations indexed by I and J.
S = sum(YL(I,:).*YR(J,:),2);
S_noiseFree = S;

% Add noise.
noiseFac = 1e-6;
noise = noiseFac*max(S)*randn(size(S));
S = S + noise;



%% Now we set parameters for the algorithm.

% We set an upper bound for the rank of the matrix and the number of cycles
% or passes over the data that we want to execute. We also set the gradient
% step size. There is an order of magnitude of step sizes for which the
% algorithm converges to the nosie level.
maxrank = truerank+5;
maxCycles = 10;
step_size = 0.1;

%% % % % % % % % % % % % % % % % % % % % %
% Now run GROUSE.
t0 = clock;

[Usg, Vsg, err_reg] = grouse(I,J,S,numr,numc,maxrank,step_size,maxCycles);

StochTime = etime(clock,t0);

%%
% % % % % % % % % % % % % % % % % % % % % %
% Compute errors.
% % % % % % % % % % % % % % % % % % % % % %


fprintf(1,'\n Results: \n');


% Select a random set of entries of Y
idx2 = unique(ceil(N*rand(1,(10*M))));
idx2 = idx2(randperm(length(idx2)));

ITest = mod(idx2(1:M)-1,numr)'+1;  % row values
JTest = floor((idx2(1:M)-1)/numr)'+1; % column values

% Values of Y at the locations indexed by I and J
STest = sum(YL(ITest,:).*YR(JTest,:),2);


% % % % % % % % % % % % % % % % % % % % % %
% Error for Stochastic gradient

dev = sum(Usg(ITest,:).*Vsg(JTest,:),2) - STest;
RelDevStochG = norm(dev,'fro') / norm(STest,'fro');
fprintf(1,'stoch grad rel.err on test = %7.2e in %4.2f seconds\n', RelDevStochG, StochTime);


