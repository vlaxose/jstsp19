%% generate problem
% problem size
% clear;
clear all;
% s = RandStream('swb2712','Seed',1000);
% RandStream.setDefaultStream(s);
rows = 1000;
cols = 1000;

maxIter = 40;

IMPULSIVE = 0; % 0 for Gaussian and 1 for impulsive Sparse matrices
sparseFac = 0.1;
rB        = 20; % rank of Low-Rank matrix

% Gaussian noise parameter
noiseFac   = 1 * 1e-8;
%==================================================================

% Low-Rank matrix
Y =  randn(rows,rB) * randn(rB,cols);
Y_noisefree = Y;

% Add Gaussian noise
noise = noiseFac*max(Y(:))*randn(size(Y));
Y = Y + noise;

% Add sparse signals
strength_level = max(abs(Y_noisefree(:)));
S = zeros(rows,cols);

% Sparse matrix
p = randperm(rows*cols);
L = round(sparseFac*rows*cols);
if IMPULSIVE,
    S(p(1:L)) = strength_level*sign(randn(L,1));
else
    S(p(1:L)) = strength_level*randn(L,1);
end
clear p;
Y = Y + S;


% Prob1;
lambda = 1/sqrt(rows);
t_start = tic;
[Y_hat2 S_hat2 iter] = inexact_alm_rpca(Y, lambda, -1, maxIter);
toc(t_start);


errSP = norm(S_hat2 - S, 'fro') / (0 + norm(S,'fro'));
errLR = norm(Y_hat2 - Y_noisefree, 'fro') / (0 + norm(Y_noisefree,'fro'));
errTotal = norm([S_hat2,Y_hat2] - [S,Y_noisefree],'fro') / (0 + norm([S,Y_noisefree],'fro'));


fprintf('Matrix size %d x %d, Rank %d, Sparse Ratio %4.1f%%\n',rows,cols,rB,sparseFac*100);
fprintf('Iter: %d, Sparse Error: %4.2e, Low-Rank Error: %4.2e, Total Error: %4.2e\n',iter,errSP,errLR,errTotal);




