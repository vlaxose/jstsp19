% SE analysis of the sparse NL simulation
% ---------------------------------------

% Set parameters from common file
sparseNLParam;

% Flag indicating whether to perform SE analysis on linear or nonlinear
useLin = true;

% Other parameters
savedat = true;   % Flag to save data

% SE options
SEopt.beta = nx/nz;       % measurement ratio nx/nz
SEopt.Avar = nz/nx;       % nz*var(A(i,j))
SEopt.tauxieq = ~useLin;  % forces taur(t) = xir(t)
SEopt.verbose = true;     % displays progress
SEopt.nit = nit;          % number of iterations

% Construct input estimator averaging function for the Gauss-Bernoulli source
nx = 400;
nw = 400;
umax = sqrt(2*log(nx/2));
u = linspace(-umax,umax,nx)';
px1 = exp(-u.^2/2);
px1 = px1/sum(px1);
x1 = xmean1 + sqrt(xvar1)*u;
x = [0; x1];
px = [1-sparseRat; sparseRat*px1];
inAvg = IntEstimInAvg(inputEst,x,px,nw);
[xcov0, taux0] = inAvg.seInit();
xvar0 = [1 -1]*xcov0*[1; -1];

% Construct otuput estimator averaging function
np = 200;
nz = 200;
nzint = 100;
ny = 200;
outAvg = NLEstimOutAvg(wvar, outFn, np, ny, nz, nzint, useLin, outDeriv);

% Run GAMP SE analysis
[mseSE, hist] = gampSE(inAvg, outAvg, SEopt);

% Load empirical values to compare results against
load data/sparseNLSim;
mseMean = 10*log10(median(mseMeth,3)/xvar0);
if (useLin)
    imeth = 1;
else
    imeth = 2;
end

% Plot results
iter = (1:nit)';
plot(iter, mseMean(:,imeth), 's', iter, mseSE, '-');
grid on;

if (savedat)
    if (useLin)
        fn = 'data/sparseNLSE_lin';
    else
        fn = 'data/sparseNLSE_nonlin';
    end
    cmd = sprintf('save %s mseSE hist', fn);
    eval(cmd);
end