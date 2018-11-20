% LNPSE:  State evolution analysis of LNP
%
% Problem details are in the file LNPParam.m

% Call parameter file
LNPParam;

% Other parameters
savedat = true;   % Flag to save data

% SE options
SEopt.beta = nx/nz;       % measurement ratio nx/nz
SEopt.Avar = nz/nx;       % nz*var(A(i,j))
SEopt.tauxieq = false;  % forces taur(t) = xir(t)
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
ny = 400;
outAvg = PoissonNLOutAvg(lamTrue,np,ny,nz);

% Run GAMP SE analysis
[mseSE, hist] = gampSE(inAvg, outAvg, SEopt);

% Load empirical values to compare results against
cmd = sprintf('load data/LNPSim_beta%d', round(100*SEopt.beta));
eval(cmd);

% Plot results
iter = (1:nit)';
imeth = 1;
plot(iter, mseMean(:,imeth), 's', iter, mseSE, '-');
grid on;

if (savedat)
    cmd = sprintf('save data/LNPSE_beta%d1 mseSE hist', round(100*SEopt.beta));
    eval(cmd);
end
