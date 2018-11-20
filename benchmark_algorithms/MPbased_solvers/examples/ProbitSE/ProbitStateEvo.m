% ProbitStateEvo
%
% As part of the probit channel state evolution test described in
% ProbitSEParam.m, this script will numerically compute the state evolution
% for the Bernoulli-Gaussian/probit channel system model.
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/12/12
% Change summary: 
%       - Created from sparseNLSE file (12/12/12; JAZ)
% Version 0.2
%

% Call the parameter file to set up the problem
ProbitSEParam;

% SE options
SEopt.beta = N/Mtrain;   	% Measurement ratio
SEopt.Avar = Mtrain/Mtrain; % Mtrain*var(A(i,j))
SEopt.tauxieq = true;       % Forces taur(t) = xir(t)
SEopt.verbose = true;       % Displays progress
SEopt.nit = nit;            % Number of state evolution iterations

% Construct input estimator averaging function for the Gauss-Bernoulli source
Nx = 400;                           % # of discrete integration points
Nw = 400;                           % # of "noise" points
umax = sqrt(2*log(Nx/2));           
u = linspace(-umax, umax, Nx)';
px1 = exp(-u.^2/2);                 % Compute p(x | x ~= 0)...
px1 = px1 / sum(px1);               % ...and normalize 
x1 = BGmean + sqrt(BGvar)*u;        % Discrete non-zero x points
x = [0; x1];                        % Discrete x points
px = [1-sparseRat; sparseRat*px1];  % p(x)
inAvg = IntEstimInAvg(inputEst, x, px, Nw);     % Build SE EstimInAvg object
[xcov0, taux0] = inAvg.seInit();
xvar0 = [1 -1]*xcov0*[1; -1];

% Construct output estimator averaging function for probit channel
Np = 500;       % # of discrete P values
Ny = 2;         % # of discrete Y values (only {0,1} matter)
Nz = 500;       % # of discrete Z values
outAvg = ProbitStateEvoEstimOut(Np, Ny, Nz, 0, probit_var, maxSumVal);

% Run GAMP SE analysis
[mseSE, histSE] = gampSE(inAvg, outAvg, SEopt);

% Load empirical values to compare results against
load(['data/' saveFile '_Emp']);
mseMean = 10*log10(median(mseEmp, 2) / xvar0);

% Plot results
iter = (1:nit)';
clf; plot(iter, mseMean, 's', iter, mseSE, '-');
grid on;
xlabel('GAMP Iteration'); ylabel('MSE');
title(sprintf(['State Evolution  |  N = %d, M_{train} = %d, ' ...
    'K = %d, Probit Var. = %g'], N, Mtrain, K, probit_var))
legend('Empirical MSE', 'State Evo. MSE', 'Location', 'NorthEast')

if (savedat)
    save(['data/' saveFile '_SE'], 'mseSE', 'histSE');
end


%% Calculation of test error rate

% Numerically evaluate error rate integrals...
zt = linspace(-2,2,1e4);
zht = linspace(-2,2,1e4);
dz = zt(2) - zt(1);
Avar = SEopt.Avar / Mtrain;
xcovSE = histSE.xcov;
varZt = N*Avar*xcovSE(1,1,end);
varZht = N*Avar*xcovSE(2,2,end);
varZtZht = N*Avar*xcovSE(1,2,end);
Kz = [varZt, varZtZht; varZtZht, varZht];
err = 0;
wait_hdl = waitbar(0, 'Computing test error rate...');
for i = 1:numel(zht)
    if zht(i) < 0
        err = err + dz^2*normcdf(zt',0,sqrt(probit_var))'*mvnpdf([zt', repmat(zht(i),numel(zt),1)], [0, 0], Kz);
    else
        err = err + dz^2*normcdf(-zt',0,sqrt(probit_var))'*mvnpdf([zt', repmat(zht(i),numel(zt),1)], [0, 0], Kz);
    end
    if mod(i, 10) == 0
        waitbar(i/numel(zht), wait_hdl);
    end
end


% Compare against empirical test error rate?
close(wait_hdl);