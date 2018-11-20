% ProbitEmpirical
%
% As part of the probit channel state evolution test described in
% ProbitSEParam.m, this script will repeatedly run GAMP on independent
% synthetically generated problems matched to the specified model
% parameters in order to obtain a Monte Carlo ensemble average performance
% characterization of GAMP.
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 12/11/12
% Change summary: 
%       - Created from sparseNL file (12/11/12; JAZ)
% Version 0.2
%

% Call the parameter file to set up the problem
ProbitSEParam;

% GAMP options
GAMPopt = GampOpt();
GAMPopt.adaptStep = false;  	% Turn off adaptive step-size to match GAMP SE analysis
GAMPopt.varNorm = false;        % Don't normalize variance messages
GAMPopt.uniformVariance = false;% Use vector message variances
GAMPopt.nit = nit;              % Set # of GAMP iterations
GAMPopt.tol = -1;           	% No early termination

% Initialize storage arrays for empirical results
xhatEmp = NaN(N,ntest);
xvarEmp = NaN(N,ntest);
zhatEmp = NaN(N,ntest);
zvarEmp = NaN(N,ntest);
mseEmp = NaN(nit,ntest);

% Empirical state evolution-related values
tauxEmp = zeros(nit,ntest);
taupEmp = zeros(nit,ntest);
xirEmp = zeros(nit,ntest);
taurEmp = zeros(nit,ntest);
alpharEmp = zeros(nit,ntest);
zpcovEmp = zeros(nit,3,ntest);

% Main Monte Carlo loop
for itest = 1:ntest
    
    % Create a random Bernoulli-Gaussian vector
    x0 = sqrt(BGvar)*randn(N,1) + BGmean; 	% a dense Gaussian vector
    x = x0.*(rand(N,1) < sparseRat);        % insert zeros
    
    % Create a zero-mean random normal training feature matrix
    A_train = (1/sqrt(Mtrain))*randn(Mtrain,N); 	% Feature matrix
    z = A_train*x;      % "True" transform vector
    
    % Compute training class labels based on synthetic "true" hyperplane
%     y_train = double(z > 0);
    y_train = double(normcdf(z, 0, sqrt(probit_var)) > rand(Mtrain,1));
    
    % Create the EstimOut class object for the probit channel (MMSE)
    outEst = ProbitEstimOut(y_train, 0, probit_var, maxSumVal);
    
    % Run the GAMP algorithm
    [xhat, xvar, ~,~,~,~,~,~,histEmp] = gampEst(inputEst, outEst, A_train, ...
        GAMPopt);
    
    % Store results
    xhatEmp(:,itest) = xhat;
    xvarEmp(:,itest) = xvar;
    
    % Compute MSE in linear scale
    mseEmp(:,itest) = mean((histEmp.xhat - repmat(x,1,nit)).^2);
    
    % Display the MSE
    mseGAMP = 20*log10(norm(x - xhat)/norm(x));
    fprintf(1,'Iteration: %d  |  MSE: %5.1f dB\n', itest, mseGAMP);
    
    % Compute empirical SE quantities
    tauxEmp(:,itest) = mean(histEmp.xvar)';
    taupEmp(:,itest) = mean(histEmp.pvar)';
    taurEmp(:,itest) = mean(histEmp.rvar)';
    alphart = histEmp.rhat'*x/(x'*x);
    alpharEmp(:,itest) = alphart;
    xirEmp(:,itest) = mean(abs(histEmp.rhat - x*alphart').^2)';
    for it = 1:GAMPopt.nit
        zpcovEmp(it,1,itest) = (z'*z)/Mtrain;
        zpcovEmp(it,2,itest) = (histEmp.phat(:,it)'*z)/Mtrain;
        zpcovEmp(it,3,itest) = (histEmp.phat(:,it)'*histEmp.phat(:,it))/Mtrain;
    end
    
end

% Store history
histEmp.taux = tauxEmp;
histEmp.taup = taupEmp;
histEmp.xir = xirEmp;
histEmp.taur = taurEmp;
histEmp.alphar = alpharEmp;
histEmp.zpcov = zpcovEmp;

% Plot the mean MSE
mseMean = 10*log10(mean(mseEmp, 2));
iter = (1:GAMPopt.nit);
hdl = plot(iter,mseMean,'-o');
grid on;
set(gca,'FontSize',16);
set(hdl, 'LineWidth', 2);
title('GAMP Bernoulli-Gaussian/Probit Channel Empirical MSE Plot')
xlabel('GAMP Iteration'); ylabel('Ensemble-Averaged MSE (dB)')

if (savedat)
    save(['data/' saveFile '_Emp'], 'mseEmp', 'mseMean', 'N', 'Mtrain', ...
        'sparseRat', 'histEmp');
end