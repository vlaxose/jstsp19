% sparseNL:  GAMP simulation for sparse nonlinear estimation
%
% The problem is described in the parmater file, sparseNLParam.

% Call the parameter file to set up the problem
sparseNLParam;

% Other parameters
ntest = 100;            % number of Monte Carlo tests
savedat = true;         % 1=save data to file

% GAMP options
opt = GampOpt();
opt.adaptStep = false;  % turn off adaptive step size to match GAMP SE analysis
opt.nit = nit;          % set number of iterations
opt.tol = 0;            % prevent from terminating early

% Methods to test and number of tests
methStr = {'lin', 'nonlin'};
%methStr = {'lin'};

% Initialize vectors
nmeth = length(methStr);
xhatMeth = zeros(nx,nmeth,ntest);
xvarMeth = zeros(nx,nmeth,ntest);
mseMeth = zeros(opt.nit, nmeth,ntest);

% Measured SE values
tauxMeth = zeros(nit,nmeth,ntest);
taupMeth = zeros(nit,nmeth,ntest);
xirMeth = zeros(nit,nmeth,ntest);
taurMeth = zeros(nit,nmeth,ntest);
alpharMeth = zeros(nit,nmeth,ntest);
zpcovMeth = zeros(nit,3,nmeth,ntest);

% Main simulation loop
for itest = 1:ntest
    
    % Create a random sparse vector
    x0 = normrnd(xmean1, sqrt(xvar1),nx,1); % a dense Gaussian vector
    x = x0.*(rand(nx,1) < sparseRat);       % insert zeros
    
    % Create a random measurement matrix
    A = (1/sqrt(nx))*randn(nz,nx);
    z = A*x;
    
    % Compute the noise level based on the specified peak SNR.
    wvar = 10^(-0.1*snr);
    w = normrnd(0, sqrt(wvar), nz, 1);
    y = outFn(z) + w;
    
    % Generate input estimation class
    inputEst0 = AwgnEstimIn(xmean1, xvar1);
    inputEst = SparseScaEstim( inputEst0, sparseRat );
    
    % Loop over methods
    for imeth = 1:nmeth
        if (strcmp(methStr{imeth}, 'lin'))       
            % Construct linear estimator based on linearization around z=0
            outEst = AwgnEstimOut(y, wvar, [], outDeriv);
        elseif (strcmp(methStr{imeth}, 'nonlin'))  
            
             % Nonlinear estimation class:  Use the NL estimator
            outEst = NLEstimOut(y,wvar,outFn);
        else
            error('Unknown method');
        end
  
        % Run the GAMP algorithm
        [xhat, xvar, ~,~,~,~,~,~,hist] = gampEst(inputEst, outEst, A, opt);
        
        % Store results
        xhatMeth(:,imeth) = xhat;
        xvarMeth(:,imeth) = xvar;
        
        % Compute MSE in linear scale
        mseMeth(:,imeth,itest) = mean((hist.xhat-repmat(x,1,opt.nit)).^2);
        
        % Display the MSE
        mseGAMP = 20*log10( norm(x-xhat)/norm(x));
        fprintf(1,'%d meth=%7s, MSE = %5.1f dB\n', itest, methStr{imeth}, mseGAMP); 
        
        % Compute empirical SE quantities
        tauxMeth(:,imeth,itest) = mean(hist.xvar)';
        taupMeth(:,imeth,itest) = mean(hist.pvar)';
        taurMeth(:,imeth,itest) = mean(hist.rvar)';
        alphart = hist.rhat'*x/(x'*x);
        alpharMeth(:,imeth,itest) = alphart;
        xirMeth(:,imeth,itest) = mean(abs(hist.rhat - x*alphart').^2)';
        for it = 1:opt.nit
            zpcovMeth(it,1,imeth,itest) = (z'*z)/nz;
            zpcovMeth(it,2,imeth,itest) = (hist.phat(:,it)'*z)/nz;
            zpcovMeth(it,3,imeth,itest) = (hist.phat(:,it)'*hist.phat(:,it))/nz;
        end
                        
    end
    
end

% Store history
hist.tauxMeth = tauxMeth;
hist.taupMeth = taupMeth;
hist.xirMeth = xirMeth;
hist.taurMeth = taurMeth;
hist.alpharMeth = alpharMeth;
hist.zpcovMeth = zpcovMeth;

% Plot the mean MSE
mseMean = 10*log10(median(mseMeth,3)/xvar0);
iter = (1:opt.nit);
h=plot(iter,mseMean(:,1),'-o', iter,mseMean(:,2),'-s');
grid on;
set(gca,'FontSize',16);
set(h, 'LineWidth', 2);
legend('Lin-GAMP', 'NL-GAMP');

if (savedat)
    save data/sparseNLSim2 mseMeth mseMean methStr nx nz sparseRat ...
        hist;
end
