% LNPTest:  Simulation program for the adaptive LNP estimation
%
% Problem details are described in LNPParam.m

% Load parameters
LNPParam;

% Other parameters
savedat = false;   % 1=save data to file
newdat = 1;       % 1=generate new data.  0=keep same data, good for debugging

% GAMP options
opt = GampOpt();
opt.adaptStep = false;  % turn off adaptive step size to match GAMP SE analysis
opt.nit = nit;           % set number of iterations

% Methods to test and number of tests
% 'oracle' = knows true parameters
% 'adapt' = estimates parameters via Gradient ascent
% 'oracle-var' = same as 'adapt', but in the adaptation, an oracle value
%           for the variance is used.
% 'adapt-var' = adaptation where the variance on z is also estimated
%methStr = {'oracle', 'oracle-var', 'adapt' };
methStr = {'oracle', 'adapt' };

% Initialize vectors
nmeth = length(methStr);
xhatMeth = zeros(nx,nmeth,ntest);
xvarMeth = zeros(nx,nmeth,ntest);
mseMeth = zeros(nit, nmeth,ntest);

% Main simulation loop
for itest = 1:ntest
    
    % Generate new random sample, if requested
    if (newdat)
        
        % Create a random sparse vector
        x0 = normrnd(xmean1, sqrt(xvar1),nx,1); % a dense Gaussian vector
        x = x0.*(rand(nx,1) < sparseRat);       % insert zeros
        
        % Create a random measurement matrix
        A = (1/sqrt(nx))*randn(nz,nx);
        z = A*x;
        
        % Generate output
        u = 1./(1+exp(-z));
        v = exp(polyval(lamTrue, u));
        y = poissrnd(v);
    end
    
    
    
    for imeth = 1:nmeth
        % Create output estimator
        outEst = PoissonNLEstim(y, z, lamTrue);
        outEst.setParam('npoly', npoly);
        outEst.setParam('verbose', 1);
        if (nx <= 5000)
            outEst.setParam('nit', 100);
        end                    
        
        % Set parameters of output estimator based on method.
        if (strcmp(methStr{imeth}, 'oracle'))            
            outEst.setParam('lam', lamTrue);
        elseif (strcmp(methStr{imeth}, 'oracle-var'))           
            outEst.setParam('adapt', true, 'oracleAdaptLev', 1);
        elseif (strcmp(methStr{imeth}, 'adapt'))            
            outEst.setParam('adapt', true);       
        else
            error(['Unknown method ', methStr{imeth}]);
        end
        
        % Run the GAMP algorithm
        [xhat, xvar, ~,~,~,~,~,~,hist] = gampEst(inputEst, outEst, A, opt);
        
        % Store results
        xhatMeth(:,imeth) = xhat;
        xvarMeth(:,imeth) = xvar;
        
        % Compute MSE in linear scale
        mseMeth(:,imeth,itest) = mean((hist.xhat-repmat(x,1,opt.nit)).^2)';
        
        % Display the MSE
        mseGAMP = 20*log10( norm(x-xhat)/norm(x));
        fprintf(1,'%d meth=%10s, MSE = %5.1f dB\n', itest, methStr{imeth}, mseGAMP);
                        
    end
    
    if (10*log10( mseMeth(nit,2,itest)/ mseMeth(nit,1,itest)) > 4)
        return;
    end
    if (newdat == 0)
        return;
    end
end

% Plot results
mseMean = 10*log10(median(mseMeth,3)/xvar0);
iter = (1:nit);
h=plot(iter,mseMean);
%h=plot(iter,mseMean(:,1),'-o', iter,mseMean(:,2),'-s');
grid on;
set(gca,'FontSize',16);
set(h, 'LineWidth', 2);
legend(methStr);

% Save data, if requested
if (savedat)    
    cmd = sprintf('save data/LNPSim_beta%d_long mseMeth mseMean methStr xhatMeth xvarMeth nx nz sparseRat',...
        round(100*nx/nz));
    eval(cmd);
end
