% RankOneBatch:  Calls rankOneTest multiple times for simulation

% Parameters
%snrTest = linspace(-5,15,21)';  % SNR values to test
snrTest = [1 3 5]';
ntest = 50;                     % number Monte Carlo tests per SNR point
nmeth = 1;                      % number of methods to tests
nsnr = length(snrTest);         % number of SNR values
runBatch = 1;                   % flag to rankOneTest that it is being run
% batch mode
savedat = 1;                    % flag to save results

% Initialize variables

% Main simulation loop
for isnr = 1:nsnr
    
    snr = snrTest(isnr);
    fprintf(1,'%d SNR = %f\n', isnr, snr);
    
    % Run test
    rankOneTest;
    
    % Initialize matrices on first iteration
    if (isnr == 1)
        ucorrSimTot = zeros(ntest,nmeth,nsnr);
        vcorrSimTot = zeros(ntest,nmeth,nsnr);
        ucorrPredTot = zeros(ntest,nmeth,nsnr);
        vcorrPredTot = zeros(ntest,nmeth,nsnr);
        ucorrIterTot = zeros(nit, 2*nmeth, ntest,nsnr);
        vcorrIterTot = zeros(nit+1, 2*nmeth,ntest,nsnr);
    end
    
    % Save results
    ucorrSimTot(:,:,isnr) = ucorrFinal;
    ucorrPredTot(:,:,isnr) = ucorrPred;
    vcorrSimTot(:,:,isnr) = vcorrFinal;
    vcorrPredTot(:,:,isnr) = vcorrPred;
    ucorrIterTot(:,:,:,isnr) = ucorrIter;
    vcorrIterTot(:,:,:,isnr) = vcorrIter;
end

% Save
if (savedat)
    save data/rankOneTest_iter snrTest methTest ucorrSimTot ...
        ucorrPredTot vcorrSimTot vcorrPredTot ucorrIterTot vcorrIterTot;
end

% Plot results
vcorrSim = reshape(median(vcorrSimTot,1),nmeth,nsnr)';
vcorrPred = reshape(median(vcorrPredTot,1),nmeth,nsnr)';
if (nmeth == 2)
    plot(snrTest, [vcorrSim(:,[1 2 ]) vcorrPred(:,[1 2])]);
else
    plot(snrTest, [vcorrSim(:,[1 2 3]) vcorrPred(:,[2 3])]);
end
grid on;
