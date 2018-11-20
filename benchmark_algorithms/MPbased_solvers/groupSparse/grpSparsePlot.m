% Load data
load data/grpSparseSim_omp;

MEAN_LIN = 1;
MEAN_DB = 2;
plotType = MEAN_LIN;

% Compute mean for each method
nmeasTest = length(nzTest);
nmeth = length(methTest0)+1;
mseMean = zeros(nmeasTest,nmeth);
for imeas = 1:nmeasTest
    
    % Lasso MSE
    if (plotType == MEAN_LIN)
        mseMean(imeas, 1) = 10*log10(mean( 10.^(0.1*mseOptGam(:,imeas))));
    else
        mseMean(imeas, 1) = mean( mseOptGam(:,imeas));
    end
    
    % GAMP and LS
    for imeth = 2:nmeth
        if (plotType == MEAN_LIN)
            mseMean(imeas, imeth) = 10*log10(mean( 10.^(0.1*mseMethTot{imeas}(:,imeth-1))));
        else
            mseMean(imeas, imeth) = mean( mseMethTot{imeas}(:,imeth-1));
        end
    end
    
end

xpow = -10;
h = plot(nzTest, mseMean(:,[2 1 4 3])-xpow, 's-');
grid on;     
set(gca,'FontSize',16);
set(h, 'LineWidth', 2);
xlabel('Num measurements (m)');
ylabel('Normalized MSE (dB)');
axis([50 200 -35 5]);
legend('LMMSE', 'Grp LASSO', 'Grp OMP', 'Hybrid-GAMP','Location', 'SouthWest');

print -depsc grpSparseSim
