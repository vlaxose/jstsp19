if 1
    % Final correlation as a function of the SNR
    % ==========================================
    
    % Get SVD estimate
    load data/rankOneTest_svd
    snrMaxSing = snrTest;
    nmeth = size(vcorrSimTot,2);
    nsnr = size(vcorrSimTot,3);
    vcorrMaxSing = reshape(mean(vcorrSimTot,1),nmeth,nsnr)';
    
    load data/rankOneSE;
    snrSE = snrTest;
    vcorrSE = corrv(end,:);
    
    load data/rankOneTest_iterFac;
    nmeth = size(vcorrSimTot,2);
    nsnr = size(vcorrSimTot,3);
    vcorrSim = reshape(median(vcorrSimTot,1),nmeth,nsnr)';
    vcorrPred = reshape(median(vcorrPredTot,1),nmeth,nsnr)';
    
    h = plot( snrTest, vcorrSim(:,2), 'rs', snrSE, vcorrSE, 'r-', ...%snrTest, vcorrPred(:,2), 'r-', ...
        snrMaxSing, vcorrMaxSing, 'bo', snrTest, vcorrSim(:,1), 'b+', ...
        snrTest, vcorrPred(:,1), 'b-' );
    grid on;
    set(gca,'FontSize',16);
    set(h, 'LineWidth', 2);
    
    legend('IterFac-mmse (sim)', 'IterFac-mmse (pred)', ...
        'MaxSing', 'IterFac-lin (sim)', 'IterFac-lin (pred)', 'Location', 'SouthEast');
    xlabel('Scaled SNR (dB)');
    ylabel('Correlation \rho_v')
    
    print -depsc data/rankOneSim
    
end

if 0
    
    % Correlation as a function of the iteration number
    % =================================================
    
    load data/rankOneTest_iter;
    
    snrPlot = [2 3]';
    nplot = length(snrPlot);
    
    for iplot = 1:nplot
        % Select the subplot
        subplot(1, nplot, iplot);

        isnr = snrPlot(iplot);
        vcorrIterAvg = median(vcorrIterTot(:,:,:,isnr), 3);
        nit = size(vcorrIterAvg,1);
        iter = [1:nit]';
        h = plot(iter, vcorrIterAvg(:,4), 'rs', iter, vcorrIterAvg(:,3), 'r-',...
            iter, vcorrIterAvg(:,2), 'b+', iter, vcorrIterAvg(:,1), 'b-');
        
        grid on;
        set(gca,'FontSize',16);
        set(h, 'LineWidth', 2);
        
        if (iplot == nplot)
            legend('IF-mmse (sim)', 'IF-mmse (pred)', ...
                'IF-lin (sim)', 'IF-lin (pred)', 'Location', 'SouthEast');
        end
        xlabel('Iteration');
        if (iplot == 1)
            ylabel('Correlation \rho_v')
        end
        titleStr = sprintf('SNR=%d dB', snrTest(isnr));
        title(titleStr);
        axis([1 nit 0 1]);
        
        print -depsc data/rankOneIter
    end
    
end

