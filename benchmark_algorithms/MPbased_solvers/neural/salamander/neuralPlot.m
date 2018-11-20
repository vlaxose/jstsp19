% Plots the results in the data file
% Also records the likelihood in the matrix plikeMeth
%
% plikeMeth(imeth,ipt,dat) = likelihood when:
%   method index = imeth, number of training points = ntrainPt(ipt)
%   and dat = 1 if the likelihood is measured on validation data or
%   dat = 2 if measured on training data.

% Plot type
RESP_PLOT = 1;
LIKE_PLOT = 2;
IMAGE_PLOT = 3;
plotType = IMAGE_PLOT;

if (plotType == RESP_PLOT)
    ntrainPt =  [40 60 100]'*1000;
elseif (plotType == LIKE_PLOT)
    ntrainPt = (20:20:100)'*1000;
elseif (plotType == IMAGE_PLOT)
    ntrainPt = 100000;
end

npt =length(ntrainPt);
nmeth = 2;
nmeth1 = nmeth + 1;
plikeTot = zeros(nmeth1,npt);
param = cell(nmeth,npt);


for ipt = 1:npt
    
    % Load points
    ntrain = ntrainPt(ipt);
    cmd = sprintf('load data/neuralSim_ch15_c1_rho10_nt%d', ntrain);
    eval(cmd);
    
    % Store likelihoods
    plikeTot(1:nmeth,ipt) = plikeMeth(:,1);
    
    % Store the parameters
    for imeth = 1:nmeth
        param{imeth,ipt} = paramMeth{imeth}.copy();
    end
    
    % Load likelihoods for non-sparse case
    cmd = sprintf('load data/neuralSim_ch15_c1_nos_nt%d', ntrain);
    eval(cmd);
    
    % Store likelihoods for non-sparse case
    plikeTot(nmeth1,ipt) = plikeMeth(2,1);
end

if (plotType == RESP_PLOT)
    % Plot the results
    iptPlot = [1:3]';
    nplot = length(iptPlot);
    for ipt = 1:length(iptPlot)
        for imeth = 1:nmeth
            subplot(nplot,nmeth,(ipt-1)*nmeth+imeth);
            p = param{imeth,iptPlot(ipt)};
            ndly = size(p.linWt,1);
            tstep = 10;
            t = tstep*(0:ndly-1)';
            plot(t, p.p(1)*p.linWt);
            axis([0 tstep*ndly -0.4 0.2]);
            set(gca,'FontSize',14);
            %set(h, 'LineWidth', 2);
            set(gca,'ytick',[]);
            if (ipt == length(iptPlot))
                xlabel('Delay (ms)');
            else
                set(gca,'xtick',[]);
            end
            if (ipt == 1) && (imeth == 1)
                title('Non-sparse LNP w/ STA');          
            elseif (ipt == 1) && (imeth == 2)
                title('Sparse LNP w/ GAMP');
            end
            if (imeth==1)
                tstr = sprintf('%d s', ntrainPt(ipt)/100);
                h = text(-80,0,tstr);
                set(h,'FontSize',14);
                h = text(-80,-0.1,'Training');
                set(h,'FontSize',14);
            end
            
        end
    end
    
    %print -depsc neuralResp
    
elseif (plotType == LIKE_PLOT)
    
    t = ntrainPt*1e-2;
    h = plot(t, plikeTrain(2,:,1)','o-', ...
        t, plikeTot(1,:)','s-',...
        t, plikeTot(3,:)','d-');
    grid on;
    set(gca,'FontSize',18);
    set(h, 'LineWidth', 3);
    legend('Sparse LNP w/ GAMP', 'Non-sparse LNP w/ STA', 'Non-sparse LNP w/ approx ML');
    xlabel('Train time (sec)');
    ylabel('Cross-valid score');
    axis([200 1000 0.895 0.925]);
    %print -depsc neuralValid
    
elseif (plotType == IMAGE_PLOT)
    
    for imeth = 1:2
        
        nx = 11;
        subplot(2,1,imeth);
        x = param{imeth}.linWt;
        x = reshape(sum(abs(x)),nx,nx);
        imagesc(x);
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        set(gca,'FontSize',24);
        if (imeth == 1)
            title('Non-sparse LNP w/ STA');     

        else
            title('Sparse LNP w/ GAMP');
        end

        %colorbar;
        axis equal;
        axis([1 nx 1 nx]);
        
    end
    
    print -depsc neuralResp2D
    
end

