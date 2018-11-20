% Simulation plot
% ----------------
if 0
    % Load empirical values
    load data/sparseNLSim2;
    mseMean = 10*log10(median(mseMeth,3)/xvar0);
    iter = (1:size(mseMean,1))';
    
    % Load SE for non-linear
    load data/sparseNLSE_nonlin;
    mseSENL = mseSE;
    iterNL = (1:size(mseSE,1))';
    
    load data/sparseNLSE_lin;
    mseSELin = mseSE;
    iterLin = (1:size(mseSE,1))';
    
    h = plot(iter, mseMean(:,1), 'bs', iterLin, mseSELin, 'b-', ...
        iter, mseMean(:,2), 'go', iterNL, mseSENL, 'g-');
    grid on;
    set(gca,'FontSize',16);
    set(h, 'LineWidth', 2);
    legend('Lin-GAMP (sim)', 'Lin-GAMP (SE)', 'NL-GAMP (sim)', 'NL-GAMP (SE)');
    xlabel('Iteration');
    ylabel('Normalized squared error (dB)');
    
    
    print -depsc data/sparseNLSim
    
end

% Output function plot
if 1
    sparseNLParam;
    
    % Generate random points
    z = randn(nz,1)*sqrt(zvar0);
    w = randn(nz,1)*sqrt(wvar);
    y = outFn(z) + w;
    
    z0 = linspace(-1,1,100)';
    y0 = outFn(z0);
    
    h = plot(z0,y0,'-', z,y,'.');
    grid on;
    axis([-1 1 -1.5 1.5]);
    set(gca,'FontSize',16);
    set(h, 'LineWidth', 2);
    xlabel('z');
    ylabel('f(z)');
    
    print -depsc data/sparseNLOut
    
    
end



