load data/connSim_nz300;
nmeth = size(xhatTot,3);
pfa = cell(nmeth,1);
pmd = cell(nmeth,1);
mseAvg = zeros(nmeth,1);

plotStr = {'b-', 'g-', 'r', 'c'};
for imeth = 1:nmeth
    xhat1 = xhatTot(:,:,imeth);
    xhat1 = xhat1(:);
    
    mseAvg(imeth) = 10*log10( sum(abs(xhat1-xTot1).^2) / sum(abs(xTot1).^2));
    [perri, topti, pfai, pmdi] = perrThresh(xTot(:), xhat1);            
    
    pfa{imeth} = pfai;
    pmd{imeth} = pmdi;
end

if (nmeth == 2)
    h = semilogx(pfa{1},pmd{1}, '-', pfa{2},pmd{2}, '-');
else
    h = semilogx(pfa{1},pmd{1}, '-', pfa{2},pmd{2}, '-', pfa{3},pmd{3}, '-');
end
set(gca,'FontSize',16);
set(h, 'LineWidth', 3);
xlabel('Prob. false alarm');
ylabel('Prob. missed detect');
methStrPlot = {'GAMP', 'LMMSE', 'CoSAMP'};
legend(methStrPlot{methTest});
axis([0.001 1 0 0.8]);
grid on;