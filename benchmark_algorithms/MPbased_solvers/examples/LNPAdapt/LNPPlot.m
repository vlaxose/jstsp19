
load data/LNPSim_beta200;
load data/LNPSE_beta200;

subplot(1,2,1);
xvar0 = sparseRat;
mseMean = 10*log10(median(mseMeth,3)/xvar0);
nit = size(mseMean,1);
iter = (1:nit);
h=plot(iter,mseMean(:,3), 'o-', iter,mseMean(:,1), 's-', iter,mseSE,'-');
grid on;
set(gca,'FontSize',16);
set(h, 'LineWidth', 2);
xlabel('Iter');
ylabel('Median MSE (dB)');
title('(m,n)=(500,1000)');
legend(methStr{3}, methStr{1}, 'SE');
axis([1 20 -27 1]);

load data/LNPSim_beta125;
load data/LNPSE_beta125;

subplot(1,2,2);
xvar0 = sparseRat;
mseMean = 10*log10(median(mseMeth,3)/xvar0);
nit = size(mseMean,1);
iter = (1:nit);
h=plot(iter,mseMean(:,3), 'o-', iter,mseMean(:,1), 's-', iter, mseSE,'-');
grid on;
set(gca,'FontSize',16);
set(h, 'LineWidth', 2);
xlabel('Iter');
title('(m,n)=(800,1000)');
legend(methStr{3}, methStr{1}, 'SE');
axis([1 20 -27 1]);