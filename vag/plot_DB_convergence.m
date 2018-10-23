clear
clc

Nt_range = [32 48 64];
Lt = 32;
SNR_db = 30;
Ns=8;
Nr=32;
Lr=Nr;
maxDBiters = 5;
kappa = zeros(length(Nt_range), maxDBiters+1);
K=5;

for nt_index = 1:length(Nt_range)
   Nt = Nt_range(nt_index);
   disp(['Lt: ', num2str(Nt_range(nt_index ))])
   [~,~,~,kappa(nt_index,:)] = monte_carlo_sims(SNR_db, Nt, Nr, Lt, Ns, K, maxDBiters);
end


figure;
p=plot(1:maxDBiters+1, kappa(1, :)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 'o', 'MarkerSize', 8, 'Color', 'Black');
p=plot(1:maxDBiters+1, kappa(2, :));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Black');
p=plot(1:maxDBiters+1, kappa(3, :));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Black');
xlabel('Dinkelbach iteration', 'FontSize', 12)
ylabel('\kappa', 'FontSize', 12)
grid on;

g = legend('N_T=32', 'N_T=48', 'N_T=64', 'Location', 'Best');
lg.FontSize = 11;
savefig('./results/DB_convergence.fig')
print('./results/DB_convergence.eps','-depsc')
