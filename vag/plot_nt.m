clear
clc

Nt_range = [32:16:96];
SNR_db = 5;
Ns=8;
Nr=32;
maxDBiters = 5;
K = 5;
mean_rate = zeros(length(Nt_range), K);
mean_ee = zeros(length(Nt_range), K);
mean_Lt_opt = zeros(length(Nt_range), K);
for nt_index = 1:length(Nt_range)
   Nt = Nt_range(nt_index);
   Lt = round(2/3*Nt);
   disp(['Nt: ', num2str(Nt_range(nt_index ))])
   [mean_rate(nt_index, :), mean_ee(nt_index, :), mean_Lt_opt(nt_index, :)] = monte_carlo_sims(SNR_db, Nt, Nr, Lt, Ns, K, maxDBiters);
end


figure;
subplot(1,2,1)
p=plot(Nt_range, mean_ee(:, 1)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
p=plot(Nt_range, mean_ee(:, 2));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p=plot(Nt_range, mean_ee(:, 3));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p=plot(Nt_range, mean_ee(:, 4));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p=plot(Nt_range, mean_ee(:, 5));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Blue');
xlabel('Number of TX antennas', 'FontSize', 14)
ylabel({'Energy Efficiency', '(bits/Joule)'}, 'FontSize', 14)

grid on;

subplot(1,2,2)
p=plot(Nt_range, mean_rate(:, 1)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
p=plot(Nt_range, mean_rate(:, 2));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p=plot(Nt_range, mean_rate(:, 3));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p=plot(Nt_range, mean_rate(:, 4));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p=plot(Nt_range, mean_rate(:, 5));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Blue');
xlabel('Number of TX antennas', 'FontSize', 14)
ylabel({'Spectral Efficiency', '(bits/Hz/s)'}, 'FontSize', 14)
grid on;
lg = legend('Digital Beamforming', '1-bit', '8-bit', 'Proposed', 'Brute-force technique', 'Location', 'southoutside', 'Orientation', 'Horizontal');
lg.FontSize = 10;

savefig('./results/result2_ee_rate_nt.fig')
print('./results/result2_ee_rate_nt.eps','-depsc')


