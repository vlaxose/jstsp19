clear
clc

transmit_snr = -20:10:20;
Nt=64;Lt=32;Ns=8;
Nr=32;
maxDBiters = 5;
K = 6;
mean_rate = zeros(length(transmit_snr), K);
mean_ee = zeros(length(transmit_snr), K);
mean_Lt_opt = zeros(length(transmit_snr), K);
for snr_index = 1:length(transmit_snr)
   disp(['pu: ', num2str(transmit_snr(snr_index ))])
   [mean_rate(snr_index, :), mean_ee(snr_index, :), mean_Lt_opt(snr_index, :)] = monte_carlo_sims(transmit_snr(snr_index), Nt, Nr, Lt, Ns, K, maxDBiters);
end


figure;
subplot(1,2,1);
p=plot(transmit_snr, mean_ee(:, 1)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
p=plot(transmit_snr, mean_ee(:, 2));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p=plot(transmit_snr, mean_ee(:, 3));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p=plot(transmit_snr, mean_ee(:, 6));hold on;
set(p,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p=plot(transmit_snr, mean_ee(:, 4));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p=plot(transmit_snr, mean_ee(:, 5));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Blue');
xlabel('Transmit SNR (dB)', 'FontSize', 14)
ylabel({'Energy Efficiency', '(bits/Joule)'}, 'FontSize', 14)
grid on;

subplot(1,2,2);
p=plot(transmit_snr, mean_rate(:, 1)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
p=plot(transmit_snr, mean_rate(:, 2));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p=plot(transmit_snr, mean_rate(:, 3));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p=plot(transmit_snr, mean_rate(:, 6));hold on;
set(p,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p=plot(transmit_snr, mean_rate(:, 4));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p=plot(transmit_snr, mean_rate(:, 5));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Blue');
xlabel('Transmit SNR (dB)', 'FontSize', 14)
ylabel({'Spectral Efficiency', '(bits/Hz/s)'}, 'FontSize', 14)
grid on;
lg = legend('Digital Beamforming', '1-bit', '8-bit', 'Randomly selected', 'Proposed', 'Brute-force technique', 'Location', 'southoutside', 'Orientation', 'Horizontal');
lg.FontSize = 10;

savefig('./results/result1_ee_rate_snr.fig')
print('./results/result1_ee_rate_snr.eps','-depsc')
