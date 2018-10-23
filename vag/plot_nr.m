clear
clc

Nt=64;
Lt = 32;
user_nr_range = [8:8:32];
Ns=8;
nr_range = user_nr_range;
SNR_db = 5;
maxDBiters = 5;
K = 5;
mean_rate = zeros(length(nr_range), K);
mean_ee = zeros(length(nr_range), K);
mean_Lt_opt = zeros(length(nr_range), K);
for nr_index = 1:length(nr_range)
   Nr = nr_range(nr_index);
   disp(['Nr: ', num2str(nr_range(nr_index ))])
   [mean_rate(nr_index, :), mean_ee(nr_index, :), mean_Lt_opt(nr_index, :)] = monte_carlo_sims(SNR_db, Nt, Nr, Lt, Ns, K, maxDBiters);
end


figure;
subplot(1,2,1)
p=plot(user_nr_range, mean_ee(:, 1)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
p=plot(user_nr_range, mean_ee(:, 2));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p=plot(user_nr_range, mean_ee(:, 3));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p=plot(user_nr_range, mean_ee(:, 4));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p=plot(user_nr_range, mean_ee(:, 5));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Blue');
xlabel('Number of antennas per user', 'FontSize', 14)
ylabel({'Energy Efficiency', '(bits/Joule)'}, 'FontSize', 14)
grid on;

subplot(1,2,2)
p=plot(user_nr_range, mean_rate(:, 1)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
p=plot(user_nr_range, mean_rate(:, 2));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p=plot(user_nr_range, mean_rate(:, 3));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p=plot(user_nr_range, mean_rate(:, 4));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p=plot(user_nr_range, mean_rate(:, 5));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Blue');
xlabel('Number of antennas per user', 'FontSize', 14)
ylabel({'Spectral Efficiency', '(bits/Hz/sec)'}, 'FontSize', 14)
grid on;
lg = legend('Digital Beamforming', '1-bit', '8-bit', 'Proposed', 'Brute-force technique', 'Location', 'southoutside', 'Orientation', 'Horizontal');lg.FontSize = 10;

savefig('./results/result3_ee_rate_nr.fig')
print('./results/result3_ee_rate_nr.eps','-depsc')

