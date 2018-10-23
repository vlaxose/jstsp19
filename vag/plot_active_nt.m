
clear
clc

SNR_db = 30;
Nt_range = [64:32:128];
Ns=16;
Nr=Ns;
Lr=;
K = 5;
maxDBiters = 5;

mean_Lt_opt = zeros(length(Nt_range), K);
for nt_index = 1:length(Nt_range)
   Nt = Nt_range(nt_index);
   Lt = Nt;
   disp(['Nt: ', num2str(Nt_range(nt_index ))])
   [~, ~, mean_Lt_opt(nt_index, :)] = monte_carlo_sims(SNR_db, Nt, Nr, Lt, Lr, Ns, K, maxDBiters);
end

figure;
p=plot(Nt_range, mean_Lt_opt(:, 5)); hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'Color', 'Black');
% 