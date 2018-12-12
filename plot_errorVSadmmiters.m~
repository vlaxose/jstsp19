clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt = 4;
Nr = 32;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 1;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1:10];
snr_range = 5;
subSamplingRatio = 0.75;
maxMCRealizations = 1;
T_range = [20 30 40];
Imax = 50;

%% Variables initialization
% mean_error_svt = zeros(length(T_range), Imax);
mean_error_proposed = zeros(length(T_range), Imax);

for r=1:maxMCRealizations
  disp(['realization: ', num2str(r)]);
%   convergence_error_svt = zeros(length(T_range), Imax);  
  convergence_error_proposed = zeros(length(T_range), Imax);

  
  for sub_indx=1:length(T_range)


    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    [Y_proposed_hbf, Y_conventional_hbf, W_tilde, Psi_bar, Omega, Lr] = wideband_hybBF_comm_system_training(H, T, square_noise_variance, subSamplingRatio, Gr);
    
%     disp('Running proposed technique...');
    tau_X = 1/norm(Y_proposed_hbf, 'fro')^2;
    tau_S = tau_X/2;
    eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
    rho = sqrt(min(eigvalues)*(tau_X+tau_S)/2);
    A = W_tilde'*Dr;
    B = zeros(L*Nt, T);
    for l=1:L
      B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
    end

    [~, ~, Y_proposed_convergence] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho);
    for i
    S_proposed = pinv(A)*Y_proposed*pinv(B);

  end
%   mean_error_svt = mean_error_svt + convergence_error_svt;
  mean_error_proposed = mean_error_proposed + convergence_error_proposed;
end
% mean_error_svt = mean_error_svt/maxMCRealizations;
mean_error_proposed = mean_error_proposed/maxMCRealizations;

%% Plotting
figure;
marker_stepsize = 50;
p_proposed_1 = semilogy(1:Imax,  (mean_error_proposed(1, :)));hold on;
set(p_proposed_1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p_proposed_2 = semilogy(1:Imax,  (mean_error_proposed(2, :)));hold on;
set(p_proposed_2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 8, 'Color', 'Blue');
p_proposed_3 = semilogy(1:Imax,  (mean_error_proposed(3, :)));hold on;
set(p_proposed_3,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'p', 'MarkerSize', 8, 'Color', 'Green');
xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
legend({'T=1','T=2','T=3'}, 'FontSize', 12);
grid on;set(gca,'FontSize',12);

savefig(strcat('results/errorVSadmmiters.fig'))