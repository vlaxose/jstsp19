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
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
subSamplingRatio = 0.75;
maxMCRealizations = 10;
T = 50;
Imax = 40;
snr_db = 5;
square_noise_variance = 10^(-snr_db/10);

%% Variables initialization
mean_error = zeros(Imax, 2);
convergence_error = zeros(Imax, 2);

parfor r=1:maxMCRealizations
    disp(['realization: ', num2str(r)]);
 

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

    [~, ~, convergence_error] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho);
    
  mean_error = mean_error + convergence_error;
end
mean_error = mean_error/maxMCRealizations;

%% Plotting
figure;
marker_stepsize = 10;
p_proposed_V1 = semilogy(1:Imax,  (mean_error(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  (mean_error(:, 2)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');

xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
legend({'\epsilon_1', '\epsilon_2'}, 'FontSize', 12);
grid on;set(gca,'FontSize',12);

savefig(strcat('results/errorVSadmmiters.fig'))
