clear;
clc;

addpath([pwd,'/basic_system_functions']);
addpath(genpath([pwd, '/benchmark_algorithms']));

numOfnz = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameter initialization
Nt = 4;
Nr = 32;
Mr_e = 32;
Mr = 16;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
L = 4;
maxMCRealizations = 20;
T = 10*Nt;
Imax = 100;
snr_db = 15;
square_noise_variance = 10^(-snr_db/10);

%% Variables initialization
mean_error_1 = zeros(Imax, 3);
convergence_error_1 = zeros(Imax, 3);
mean_error_angles_1 = zeros(Imax, 3);
convergence_error_angles_1 = zeros(Imax, 3);

for r=1:maxMCRealizations
    disp(['realization: ', num2str(r)]);
 

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
    Psi_i = zeros(T, T, Nt);
    % Generate the training symbols
    for k=1:Nt
    % Gaussian random symbols
    %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
    end
    W = createBeamformer(Nr, 'ps');
    
   [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);
    
   tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
   tau_Z = 1/norm(Zbar, 'fro')^2/2;
   eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
   rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));

    A = W_tilde'*Dr;
    B = zeros(L*Gt, T);
    for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
    end    
    [~, ~, convergence_error_1] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
  
    [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
    [~, ~, convergence_error_angles_1] = proposed_algorithm_angles(Y_proposed_hbf, Omega, indx_S, A, B, Imax, tau_Y, tau_Z, rho, 'approximate', numOfnz);
    
    
  mean_error_1 = mean_error_1 + convergence_error_1;
  mean_error_angles_1 = mean_error_angles_1 + convergence_error_angles_1;
end
mean_error_1 = mean_error_1/maxMCRealizations;
mean_error_angles_1 = mean_error_angles_1/maxMCRealizations;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameter initialization
Nt = 16;
Nr = 32;
Mr_e = 32;
Mr = 16;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
L = 4;
maxMCRealizations = 20;
T = 10*Nt;
Imax = 100;
snr_db = 15;
square_noise_variance = 10^(-snr_db/10);

%% Variables initialization
mean_error_2 = zeros(Imax, 3);
convergence_error_2 = zeros(Imax, 3);
mean_error_angles_2 = zeros(Imax, 3);
convergence_error_angles_2 = zeros(Imax, 3);

parfor r=1:maxMCRealizations
    disp(['realization: ', num2str(r)]);
 

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
    Psi_i = zeros(T, T, Nt);
    % Generate the training symbols
    for k=1:Nt
    % Gaussian random symbols
    %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
    end
    W = createBeamformer(Nr, 'ps');
    
   [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);
    
   tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
   tau_Z = 1/norm(Zbar, 'fro')^2/2;
   eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
   rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));

    A = W_tilde'*Dr;
    B = zeros(L*Gt, T);
    for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
    end    
    [~, ~, convergence_error_2] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
  
    [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
    
    [~, ~, convergence_error_angles_2]  = proposed_algorithm_angles(Y_proposed_hbf, Omega, indx_S, A, B, Imax, tau_Y, tau_Z, rho, 'approximate', numOfnz);
    
    
  mean_error_2 = mean_error_2 + convergence_error_2;
  mean_error_angles_2 = mean_error_angles_2 + convergence_error_angles_2;
end
mean_error_2 = mean_error_2/maxMCRealizations;
mean_error_angles_2 = mean_error_angles_2/maxMCRealizations;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameter initialization
Nt = 16;
Nr = 32;
Mr_e = 32;
Mr = 16;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
L = 4;
maxMCRealizations = 20;
T = 10*Nt;
Imax = 100;
snr_db = 5;
square_noise_variance = 10^(-snr_db/10);

%% Variables initialization
mean_error_3 = zeros(Imax, 3);
convergence_error_3 = zeros(Imax, 3);
mean_error_angles_3 = zeros(Imax, 3);
convergence_error_angles_3 = zeros(Imax, 3);

parfor r=1:maxMCRealizations
    disp(['realization: ', num2str(r)]);
 

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
    Psi_i = zeros(T, T, Nt);
    % Generate the training symbols
    for k=1:Nt
    % Gaussian random symbols
    %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
    end
    W = createBeamformer(Nr, 'ps');
    
   [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);
    
   tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
   tau_Z = 1/norm(Zbar, 'fro')^2/2;
   eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
   rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));

    A = W_tilde'*Dr;
    B = zeros(L*Gt, T);
    for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
    end    
    [~, ~, convergence_error_3] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
  
    [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
    [~, ~, convergence_error_angles_3]  = proposed_algorithm_angles(Y_proposed_hbf, Omega, indx_S, A, B, Imax, tau_Y, tau_Z, rho, 'approximate', numOfnz);
    
    
  mean_error_3 = mean_error_3 + convergence_error_3;
  mean_error_angles_3 = mean_error_angles_3 + convergence_error_angles_3;
end
mean_error_3 = mean_error_3/maxMCRealizations;
mean_error_angles_3 = mean_error_angles_3/maxMCRealizations;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameter initialization
Nt = 16;
Nr = 32;
Mr_e = 32;
Mr = 16;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
L = 4;
maxMCRealizations = 20;
T = 30*16;
Imax = 100;
snr_db = 5;
square_noise_variance = 10^(-snr_db/10);

%% Variables initialization
mean_error_4 = zeros(Imax, 3);
convergence_error_4 = zeros(Imax, 3);
mean_error_angles_4 = zeros(Imax, 3);
convergence_error_angles_4 = zeros(Imax, 3);

parfor r=1:maxMCRealizations
    disp(['realization: ', num2str(r)]);
 

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
    Psi_i = zeros(T, T, Nt);
    % Generate the training symbols
    for k=1:Nt
    % Gaussian random symbols
    %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
    end
    W = createBeamformer(Nr, 'ps');
    
   [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);
    
   tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
   tau_Z = 1/norm(Zbar, 'fro')^2/2;
   eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
   rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));

    A = W_tilde'*Dr;
    B = zeros(L*Gt, T);
    for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
    end
    [~, ~, convergence_error_4] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
  
    [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
    [~, ~, convergence_error_angles_4]  = proposed_algorithm_angles(Y_proposed_hbf, Omega, indx_S, A, B, Imax, tau_Y, tau_Z, rho, 'approximate', numOfnz);
    
    
  mean_error_4 = mean_error_4 + convergence_error_4;
  mean_error_angles_4 = mean_error_angles_4 + convergence_error_angles_4;
end
mean_error_4 = mean_error_4/maxMCRealizations;
mean_error_angles_4 = mean_error_angles_4/maxMCRealizations;

%% Plotting
marker_stepsize = 10;
figure;subplot(2,2,1)
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_1(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_1(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_angles_1(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_angles_1(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');

xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
grid on;set(gca,'FontSize',12);
title('N_T=4, T=10, SNR=15db')

subplot(2,2,2)
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_2(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_2(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_angles_2(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_angles_2(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');

xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
grid on;set(gca,'FontSize',12);
title('N_T=16, T=10, SNR=15db')

subplot(2,2,3)
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_3(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_3(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_angles_3(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_angles_3(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');

xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
grid on;set(gca,'FontSize',12);
title('N_T=16, T=10, SNR=5db')

subplot(2,2,4)
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_4(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_4(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V1 = semilogy(1:Imax,  10*log10(mean_error_angles_4(:, 1)));hold on;
set(p_proposed_V1,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p_proposed_V2 = semilogy(1:Imax,  10*log10(mean_error_angles_4(:, 3)));hold on;
set(p_proposed_V2,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');

xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
grid on;set(gca,'FontSize',12);
title('N_T=16, T=30, SNR=5db')

legend({'Algorithm 1 - \epsilon_1', 'Algorithm 1 - \epsilon_2', 'Algorithm 2 - \epsilon_1', 'Algorithm 2 - \epsilon_2'}, 'FontSize', 12);

savefig(strcat('results/errorVSadmmiters.fig'))

