
clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt = 8;
Nr = 32;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
snr_range = 5;
subSamplingRatio = 0.4;
Imax = 200;
<<<<<<< HEAD
maxRealizations = 1;
T_range = [10:20:80];
=======
maxMCRealizations = 10;
T_range = [20:20:80];
>>>>>>> 5bb75a96866ebd179ef5fdffcba962bf98b6daab

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_omp = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
% error_twostage = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(T_range), length(snr_range));
mean_error_omp =  zeros(length(T_range), length(snr_range));
mean_error_vamp =  zeros(length(T_range), length(snr_range));
% mean_error_twostage =  zeros(length(T_range), length(snr_range));

%% Iterations for different SNRs, training length and MC realizations
for snr_indx = 1:length(snr_range)
  snr = 10^(-snr_range(snr_indx)/10);
  snr_db = snr_range(snr_indx);
  
  for t_indx=1:length(T_range)
   T = T_range(t_indx);

   parfor r=1:maxMCRealizations
   disp(['realization: ', num2str(r)]);

    [H,Ar,At] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays);
    Gr = Nr;
    Gt = Nt;
    Dr = 1/sqrt(Nr)*exp(-1j*(0:Nr-1)'*2*pi*(0:Gr-1)/Gr);
    Dt = 1/sqrt(Nt)*exp(-1j*(0:Nt-1)'*2*pi*(0:Gt-1)/Gt);
    [Y, Abar, Zbar, W] = wideband_hybBF_comm_system_training(H, Dr, Dt, T, snr);
    Heff = W'*Dr*Zbar*Abar;
    Mr = size(W'*Dr, 2);
    Mt = size(Abar, 1);
    Heff = W'*Dr*Zbar*Abar;
    % Random sub-sampling
    Omega = zeros(Nr, T);
    for t = 1:T
        indices = randperm(Nr);
        sT = round(subSamplingRatio*Nr);
        indices_sub = indices(1:sT);
        Omega(indices_sub, t) = ones(sT, 1);
    end
    OY = Omega.*Y;
    sT2 = round(subSamplingRatio*T);
    Phi = kron(Abar(:, 1:sT2).', W'*Dr);
    y = vec(Y(:,1:sT2));

    % VAMP sparse recovery
    disp('Running VAMP...');
    s_vamp = vamp(y, Phi+1e-6*eye(size(Phi)), snr, 200*L);
    S_vamp = reshape(s_vamp, Mr, Mt);
    error_vamp(r) = norm(W'*Dr*S_vamp*Abar-Heff)^2/norm(Heff)^2
    if(error_vamp(r)>1)
        error_vamp(r) = 1;
    end
       
    
    % Sparse channel estimation
    disp('Running OMP...');
    s_omp = OMP(Phi, y, 200*L, snr);
    S_omp = reshape(s_omp, Mr, Mt);
    error_omp(r) = norm(W'*Dr*S_omp*Abar-Heff)^2/norm(Heff)^2
    if(error_omp(r)>1)
        error_omp(r)=1;
    end
    
%     % Two-stage scheme matrix completion and sparse recovery
%     disp('Running Two-stage-based Technique..');
%     Y_twostage = mc_svt(Y, OY, Omega, Imax, 0.1);
% %     s_twostage = vamp(vec(Y_twostage), kron(Abar.', W'*Dr), snr, 200*L);
%     s_twostage = OMP(kron(Abar.', W'*Dr), vec(Y_twostage), 200*L, norm(Y_twostage-Y)^2/norm(Y)^2);
%     S_twostage = reshape(s_twostage, Mr, Mt);
%     error_twostage(r) = norm(S_twostage-Zbar)^2/norm(Zbar)^2
%     if(error_twostage(r)>1)
%         error_twostage(r) = 1;
%     end
    
    % Proposed
    disp('Running ADMM-based MCSI...');
<<<<<<< HEAD
    rho = 1e-4;
    tau_S = 1e-8; %1/norm(OY, 'fro')^2;
    [~, Y_mcsi] = proposed_algorithm(OY, Omega, W'*Dr, Abar, Imax, rho*norm(OY, 'fro'), tau_S, rho, Y, Zbar);
%     S_mcsi = pinv(W'*Dr)*Y_mcsi*pinv(Abar);
    error_proposed(r) = norm(Y_mcsi-Heff)^2/norm(Heff)^2;
=======
    rho = 1e-5;
    tau_S = rho/norm(OY, 'fro')^2;
    [~, Y_proposed] = proposed_algorithm(OY, Omega, W'*Dr, Abar, Imax, rho*norm(OY, 'fro'), tau_S, rho, Y, Zbar);
%     S_mcsi = pinv(W'*Dr)*Y_mcsi*pinv(Abar);
    error_proposed(r) = norm(Y_proposed-Heff)^2/norm(Heff)^2;
>>>>>>> 5bb75a96866ebd179ef5fdffcba962bf98b6daab

   end

    mean_error_proposed(t_indx, snr_indx) = mean(error_proposed);
    mean_error_omp(t_indx, snr_indx) = mean(error_omp);
    mean_error_vamp(t_indx, snr_indx) = mean(error_vamp);
%     mean_error_twostage(t_indx, snr_indx) = mean(error_twostage);

  end

end


figure;
p11 = semilogy(T_range, (mean_error_omp(:, 1)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p12 = semilogy(T_range, (mean_error_vamp(:, 1)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 8, 'Color', 'Blue');
% p13 = semilogy(T_range, (mean_error_twostage(:, 1)));hold on;
% set(p13,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Black');
p14 = semilogy(T_range, (mean_error_proposed(:, 1)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Green');
 
% legend({'TD-OMP [11]', 'VAMP [23]', 'TSSR [15]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');
legend({'TD-OMP [11]', 'VAMP [23]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');


xlabel('number of training blocks');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);
 
savefig('results/errorVStraining.fig')
save('results/errorVStraining.mat')
