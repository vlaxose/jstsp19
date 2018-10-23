clear;
clc;

%%% Initialization
Nt = 32;
Nr = 8;
total_num_of_clusters = 2;
total_num_of_rays = 1;
Np = total_num_of_clusters*total_num_of_rays;
path_range = [1 2 3 4];
snr_range = 5;
snr = 10^(-snr_range/10);
subSamplingRatio_range = 0.6;
Imax = 200;
maxRealizations = 50;

error_mcsi = zeros(maxRealizations,1);
error_omp = zeros(maxRealizations,1);
error_vamp = zeros(maxRealizations,1);
error_twostage = zeros(maxRealizations,1);

mean_error_mcsi = zeros(length(subSamplingRatio_range), length(path_range));
mean_error_omp =  zeros(length(subSamplingRatio_range), length(path_range));
mean_error_vamp =  zeros(length(subSamplingRatio_range), length(path_range));
mean_error_twostage =  zeros(length(subSamplingRatio_range), length(path_range));

for paths_indx = 1:length(path_range)
  L = path_range(paths_indx);
  for sub_indx=1:length(subSamplingRatio_range)
    T = round(subSamplingRatio_range(sub_indx)*Nt*Nr);

   for r=1:maxRealizations
   disp(['realization: ', num2str(r)]);

    [H,Ar,At] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays);
    Dr = 1/sqrt(Nr)*exp(-1j*(0:Nr-1)'*2*pi*(0:Nr-1)/Nr);
    Dt = 1/sqrt(Nt)*exp(-1j*(0:Nt-1)'*2*pi*(0:Nt-1)/Nt);
    [Y, Abar, Zbar, W] = wideband_hybBF_comm_system_training(H, Dr, Dt, T, snr);
    Gr = size(W'*Dr, 2);
    Gt = size(Abar, 1);
    % Random sub-sampling
    indices = randperm(Nr*T);
    sT = round(subSamplingRatio_range*Nr*T);
    indices_sub = indices(1:sT);
  	Omega = zeros(Nr, T);
    Omega(indices_sub) = ones(sT, 1);
    OY = Omega.*Y;
    sT2 = round(subSamplingRatio_range*T);
    Phi = kron(Abar(:, 1:sT2).', W'*Dr);
    y = vec(Y(:,1:sT2));
    
    % VAMP sparse recovery
    disp('Running VAMP...');
    s_vamp = vamp(y, Phi, snr, 2*L);
    S_vamp = reshape(s_vamp, Gr, Gt);
    error_vamp(r) = norm(S_vamp-Zbar)^2/norm(Zbar)^2;
    
    % Sparse channel estimation
    disp('Running OMP...');
    s_omp = OMP(Phi, y, 2*L);
    S_omp = reshape(s_omp, Gr, Gt);
    error_omp(r) = norm(S_omp-Zbar)^2/norm(Zbar)^2;      
% 
% 
    
%     
% 
%     % Two-stage scheme matrix completion and sparse recovery
%     disp('Running Two-stage-based Technique..');
%     X_twostage_1 = mc_svt(H, OH, Omega, Imax);
%     s_twostage = vamp(vec(X_twostage_1), kron(conj(Dt), Dr), 0.001, 2*L);
%     X_twostage = Dr*reshape(s_twostage, Nr, Nt)*Dt';
%     error_twostage(r) = norm(H-X_twostage)^2/norm(H)^2;
%     
    % ADMM matrix completion with side-information
    disp('Running ADMM-based MCSI...');
    rho = 0.0001;
    [~, Y_mcsi] = mcsi_admm(OY, Omega, W'*Dr, Abar, Imax, rho*norm(OY), rho, Zbar, Y, L, snr);
    S_mcsi = pinv(W'*Dr)*Y_mcsi*pinv(Abar);
    error_mcsi(r) = norm(S_mcsi-Zbar)^2/norm(Zbar)^2;

   end

    mean_error_mcsi(sub_indx, paths_indx) = min(mean(error_mcsi), 1);
    mean_error_omp(sub_indx, paths_indx) = min(mean(error_omp), 1);
    mean_error_vamp(sub_indx, paths_indx) = min(mean(error_vamp), 1);
    mean_error_twostage(sub_indx, paths_indx) = min(mean(error_twostage), 1);

  end

end


figure;
p11 = semilogy(path_range, (mean_error_omp(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p12 = semilogy(path_range, (mean_error_vamp(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 8, 'Color', 'Blue');
% p13 = semilogy(snr_range, (mean_error_twostage(1, :)));hold on;
% set(p13,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Cyan', 'MarkerFaceColor', 'Cyan', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Cyan');
p14 = semilogy(path_range, (mean_error_mcsi(1, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Green');
 
legend({'OMP [4]', 'VAMP [12]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');
 
xlabel('Number of paths');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);
 
 savefig(strcat('results/errorVSpaths.fig'))