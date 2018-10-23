clear;
clc;

% Setup parameters
Mt = 32;
Mr = Mt;
Gt = Mt;
Gr = Mr;
Mt_rf=Mt;
Mr_rf=Mr;
total_num_of_clusters = 3;
total_num_of_rays = 10;
L = total_num_of_clusters*total_num_of_rays;
subSamplingRatio_range = [0.1 0.2 0.6];
Imax = 100;
maxRealizations = 1;
snr_db=30;
snr = 10^(-snr_db/10);

% Initialization
mean_error_svt = zeros(length(subSamplingRatio_range), Imax);
mean_error_mc = zeros(length(subSamplingRatio_range), Imax);
mean_error_mcsi = zeros(length(subSamplingRatio_range), Imax);

for r=1:maxRealizations
  disp(['realization: ', num2str(r)]);
  convergence_error_svt = zeros(length(subSamplingRatio_range), Imax);  
  convergence_error_mc = zeros(length(subSamplingRatio_range), Imax);  
  convergence_error_mcsi = zeros(length(subSamplingRatio_range), Imax);

  %%% Signal formulation (channel and training sequence)
  
  for sub_indx=1:length(subSamplingRatio_range)

    [H,Ar,At] = parametric_mmwave_channel(Mr, Mt, total_num_of_clusters, total_num_of_rays);
    Fr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Gr-1]/Gr);
    Ft = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Gt-1]/Gt);
    S = Fr'*H*Ft;
    rank(H)
    rank(S)
    [y,M,OH,Omega] = system_model(H, Fr, Ft, round(subSamplingRatio_range(sub_indx)*Mt*Mr), snr);

    % SVT matrix completion
    [~, convergence_error_svt(sub_indx, :)] = mc_svt(H, OH, Omega, Imax);

    % ADMM matrix completion
%     rho = 0.005;
%     [~, convergence_error_mc(sub_indx, :)] = mcsi_admm(H, OH, Omega, Fr, Ft, Imax, rho*norm(OH), .1/(1+snr_db), rho, 1);%mc_admm(H, OH, Omega, Imax, rho*norm(OH), rho);
    
    % ADMM matrix completion with side-information
    rho = 0.005;
    tau_S = .1/(1+snr_db);
    [~, convergence_error_mcsi(sub_indx, :)] = mcsi_admm(H, OH, Omega, Fr, Ft, Imax, rho*norm(OH), tau_S, rho, 1);

  end
  mean_error_svt = mean_error_svt + convergence_error_svt;
  mean_error_mc = mean_error_mc + convergence_error_mc;
  mean_error_mcsi = mean_error_mcsi + convergence_error_mcsi;
end
mean_error_svt = mean_error_svt/maxRealizations;
mean_error_mc = mean_error_mc/maxRealizations;
mean_error_mcsi = mean_error_mcsi/maxRealizations;

%% Plotting
figure;
marker_stepsize = 50;
p_svt_1 = semilogy(1:Imax,  (mean_error_svt(1, :)));hold on;
set(p_svt_1,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p_svt_2 = semilogy(1:Imax,  (mean_error_svt(2, :)));hold on;
set(p_svt_2,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 8, 'Color', 'Blue');
p_svt_3 = semilogy(1:Imax,  (mean_error_svt(3, :)));hold on;
set(p_svt_3,'LineWidth',2, 'LineStyle', ':', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'p', 'MarkerSize', 8, 'Color', 'Green');
% 
% p_mc_1 = semilogy(1:Imax,  (mean_error_mc(1, :)));hold on;
% set(p_mc_1,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
% p_mc_2 = semilogy(1:Imax,  (mean_error_mc(2, :)));hold on;
% set(p_mc_2,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 8, 'Color', 'Blue');
% p_mc_3 = semilogy(1:Imax,  (mean_error_mc(3, :)));hold on;
% set(p_mc_3,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'p', 'MarkerSize', 8, 'Color', 'Green');

p_mcsi_1 = semilogy(1:Imax,  (mean_error_mcsi(1, :)));hold on;
set(p_mcsi_1,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p_mcsi_2 = semilogy(1:Imax,  (mean_error_mcsi(2, :)));hold on;
set(p_mcsi_2,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 's', 'MarkerSize', 8, 'Color', 'Blue');
p_mcsi_3 = semilogy(1:Imax,  (mean_error_mcsi(3, :)));hold on;
set(p_mcsi_3,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'MarkerIndices', 1:marker_stepsize:Imax, 'Marker', 'p', 'MarkerSize', 8, 'Color', 'Green');
xlabel('algorithm iterations', 'FontSize', 11)
ylabel('MSE (dB)', 'FontSize', 11)
legend({'T=100','T=200','T=300'}, 'FontSize', 12);
grid on;set(gca,'FontSize',12);

savefig(strcat('results/nmse_admm_iterations_',num2str(Mt), '_',num2str(subSamplingRatio_range),'.fig'))