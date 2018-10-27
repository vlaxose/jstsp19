clear;
clc;

%%% Initialization
Mt = 32;
Mr = Mt;
total_num_of_clusters = 2;
total_num_of_rays = 1;
L = total_num_of_clusters*total_num_of_rays;

snr_range = [0:10:30];
subSamplingRatio_range = [0.5];
Imax = 50;
maxRealizations = 1;

rate_mcsi = zeros(maxRealizations,1);
rate_opt = zeros(maxRealizations,1);
rate_omp = zeros(maxRealizations,1);
rate_vamp = zeros(maxRealizations,1);
rate_twostage = zeros(maxRealizations,1);

mean_rate_mcsi = zeros(length(subSamplingRatio_range), length(snr_range));
mean_rate_opt =  zeros(length(subSamplingRatio_range), length(snr_range));
mean_rate_omp =  zeros(length(subSamplingRatio_range), length(snr_range));
mean_rate_vamp =  zeros(length(subSamplingRatio_range), length(snr_range));
mean_rate_twostage =  zeros(length(subSamplingRatio_range), length(snr_range));

for snr_indx = 1:length(snr_range)
  snr = 10^(-snr_range(snr_indx)/10);

  for sub_indx=1:length(subSamplingRatio_range)

   for r=1:maxRealizations
   disp(['realization: ', num2str(r)]);

    [H,Ar,At] = parametric_mmwave_channel(Mr, Mt, total_num_of_clusters, total_num_of_rays);
    [Uh,Sh,Vh] = svd(H);
    rate_opt(r) = log2(real(det(eye(Mr)+1/(Mt*Mr)*1/snr*H*H')));
    
   
    Fr = 1/sqrt(Mr)*exp(-1j*[0:Mr-1]'*2*pi*[0:Mr-1]/Mr);
    Ft = 1/sqrt(Mt)*exp(-1j*[0:Mt-1]'*2*pi*[0:Mt-1]/Mt);
    [y,M,OH,Omega] = system_model(H, Fr, Ft, round(subSamplingRatio_range(sub_indx)*Mt*Mr), snr);


    % VAMP sparse recovery
    s_vamp = vamp(y, M, snr, 2*L);
    X_vamp = Fr*reshape(s_vamp, Mr, Mt)*Ft';
    [U_vamp, S_vamp, V_vamp] = svd(X_vamp);
    rate_vamp(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_vamp)^2/norm(H)^2))));

    % Sparse channel estimation
    s_omp = OMP(M, y, 2*L);
    X_omp = Fr*reshape(s_omp, Mr, Mt)*Ft';
    [U_omp, S_omp, V_omp] = svd(X_omp);
    rate_omp(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_omp)^2/norm(H)^2))));
    
    % ADMM matrix completion with side-information
    rho = 0.005;
    tau_S = .1/(1+snr_range(snr_indx));
    X_mcsi = mcsi_admm(H, OH, Omega, Fr, Ft, Imax, rho*norm(OH), tau_S, rho, 1);
    [U_mcsi,S_mcsi,V_mcsi] = svd(X_mcsi);
    rate_mcsi(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_mcsi)^2/norm(H)^2))));
    
    % Two-stage scheme matrix completion and sparse recovery
    X_twostage_1 = mc_svt(H, OH, Omega, Imax);
    s_twostage = vamp(vec(X_twostage_1), kron(conj(Ft), Fr), 0.001, 2*L);
    X_twostage = Fr*reshape(s_twostage, Mr, Mt)*Ft';
    [U_twostage,S_twostage,V_twostage] = svd(X_twostage);
    rate_twostage(r) = log2(real(det(eye(Mr) + 1/(Mt*Mr)*H*H'*1/(snr+norm(H-X_twostage)^2/norm(H)^2))));
   end

    mean_rate_mcsi(sub_indx, snr_indx) = mean(rate_mcsi);
    mean_rate_omp(sub_indx, snr_indx) = mean(rate_omp);
    mean_rate_opt(sub_indx, snr_indx) = mean(rate_opt);
    mean_rate_vamp(sub_indx, snr_indx) = mean(rate_vamp);
    mean_rate_twostage(sub_indx, snr_indx) = mean(rate_twostage);

  end

end


figure;
p11 = plot(snr_range, (mean_rate_omp(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 8, 'Color', 'Black');
p12 = plot(snr_range, (mean_rate_twostage(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Cyan', 'MarkerFaceColor', 'Cyan', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Cyan');
p15 = plot(snr_range, (mean_rate_vamp(1, :)));hold on;
set(p15,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 8, 'Color', 'Blue');
p14 = plot(snr_range, (mean_rate_mcsi(1, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Green');
p16 = plot(snr_range, (mean_rate_opt(1, :)));hold on;
set(p16,'LineWidth',2, 'LineStyle', '--', 'Color', 'Black');

legend({'OMP [19]', 'Two-stage scheme [21]', 'VAMP [20]', 'Proposed', 'Perfect CSI'}, 'FontSize', 12);

xlabel('SNR (dB)');
ylabel('ASE')
grid on;set(gca,'FontSize',12);

savefig(strcat('results/rateVSsnr_',num2str(subSamplingRatio_range),'.fig'))