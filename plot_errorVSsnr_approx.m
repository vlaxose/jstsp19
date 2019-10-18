clear;
clc;

addpath([pwd, '/basic_system_functions']);
addpath(genpath([pwd, '/benchmark_algorithms']));

%% Parameter initialization
Nt = 4;
Nr = 32;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
snr_range = [-15:5:15];
subSamplingRatio = 0.75;
maxMCRealizations = 50;
T = 70;
Imax_range = [10 30 50];

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_proposed_approx = zeros(maxMCRealizations,1);
error_omp = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_twostage = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(subSamplingRatio), length(snr_range));
mean_error_proposed_approx = zeros(length(subSamplingRatio), length(snr_range));
mean_error_omp =  zeros(length(subSamplingRatio), length(snr_range));
mean_error_vamp =  zeros(length(subSamplingRatio), length(snr_range));
mean_error_twostage =  zeros(length(subSamplingRatio), length(snr_range));

%% Iterations for different SNRs, training length and MC realizations
for snr_indx = 1:length(snr_range)
  square_noise_variance = 10^(-snr_range(snr_indx)/10);

  for imax_indx=1:length(Imax_range)
  Imax = Imax_range(imax_indx);
      
  parfor r=1:maxMCRealizations
      
   disp(['square_noise_variance = ', num2str(square_noise_variance), ', realization: ', num2str(r)]);

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    [Y_proposed_hbf, Y_conventional_hbf, W_tilde, Psi_bar, Omega, Lr] = wideband_hybBF_comm_system_training(H, T, square_noise_variance, subSamplingRatio);
    numOfnz = 100;%length(find(abs(Zbar)/norm(Zbar)^2>1e-3));
    
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

    [~, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho, 'std');
    S_proposed = pinv(A)*Y_proposed*pinv(B);
    error_proposed(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2;
    if(error_proposed(r)>1)
        error_proposed(r)=1;
    end

    [~, Y_proposed_approx] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho, 'approximate');
    S_proposed_approx = pinv(A)*Y_proposed_approx*pinv(B);
    error_proposed_approx(r) = norm(S_proposed_approx-Zbar)^2/norm(Zbar)^2;
    if(error_proposed_approx(r)>1)
        error_proposed_approx(r)=1;
    end

    
   end

    mean_error_proposed(imax_indx, snr_indx) = min(mean(error_proposed), 1);
    mean_error_proposed_approx(imax_indx, snr_indx) = min(mean(error_proposed_approx), 1);
    mean_error_omp(imax_indx, snr_indx) = min(mean(error_omp), 1);
    mean_error_vamp(imax_indx, snr_indx) = min(mean(error_vamp), 1);
    mean_error_twostage(imax_indx, snr_indx) = min(mean(error_twostage), 1);

  end

end


figure;
p11 = semilogy(snr_range, (mean_error_proposed(1, :)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p12 = semilogy(snr_range, (mean_error_proposed_approx(1, :)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');
p13 = semilogy(snr_range, (mean_error_proposed(2, :)));hold on;
set(p13,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p14 = semilogy(snr_range, (mean_error_proposed_approx(2, :)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');
p15 = semilogy(snr_range, (mean_error_proposed(3, :)));hold on;
set(p15,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Blue');
p16 = semilogy(snr_range, (mean_error_proposed_approx(3, :)));hold on;
set(p16,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');
 
legend({'Algorithm 1', 'Algorithm 2'}, 'FontSize', 12, 'Location', 'Best');


xlabel('SNR (dB)');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);

savefig('results/errorVSsnr.fig')
