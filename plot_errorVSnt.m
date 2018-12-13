clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt_range = [2:12];
Nr = 32;
Gr = Nr;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
snr_range = 5;
subSamplingRatio = 0.75;
maxMCRealizations = 50;
T = 50;
Imax = 100;

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_omp = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_twostage = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(Nt_range), length(snr_range));
mean_error_omp =  zeros(length(Nt_range), length(snr_range));
mean_error_vamp =  zeros(length(Nt_range), length(snr_range));
mean_error_twostage =  zeros(length(Nt_range), length(snr_range));
    
%% Iterations for different SNRs, training length and MC realizations
for snr_indx = 1:length(snr_range)
  snr = 10^(-snr_range(snr_indx)/10);
  
  for nt_indx=1:length(Nt_range)
   Nt = Nt_range(nt_indx);
   Gt = Nt;
   
   parfor r=1:maxMCRealizations
   disp(['Nt = ', num2str(Nt), ', realization: ', num2str(r)]);

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    [Y_proposed_hbf, Y_conventional_hbf, W_tilde, Psi_bar, Omega, Lr] = wideband_hybBF_comm_system_training(H, T, snr, subSamplingRatio, Gr);
    numOfnz = 100;
%     length(find(abs(Zbar)/norm(Zbar)^2>1e-3))

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
    [~, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho);

    S_proposed = pinv(A)*Y_proposed*pinv(B);
    error_proposed(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2;
    if(error_proposed(r)>1)
        error_proposed(r)=1;
    end
  
%     disp('Running Two-stage-based Technique..');
    Y_twostage = mc_svt(Y_proposed_hbf, Omega, Imax,  tau_X, 0.1);
    S_twostage = pinv(A)*Y_twostage*pinv(B);    
    error_twostage(r) = norm(S_twostage-Zbar)^2/norm(Zbar)^2;
    if(error_twostage(r)>1)
        error_twostage(r) = 1;
    end
    
%     disp('Running VAMP...');
    Phi = kron(B.', A);
    y = vec(Y_conventional_hbf);
    s_vamp = vamp(y, Phi, snr, numOfnz);
    S_vamp = reshape(s_vamp, Nr, L*Nt);
    error_vamp(r) = norm(S_vamp-Zbar)^2/norm(Zbar)^2;
    if(error_vamp(r)>1)
        error_vamp(r) = 1;
    end
       
    
%     disp('Running OMP...');
    s_omp = OMP(Phi, y, numOfnz, snr);
    S_omp = reshape(s_omp, Nr, L*Nt);
    error_omp(r) = norm(S_omp-Zbar)^2/norm(Zbar)^2;
    if(error_omp(r)>1)
        error_omp(r)=1;
    end
  


   end

    mean_error_proposed(nt_indx, snr_indx) = mean(error_proposed);
    mean_error_omp(nt_indx, snr_indx) = mean(error_omp);
    mean_error_vamp(nt_indx, snr_indx) = mean(error_vamp);
    mean_error_twostage(nt_indx, snr_indx) = mean(error_twostage);

  end

end


figure;
p11 = semilogy(Nt_range, (mean_error_omp(:, 1)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p12 = semilogy(Nt_range, (mean_error_vamp(:, 1)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Blue');
p13 = semilogy(Nt_range, (mean_error_twostage(:, 1)));hold on;
set(p13,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p14 = semilogy(Nt_range, (mean_error_proposed(:, 1)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');
 
legend({'TD-OMP [11]', 'VAMP [23]', 'TSSR [15]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');


xlabel('number of transmitting antennas');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);
 
savefig('results/errorVSnt.fig')

