clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));
load('./basic_system_functions/nywireless_channel.mat');

%% Parameter initialization
Nt = 4;
Nr = 32;
Mr_e = 32;
Mr_hbf = Nr;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
maxMCRealizations = 50;
Mr = 4;
numOfnz = 250;
T = 25;
T_hbf = round(T/(Nr/Mr))*Nt;
T_prop = T*Nt;
SNR_range = [-15:3:15];
Imax = 100;

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_proposed_angles = zeros(maxMCRealizations,1);
error_ls = zeros(maxMCRealizations,1);
error_omp_nr = zeros(maxMCRealizations,1);
error_svt = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_cosamp = zeros(maxMCRealizations,1);
error_omp_mmv = zeros(maxMCRealizations,1);
error_tssr = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(SNR_range),1);
mean_error_proposed_angles = zeros(length(SNR_range),1);
mean_error_ls =  zeros(length(SNR_range),1);
mean_error_omp_nr =  zeros(length(SNR_range),1);
mean_error_svt =  zeros(length(SNR_range),1);
mean_error_vamp =  zeros(length(SNR_range),1);
mean_error_cosamp =  zeros(length(SNR_range),1);
mean_error_omp_mmv =  zeros(length(SNR_range),1);
mean_error_tssr =  zeros(length(SNR_range),1);

%% Iterations for different SNRs, training length and MC realizations
for snr_indx = 1:length(SNR_range)
  square_noise_variance = 10^(-SNR_range(snr_indx)/10);

  parfor r=1:maxMCRealizations
      
   disp(['SNR  = ', num2str(square_noise_variance), ', realization: ', num2str(r)]);

   
   %% System model
   Dr = 1/sqrt(Nr)*exp(-1j*(0:Nr-1)'*2*pi*(0:Gr-1)/Gr);
   Dt = 1/sqrt(Nt)*exp(-1j*(0:Nt-1)'*2*pi*(0:Gt-1)/Gt);
   H = zeros(Nr, Nt, L);
   Z = zeros(Nr, Nt, L);
   for l=1:L
        Hfl = Hf{l};
        H(:,:,l) = Hfl(1:Nr, 1:Nt);
        rho = 1/norm(H(:,:,l))^2;
        H(:,:,l) = rho*H(:,:,l);
        Z(:,:,l) = Dr'*H(:,:,l)*Dt;
   end
   Zbar = reshape(Z, Nr, L*Nt);

   % Additive white Gaussian noise
   N = sqrt(square_noise_variance/2)*(randn(Nr, T_prop) + 1j*randn(Nr, T_prop));
   Psi_i = zeros(T_prop, T_prop, Nt);
   % Generate the training symbols
   for k=1:Nt
    % 4-QAM symbols
    s = qam4mod([], 'mod', T_prop);
    Psi_i(:,:,k) =  toeplitz(s);
   end
%    
%    
%          
   %% Conventional HBF with ZC, which gathers Nr RF chains by assuming longer channel coherence time
    [Y_hbf_nr, W_c, Psi_bar] = hbf(H, N(:, 1:T_hbf), Psi_i(1:T_hbf,1:T_hbf,:), T_hbf, Mr_hbf, createBeamformer(Nr, 'ZC'));
    A = W_c'*Dr;
    B = zeros(L*Gt, T_hbf);
    for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
    end    
    Phi = kron((B*B').', A);
    y = vec(Y_hbf_nr*B');
    
    % LS based
    S_ls = pinv(A)*Y_hbf_nr*pinv(B);
    error_ls(r) = norm(S_ls-Zbar)^2/norm(Zbar)^2;
    if(error_ls(r)>1)
        error_ls(r)=1;
    end
% 
%     % OMP based
%     s_omp_nr_solver = spx.pursuit.single.OrthogonalMatchingPursuit(Phi, numOfnz);
%     s_omp_nr = s_omp_nr_solver.solve(y);
%     S_omp_nr = reshape(s_omp_nr.z, Gr, L*Gt);
%     error_omp_nr(r) = norm(S_omp_nr-Zbar)^2/norm(Zbar)^2;
%     if(error_omp_nr(r)>1)
%         error_omp_nr(r)=1;
%     end
%    
% 
    % VAMP-based with ZC and Nr RF chains
    s_vamp = vamp(y, Phi, 1, numOfnz);
    S_vamp = reshape(s_vamp, Gr, L*Gt);
    error_vamp(r) = norm(S_vamp-Zbar)^2/norm(Zbar)^2;
    if(error_vamp(r)>1)
        error_vamp(r) = 1;
    end
    
%     % CoSaMP-based with ZC and Nr RF chains
%     s_cosamp = CoSaMP(Phi, y, numOfnz);
%     S_cosamp = reshape(s_cosamp, Gr, L*Gt);
%     error_cosamp(r) = norm(S_cosamp-Zbar)^2/norm(Zbar)^2;
%     if(error_cosamp(r)>1)
%         error_cosamp(r) = 1;
%     end

    % OMP with MMV based
    s_omp_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, numOfnz);
    S_omp_mmv = s_omp_solver.solve(Y_hbf_nr*pinv(B));
    error_omp_mmv(r) = norm(S_omp_mmv.Z-Zbar)^2/norm(Zbar)^2;
    if(error_omp_mmv(r)>1)
        error_omp_mmv(r)=1;
    end
    
   %% Proposed HBF
   W = createBeamformer(Nr, 'ZC');    
   [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T_prop, Mr_e, Mr, W);
    
   tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
   tau_Z = 1/norm(Zbar, 'fro')^2/2;
   eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
   rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2));

    A = W_tilde'*Dr;
    B = zeros(L*Gt, T_prop);
    for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
    end    
    [S_proposed, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
    error_proposed(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2;
    if(error_proposed(r)>1)
        error_proposed(r)=1;
    end
  
    [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
    [S_proposed_angles, Y_proposed_angles] = proposed_algorithm_angles(Y_proposed_hbf, Omega, indx_S, A, B, Imax, tau_Y, tau_Z, rho, 'approximate', numOfnz);
    error_proposed_angles(r) = norm(S_proposed_angles-Zbar)^2/norm(Zbar)^2;
    if(error_proposed_angles(r)>1)
        error_proposed_angles(r)=1;
    end
    
%     
%     % SVT-based
%     Y_svt = mc_svt(Y_proposed, Omega, Imax, tau_Y, 0.1);
%     S_svt = pinv(A)*Y_svt*pinv(B);
%     error_svt(r) = norm(S_svt-Zbar)^2/norm(Zbar)^2;
%     if(error_svt(r)>1)
%         error_svt(r) = 1;
%     end
%     
%     % TSSR-based
%     A = W_tilde'*Dr;
%     Phi = kron(B.', A);    
%     s_tssr_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, 2*numOfnz);
%     S_tssr = s_tssr_solver.solve(Y_svt*pinv(B));
%     error_tssr(r) = norm(S_tssr.Z-Zbar)^2/norm(Zbar)^2;
%     if(error_tssr(r)>1)
%         error_tssr(r) = 1;
%     end
  end
  
    mean_error_proposed(snr_indx) = mean(error_proposed);
    mean_error_proposed_angles(snr_indx) = mean(error_proposed_angles);
    mean_error_ls(snr_indx) = mean(error_ls);
    mean_error_omp_nr(snr_indx) = mean(error_omp_nr);
    mean_error_svt(snr_indx) = mean(error_svt);   
    mean_error_vamp(snr_indx) = mean(error_vamp);
    mean_error_cosamp(snr_indx) = mean(error_cosamp);
    mean_error_omp_mmv(snr_indx) = mean(error_omp_mmv);
    mean_error_tssr(snr_indx) = mean(error_tssr);

end


figure;
p = semilogy(SNR_range, mean_error_ls);hold on;
set(p, 'LineWidth',2, 'LineStyle', ':', 'Color', 'Black');
% p = semilogy(SNR_range, mean_error_svt);hold on;
% set(p, 'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(SNR_range, mean_error_omp_nr);hold on;
% set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '<', 'MarkerSize', 6, 'Color', 'Black');
p = semilogy(SNR_range, mean_error_vamp);hold on;
set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(SNR_range, mean_error_cosamp);hold on;
% set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Black');
p = semilogy(SNR_range, mean_error_omp_mmv);hold on;
set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'White', 'Marker', '+', 'MarkerSize', 6, 'Color', 'Black');
% p = semilogy(SNR_range, mean_error_tssr);hold on;
% set(p, 'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'x', 'MarkerSize', 6, 'Color', 'Black');
p = semilogy(SNR_range, (mean_error_proposed));hold on;
set(p,'LineWidth',1, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'h', 'MarkerSize', 8, 'Color', 'Blue');
p = semilogy(SNR_range, (mean_error_proposed_angles));hold on;
set(p,'LineWidth',1, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 's', 'MarkerSize', 8, 'Color', 'Green');

% legend({'LS', 'SVT', 'OMP', 'VAMP', 'CoSaMP', 'MMV-OMP', 'TSSR',  'Proposed', 'Proposed with angle information'}, 'FontSize', 12, 'Location', 'Best');
legend({'LS', 'VAMP', 'MMV-OMP', 'Proposed', 'Proposed with angle information'}, 'FontSize', 11, 'Location', 'Best');


xlabel('SNR (dB)');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',11);

savefig('results/errorVSsnr_angles_nyuwireless.fig')
