clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt = 16;
Nr = 32;
Mr_e = 32;
Mr = 16;
Gr = Nr;
Gt = Nt;
FrameSize = 16;
total_num_of_clusters = 2;
total_num_of_rays = 6;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
maxMCRealizations = 1;
F_range = [5];
T_range = FrameSize*F_range;
Imax = 50;
square_noise_variance = 10^(-15/10);

%% Variables initialization
error_proposed_z = zeros(maxMCRealizations,1);
error_proposed_y = zeros(maxMCRealizations,1);
mean_error_proposed_z =  zeros(length(T_range),1);
mean_error_proposed_y =  zeros(length(T_range),1);

%% Iterations for different SNRs, training length and MC realizations
for t_index = 1:length(T_range)
   T = T_range(t_index);
   
   for r=1:maxMCRealizations
      
   disp(['frame index = ', num2str(t_index), ', realization: ', num2str(r)]);

 %% System model
   [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
%    F = createBeamformer(Nt, 'fft');

   % Additive white Gaussian noise
   N = sqrt(square_noise_variance/2)*(randn(Nr, T) + 1j*randn(Nr, T));
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols

   for k=1:Nt
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end
   
   W = createBeamformer(Nr, 'ps');
    
    %% Proposed HBF
   [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);
   B = zeros(L*Gt, T);
   for l=1:L
      B((l-1)*Gt+1:l*Gt, :) = Dt'*Psi_bar(:,:,l);
   end    

   tau_Y = 1/norm(Y_proposed_hbf, 'fro')^2;
   tau_Z = 1/norm(Zbar, 'fro')^2/2;
   eigvalues = eigs(Y_proposed_hbf'*Y_proposed_hbf);
   rho = sqrt(min(eigvalues)*(1/norm(Y_proposed_hbf, 'fro')^2))/2;
    A = W_tilde'*Dr;
    [S_proposed, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
    error_proposed_z(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2;
    if(error_proposed_z(r)>1)
        error_proposed_z(r)=1;
    end
  
    S_proposed = A'*Y_proposed*pinv(B);
    error_proposed_y(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2;
    if(error_proposed_y(r)>1)
        error_proposed_y(r)=1;
    end
     
   end

    mean_error_proposed_z(t_index) = mean(error_proposed_z);
    mean_error_proposed_y(t_index) = mean(error_proposed_y);
   

end


figure;
p = semilogy(F_range, (mean_error_proposed_z));hold on;
set(p,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 'h', 'MarkerSize', 4, 'Color', 'Black');
p = semilogy(F_range, (mean_error_proposed_y));hold on;
set(p,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 's', 'MarkerSize', 4, 'Color', 'Green');

legend({'Z', 'Y'}, 'FontSize', 12, 'Location', 'Best');


xlabel('Frame index (T)');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);
