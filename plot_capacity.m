clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%%
Nt = 16;
Nr = 32;
Mr_e = 32;
Mr_range = 1:3:floor(Mr_e);
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
T = 5;
square_noise_variance = 10^(-15/10);
maxMCRealizations = 1e4;


Capacity_dbf = zeros(maxMCRealizations,1);
Capacity_hbf_ps = zeros(maxMCRealizations,1);
Capacity_hbf_zc = zeros(maxMCRealizations,1);
Capacity_proposed = zeros(maxMCRealizations,1);
mean_Capacity_dbf = zeros(length(Mr_range),1);
mean_Capacity_hbf_ps = zeros(length(Mr_range),1);
mean_Capacity_hbf_zc = zeros(length(Mr_range),1);
mean_Capacity_proposed = zeros(length(Mr_range),1);

for mr_index=1:length(Mr_range)
 Mr = Mr_range(mr_index);
 
 parfor r=1:maxMCRealizations
     [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
      Psi_i = zeros(T, T, Nt);
      for k=1:Nt
       s = qam4mod([], 'mod', T);
       Psi_i(:,:,k) =  toeplitz(s);
      end
      

     % Digital BF
     W = createBeamformer(Nr, 'ZC');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Nr, W);     
     Capacity_dbf(r) = real(log2(det(eye(Nr) + 1/square_noise_variance* 1/Nt*W_c'*(Y*Y')*W_c)));

     % Conventional HBF with phase-shifters 
     W = createBeamformer(Nr, 'quantized');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_ps(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     % Conventional HBF with ZC
     W = createBeamformer(Nr, 'ZC');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_zc(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     
     % Proposed design
     W = createBeamformer(Nr, 'quantized');     
     [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, zeros(Nr, T), Psi_i, T, Mr_e, Mr, W);    
     ind = randperm(Mr_e);
     Capacity_proposed(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* (W(:, ind(1:Mr))'*(Y*Y')*W(:, ind(1:Mr))))));

 end
 
 
 
 mean_Capacity_dbf(mr_index) = mean(Capacity_dbf);
 mean_Capacity_hbf_ps(mr_index) = mean(Capacity_hbf_ps);
 mean_Capacity_hbf_zc(mr_index) = mean(Capacity_hbf_zc);
 mean_Capacity_proposed(mr_index) = mean(Capacity_proposed);
 
end

figure;
subplot(1,3,1)
plot(Mr_range, mean_Capacity_dbf, 'k-');
hold on;
plot(Mr_range, mean_Capacity_hbf_ps, 'kx-');
hold on;
plot(Mr_range, mean_Capacity_hbf_zc, 'k--');
hold on;
plot(Mr_range, mean_Capacity_proposed, 'bs-');
xlabel('Number of RF chains');
ylabel('ASE (bits/sec)')
grid on


%%
Nt = 16;
Nr = 64;
Mr_e = 32;
Mr_range = 1:3:floor(Mr_e);
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
T = 5;
square_noise_variance = 10^(-15/10);
maxMCRealizations = 1e4;


Capacity_dbf = zeros(maxMCRealizations,1);
Capacity_hbf_ps = zeros(maxMCRealizations,1);
Capacity_hbf_zc = zeros(maxMCRealizations,1);
Capacity_proposed = zeros(maxMCRealizations,1);
mean_Capacity_dbf = zeros(length(Mr_range),1);
mean_Capacity_hbf_ps = zeros(length(Mr_range),1);
mean_Capacity_hbf_zc = zeros(length(Mr_range),1);
mean_Capacity_proposed = zeros(length(Mr_range),1);

for mr_index=1:length(Mr_range)
 Mr = Mr_range(mr_index);
 
 parfor r=1:maxMCRealizations
     [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
      Psi_i = zeros(T, T, Nt);
      for k=1:Nt
       s = qam4mod([], 'mod', T);
       Psi_i(:,:,k) =  toeplitz(s);
      end
      

     % Digital BF
     W = createBeamformer(Nr, 'ZC');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Nr, W);     
     Capacity_dbf(r) = real(log2(det(eye(Nr) + 1/square_noise_variance* 1/Nt*W_c'*(Y*Y')*W_c)));

     % Conventional HBF with phase-shifters 
     W = createBeamformer(Nr, 'quantized');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_ps(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     % Conventional HBF with ZC
     W = createBeamformer(Nr, 'ZC');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_zc(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     
     % Proposed design
     W = createBeamformer(Nr, 'quantized');     
     [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, zeros(Nr, T), Psi_i, T, Mr_e, Mr, W);    
     ind = randperm(Mr_e);
     Capacity_proposed(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* (W(:, ind(1:Mr))'*(Y*Y')*W(:, ind(1:Mr))))));

 end
 
 
 
 mean_Capacity_dbf(mr_index) = mean(Capacity_dbf);
 mean_Capacity_hbf_ps(mr_index) = mean(Capacity_hbf_ps);
 mean_Capacity_hbf_zc(mr_index) = mean(Capacity_hbf_zc);
 mean_Capacity_proposed(mr_index) = mean(Capacity_proposed);
 
end

subplot(1,3,2)
plot(Mr_range, mean_Capacity_dbf, 'k-');
hold on;
plot(Mr_range, mean_Capacity_hbf_ps, 'kx-');
hold on;
plot(Mr_range, mean_Capacity_hbf_zc, 'k--');
hold on;
plot(Mr_range, mean_Capacity_proposed, 'bs-');
xlabel('Number of RF chains');
ylabel('ASE (bits/sec)')
grid on


%%
Nt = 16;
Nr = 128;
Mr_e = 64;
Mr_range = 1:3:floor(Mr_e/2);
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
T = 5;
square_noise_variance = 10^(-15/10);
maxMCRealizations = 1e4;


Capacity_dbf = zeros(maxMCRealizations,1);
Capacity_hbf_ps = zeros(maxMCRealizations,1);
Capacity_hbf_zc = zeros(maxMCRealizations,1);
Capacity_proposed = zeros(maxMCRealizations,1);
mean_Capacity_dbf = zeros(length(Mr_range),1);
mean_Capacity_hbf_ps = zeros(length(Mr_range),1);
mean_Capacity_hbf_zc = zeros(length(Mr_range),1);
mean_Capacity_proposed = zeros(length(Mr_range),1);

for mr_index=1:length(Mr_range)
 Mr = Mr_range(mr_index);
 
 parfor r=1:maxMCRealizations
     [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
      Psi_i = zeros(T, T, Nt);
      for k=1:Nt
       s = qam4mod([], 'mod', T);
       Psi_i(:,:,k) =  toeplitz(s);
      end
      

     % Digital BF
     W = createBeamformer(Nr, 'ZC');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Nr, W);     
     Capacity_dbf(r) = real(log2(det(eye(Nr) + 1/square_noise_variance* 1/Nt*W_c'*(Y*Y')*W_c)));

     % Conventional HBF with phase-shifters 
     W = createBeamformer(Nr, 'quantized');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_ps(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     % Conventional HBF with ZC
     W = createBeamformer(Nr, 'ZC');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_zc(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     
     % Proposed design
     W = createBeamformer(Nr, 'quantized');     
     [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, zeros(Nr, T), Psi_i, T, Mr_e, Mr, W);    
     ind = randperm(Mr_e);
     Capacity_proposed(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* (W(:, ind(1:Mr))'*(Y*Y')*W(:, ind(1:Mr))))));

 end
 
 

 mean_Capacity_dbf(mr_index) = mean(Capacity_dbf);
 mean_Capacity_hbf_ps(mr_index) = mean(Capacity_hbf_ps);
 mean_Capacity_hbf_zc(mr_index) = mean(Capacity_hbf_zc);
 mean_Capacity_proposed(mr_index) = mean(Capacity_proposed);
 
end

subplot(1,3,3)
plot(Mr_range, mean_Capacity_dbf, 'k-');
hold on;
plot(Mr_range, mean_Capacity_hbf_ps, 'kx-');
hold on;
plot(Mr_range, mean_Capacity_hbf_zc, 'k--');
hold on;
plot(Mr_range, mean_Capacity_proposed, 'bs-');
xlabel('Number of RF chains');
ylabel('ASE (bits/sec)')
legend('Digital Beamforming', 'Conventional HBF with PS (6 bits)', 'Conventional HBF with ZC', 'Proposed HBF with PS (6 bits)', 'Orientation', 'Horizontal','NumColumns',2)
grid on

savefig('results/capacity.fig')
