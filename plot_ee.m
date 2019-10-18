Nt = 16;
Nr = 64;
Mr_e = 32;
Mr_range = 1:3:Mr_e;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
T = 5;
square_noise_variance = 10^(-15/10);
maxMCRealizations = 10000;


Capacity_dbf = zeros(maxMCRealizations,1);
Capacity_hbf_ps = zeros(maxMCRealizations,1);
Capacity_hbf_zc = zeros(maxMCRealizations,1);
Capacity_proposed = zeros(maxMCRealizations,1);
mean_Capacity_dbf = zeros(length(Mr_range),1);
mean_Capacity_hbf_ps = zeros(length(Mr_range),1);
mean_Capacity_hbf_zc = zeros(length(Mr_range),1);
mean_Capacity_proposed = zeros(length(Mr_range),1);
power_dbf = zeros(length(Mr_range),1);
power_hbf = zeros(length(Mr_range),1);
power_hbf_zc = zeros(length(Mr_range),1);
power_proposed = zeros(length(Mr_range),1);
ee_dbf = zeros(length(Mr_range),1);
ee_hbf_ps = zeros(length(Mr_range),1);
ee_hbf_zc = zeros(length(Mr_range),1);
ee_proposed = zeros(length(Mr_range),1);

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
     Capacity_dbf(r) = real(log2(det(eye(Nr) + 1/square_noise_variance*1/Nt*W_c'*(Y*Y')*W_c)));
     
     % Conventional HBF with phase-shifters 
     W = createBeamformer(Nr, 'quantized');     
     [~, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_ps(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     % Conventional HBF with ZC
     W = createBeamformer(Nr, 'ZC');     
     [Y_conventional, W_c, ~, Y] = hbf(H, zeros(Nr, T), Psi_i, T, Mr, W);
     Capacity_hbf_zc(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* W_c'*(Y*Y')*W_c)));

     
     % Proposed design
     W = createBeamformer(Nr, 'quantized');
     [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, zeros(Nr, T), Psi_i, T, Mr_e, Mr, W);    
     ind = randperm(Mr_e);
     Capacity_proposed(r) = real(log2(det(eye(Mr) + 1/square_noise_variance*1/Nt* (W_tilde(:, ind(1:Mr))'*(Y*Y')*W_tilde(:, ind(1:Mr))))));

 end
 
 Pcirc = 0;
 Psw = 0.005;
 Pps = 0.015;
 Plna = 0.02;
 Pps_zc = 0.06;
 power_dbf(mr_index) = Pcirc + Nr*Nr*Plna + Nr*(Nr+1)*Pps_zc;
 power_hbf(mr_index) = Pcirc + Mr*Nr*Plna + Nr*(Mr+1)*Pps;
 power_hbf_zc(mr_index) = Pcirc + Mr*Nr*Plna + Nr*(Mr+1)*Pps_zc;
 power_proposed(mr_index) = Pcirc + Mr_e*Nr*Plna+Mr_e*Psw+Nr*(Mr_e+1)*Pps;
 
 mean_Capacity_dbf(mr_index) = mean(Capacity_dbf);
 mean_Capacity_hbf_ps(mr_index) = mean(Capacity_hbf_ps);
 mean_Capacity_hbf_zc(mr_index) = mean(Capacity_hbf_zc);
 mean_Capacity_proposed(mr_index) = mean(Capacity_proposed);
 
 ee_dbf(mr_index) = mean_Capacity_dbf(mr_index)/power_dbf(mr_index);
 ee_hbf_ps(mr_index) = mean_Capacity_hbf_ps(mr_index)/power_hbf(mr_index);
 ee_hbf_zc(mr_index) = mean_Capacity_hbf_zc(mr_index)/power_hbf_zc(mr_index);
 ee_proposed(mr_index) = mean_Capacity_proposed(mr_index)/power_proposed(mr_index);
end

figure;
subplot(1,2,1)
plot(Mr_range, power_dbf, 'k-'); hold on;
plot(Mr_range, power_hbf, 'ko-'); hold on;
plot(Mr_range, power_hbf_zc, 'kx-'); hold on;
plot(Mr_range, power_proposed, 'bs-'); hold on;
grid on;
ylabel('Power consumption (mW)');
xlabel('Number of RF chains')
legend('Digital Beamforming', 'Conventional HBF with PS (6 bits)', 'Conventional HBF with ZC', 'Proposed HBF with PS (6 bits)')

subplot(1,2,2)
plot(Mr_range, ee_dbf, 'k-'); hold on;
plot(Mr_range, ee_hbf_ps, 'ko-'); hold on;
plot(Mr_range, ee_hbf_zc, 'kx-'); hold on;
plot(Mr_range, ee_proposed, 'bs-'); hold on;
grid on;
ylabel('Energy Efficiency (bits/Joule)');
xlabel('Number of RF chains')
legend('Digital Beamforming', 'Conventional HBF with PS (6 bits)', 'Conventional HBF with ZC', 'Proposed HBF with PS (6 bits)')

savefig('results/power_ee.fig')
