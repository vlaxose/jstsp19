clc
clear

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));
figure;

%% 
Nt = 4;
Nr = 32;
Mr_e = 32;
Mr = 4;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1 4 8];
T = 50;

Nr_min = min(Nr, Mr_e);
eig_dist = zeros(length(L_range), Nr_min);

 for l_indx=1:length(L_range)
  L = L_range(l_indx);
  [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols
   for k=1:Nt
    % Gaussian random symbols
 %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end

  N = zeros(Nr, T);
  W = createBeamformer(Nr, 'ZC');
  [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);


  A = W_tilde'*Dr;
  B = zeros(L*Nt, T);
  for l=1:L
     B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
  end

  [U,S,V]=svd(Y);
  eig_dist(l_indx, :) = diag(S(1:Nr_min, 1:Nr_min));
 end

eig_dist = squeeze(mean(eig_dist, 3));

subplot(2, 3, 1)
for l_indx=1:length(L_range)
plot(eig_dist(l_indx,:))
hold on;
end
r = min(Np, L*Nt);
vline(r+1, '--', ['min(N_p, L N_T)=N_p=', num2str(min(Np, L*Nt))]);
legend('L=1', 'L=4', 'L=8')
grid on;
ylabel('Singular value');
xlabel('Index')
title(strcat('N_r=', num2str(Nr),', M_r^e=', num2str(Mr_e),',N_p=', num2str(Np)))

%% 
Nt = 4;
Nr = 64;
Mr_e = 32;
Mr = 4;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1 4 8];
T = 50;

Nr_min = min(Nr, Mr_e);

 for l_indx=1:length(L_range)
  L = L_range(l_indx);
  [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols
   for k=1:Nt
    % Gaussian random symbols
 %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end

  N = zeros(Nr, T);

  W = createBeamformer(Nr, 'ZC');
  [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);

  A = W_tilde'*Dr;
  B = zeros(L*Nt, T);
  for l=1:L
     B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
  end

  [U,S,V]=svd(Y);
  eig_dist(l_indx, :) = diag(S(1:Nr_min, 1:Nr_min));
 end

subplot(2, 3, 2)
for l_indx=1:length(L_range)
plot(eig_dist(l_indx,:))
hold on;
end
r = min(Np, L*Nt);
vline(r+1, '--', ['min(N_p, L N_T)=N_p=', num2str(min(Np, L*Nt))]);
legend('L=1', 'L=4', 'L=8')
grid on;
ylabel('Singular value');
xlabel('Index')
title(strcat('N_r=', num2str(Nr),', M_r^e=', num2str(Mr_e),',N_p=', num2str(Np)))

%% 
Nt = 4;
Nr = 128;
Mr_e = 32;
Mr = 4;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1 4 8];
T = 50;

Nr_min = min(Nr, Mr_e);
eig_dist = zeros(length(L_range), Nr_min);

 for l_indx=1:length(L_range)
  L = L_range(l_indx);
  [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols
   for k=1:Nt
    % Gaussian random symbols
 %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end

  N = zeros(Nr, T);

  W = createBeamformer(Nr, 'ZC');
  [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);

  A = W_tilde'*Dr;
  B = zeros(L*Nt, T);
  for l=1:L
     B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
  end

  [U,S,V]=svd(Y);
  eig_dist(l_indx, :) = diag(S(1:Nr_min, 1:Nr_min));
 end

subplot(2, 3, 3)
for l_indx=1:length(L_range)
plot(eig_dist(l_indx,:))
hold on;
end
r = min(Np, L*Nt);
vline(r+1, '--', ['min(N_p, L N_T)=N_p=', num2str(min(Np, L*Nt))]);
legend('L=1', 'L=4', 'L=8')
grid on;
ylabel('Singular value');
xlabel('Index')
title(strcat('N_r=', num2str(Nr),', M_r^e=', num2str(Mr_e),',N_p=', num2str(Np)))


%% 
Nt = 4;
Nr = 32;
Mr_e = 32;
Mr = 4;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 3;
total_num_of_rays = 12;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1 4 8];
T = 50;

Nr_min = min(Nr, Mr_e);
eig_dist = zeros(length(L_range), Nr_min);

 for l_indx=1:length(L_range)
  L = L_range(l_indx);
  [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols
   for k=1:Nt
    % Gaussian random symbols
 %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end

  N = zeros(Nr, T);

  W = createBeamformer(Nr, 'ZC');
  [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);

  A = W_tilde'*Dr;
  B = zeros(L*Nt, T);
  for l=1:L
     B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
  end

  [U,S,V]=svd(Y);
  eig_dist(l_indx, :) = diag(S(1:Nr_min, 1:Nr_min));
 end

subplot(2, 3, 4)
for l_indx=1:length(L_range)
plot(eig_dist(l_indx,:))
r = min(Np, L_range(l_indx)*Nt);
vline(r+1, '--', ['min(N_p, L N_T)=', num2str(min(Np, L_range(l_indx)*Nt))]);
hold on;
end
legend('L=1', 'L=4', 'L=8')
grid on;
ylabel('Singular value');
xlabel('Index')
title(strcat('N_r=', num2str(Nr),', M_r^e=', num2str(Mr_e),',N_p=', num2str(Np)))

%% 
Nt = 4;
Nr = 64;
Mr_e = 32;
Mr = 4;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 3;
total_num_of_rays = 12;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1 4 8];
T = 50;

Nr_min = min(Nr, Mr_e);
eig_dist = zeros(length(L_range), Nr_min);

 for l_indx=1:length(L_range)
  L = L_range(l_indx);
  [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols
   for k=1:Nt
    % Gaussian random symbols
 %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end

  N = zeros(Nr, T);

  W = createBeamformer(Nr, 'ZC');
  [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);

  A = W_tilde'*Dr;
  B = zeros(L*Nt, T);
  for l=1:L
     B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
  end

  [U,S,V]=svd(Y);
  eig_dist(l_indx, :) = diag(S(1:Nr_min, 1:Nr_min));
 end

subplot(2, 3, 5)
for l_indx=1:length(L_range)
plot(eig_dist(l_indx,:))
r = min(Np, L_range(l_indx)*Nt);
vline(r+1, '--', ['min(N_p, L N_T)=', num2str(min(Np, L_range(l_indx)*Nt))]);
hold on;
end
legend('L=1', 'L=4', 'L=8')
grid on;
ylabel('Singular value');
xlabel('Index')
title(strcat('N_r=', num2str(Nr),', M_r^e=', num2str(Mr_e),',N_p=', num2str(Np)))

%% 
Nt = 4;
Nr = 128;
Mr_e = 32;
Mr = 4;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 3;
total_num_of_rays = 12;
Np = total_num_of_clusters*total_num_of_rays;
L_range = [1 4 8];
T = 50;

Nr_min = min(Nr, Mr_e);
eig_dist = zeros(length(L_range), Nr_min);

 for l_indx=1:length(L_range)
  L = L_range(l_indx);
  [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
   Psi_i = zeros(T, T, Nt);
   % Generate the training symbols
   for k=1:Nt
    % Gaussian random symbols
 %     s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    % 4-QAM symbols
    s = qam4mod([], 'mod', T);
    Psi_i(:,:,k) =  toeplitz(s);
   end

  N = zeros(Nr, T);

  W = createBeamformer(Nr, 'ZC');
  [Y_proposed_hbf, W_tilde, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Mr_e, Mr, W);

  A = W_tilde'*Dr;
  B = zeros(L*Nt, T);
  for l=1:L
     B((l-1)*Nt+1:l*Nt, :) = Dt'*Psi_bar(:,:,l);
  end

  [U,S,V]=svd(Y);
  eig_dist(l_indx, :) = diag(S(1:Nr_min, 1:Nr_min));
 end

subplot(2, 3, 6)
for l_indx=1:length(L_range)
plot(eig_dist(l_indx,:))
r = min(Np, L_range(l_indx)*Nt);
vline(r+1, '--', ['min(N_p, L N_T)=', num2str(min(Np, L_range(l_indx)*Nt))]);
hold on;
end
legend('L=1', 'L=4', 'L=8')
grid on;
ylabel('Singular value');
xlabel('Index')
title(strcat('N_r=', num2str(Nr),', M_r^e=', num2str(Mr_e),',N_p=', num2str(Np)))