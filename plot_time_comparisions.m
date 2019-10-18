clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

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
maxMCRealizations = 1;
Mr = 4;
numOfnz = 5*20;
T = 35;
T_hbf = round(T/(Nr/Mr))*Nt;
T_prop = T*Nt;
  square_noise_variance = 10^(-5/10);
Imax = 100;

%% Variables initialization
elapsed_time_proposed = zeros(maxMCRealizations,1);
elapsed_time_proposed_angles = zeros(maxMCRealizations,1);
elapsed_time_ls = zeros(maxMCRealizations,1);
elapsed_time_omp_nr = zeros(maxMCRealizations,1);
elapsed_time_svt = zeros(maxMCRealizations,1);
elapsed_time_vamp = zeros(maxMCRealizations,1);
elapsed_time_cosamp = zeros(maxMCRealizations,1);
elapsed_time_omp_mmv = zeros(maxMCRealizations,1);
elapsed_time_tssr = zeros(maxMCRealizations,1);
mean_elapsed_time_proposed = 0;
mean_elapsed_time_proposed_angles = 0;
mean_elapsed_time_ls =  0;
mean_elapsed_time_omp_nr =  0;
mean_elapsed_time_svt =  0;
mean_elapsed_time_vamp =  0;
mean_elapsed_time_cosamp =  0;
mean_elapsed_time_omp_mmv =  0;
mean_elapsed_time_tssr =  0;

%% Iterations for different SNRs, training length and MC realizations
  parfor r=1:maxMCRealizations
      
   disp(['SNR = ', num2str(square_noise_variance), ', realization: ', num2str(r)]);

   
   %% System model
   [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);

   % Additive white Gaussian noise
   N = sqrt(square_noise_variance/2)*(randn(Nr, T_prop) + 1j*randn(Nr, T_prop));
   Psi_i = zeros(T_prop, T_prop, Nt);
   % Generate the training symbols
   for k=1:Nt
    % 4-QAM symbols
    s = qam4mod([], 'mod', T_prop);
    Psi_i(:,:,k) =  toeplitz(s);
   end

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
    tic;
    S_ls = pinv(A)*Y_hbf_nr*pinv(B);
    elapsed_time_ls(r) = toc;

    % OMP based
    s_omp_nr_solver = spx.pursuit.single.OrthogonalMatchingPursuit(Phi, numOfnz);
    s_omp_nr = s_omp_nr_solver.solve(y);
    S_omp_nr = reshape(s_omp_nr.z, Gr, L*Gt);
    elapsed_time_omp_nr(r) = toc;
   

    % VAMP-based with ZC and Nr RF chains
    tic;
    s_vamp = vamp(y, Phi, 1, numOfnz);
    S_vamp = reshape(s_vamp, Gr, L*Gt);
    elapsed_time_vamp(r) = toc;
    
    % CoSaMP-based with ZC and Nr RF chains
    s_cosamp = CoSaMP(Phi, y, numOfnz);
    S_cosamp = reshape(s_cosamp, Gr, L*Gt);
    elapsed_time_cosamp(r) = toc;

    % OMP with MMV based
    s_omp_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, numOfnz);
    S_omp_mmv = s_omp_solver.solve(Y_hbf_nr*pinv(B));
    elapsed_time_omp_mmv(r) = toc;
    
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
    tic;
    [S_proposed, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_Y, tau_Z, rho, 'approximate');
    elapsed_time_proposed(r) = toc;
  
    [~, indx_S] = sort(abs(vec(Zbar)), 'descend');
    tic;
    [S_proposed_angles, Y_proposed_angles] = proposed_algorithm_angles(Y_proposed_hbf, Omega, indx_S, A, B, Imax, tau_Y, tau_Z, rho, 'approximate', numOfnz);
    elapsed_time_proposed_angles(r) = toc;
    
%     
%     % SVT-based
%     Y_svt = mc_svt(Y_proposed, Omega, Imax, tau_Y, 0.1);
%     S_svt = pinv(A)*Y_svt*pinv(B);
%     elapsed_time_svt(r) = norm(S_svt-Zbar)^2/norm(Zbar)^2;
%     if(elapsed_time_svt(r)>1)
%         elapsed_time_svt(r) = 1;
%     end
%     
%     % TSSR-based
%     A = W_tilde'*Dr;
%     Phi = kron(B.', A);    
%     s_tssr_solver = spx.pursuit.joint.OrthogonalMatchingPursuit(A, 2*numOfnz);
%     S_tssr = s_tssr_solver.solve(Y_svt*pinv(B));
%     elapsed_time_tssr(r) = norm(S_tssr.Z-Zbar)^2/norm(Zbar)^2;
%     if(elapsed_time_tssr(r)>1)
%         elapsed_time_tssr(r) = 1;
%     end
  end
  
mean_elapsed_time_proposed = mean(elapsed_time_proposed);
mean_elapsed_time_proposed_angles = mean(elapsed_time_proposed_angles);
mean_elapsed_time_ls = mean(elapsed_time_ls);
mean_elapsed_time_omp_nr = mean(elapsed_time_omp_nr);
mean_elapsed_time_svt = mean(elapsed_time_svt);   
mean_elapsed_time_vamp = mean(elapsed_time_vamp);
mean_elapsed_time_cosamp = mean(elapsed_time_cosamp);
mean_elapsed_time_omp_mmv = mean(elapsed_time_omp_mmv);
mean_elapsed_time_tssr = mean(elapsed_time_tssr);

figure;
x = categorical({'LS', 'OMP', 'VAMP', 'CoSaMP', 'OMP-MMV', 'Proposed', 'Proposed - PAI'});
y = [mean_elapsed_time_ls mean_elapsed_time_omp_nr mean_elapsed_time_vamp mean_elapsed_time_cosamp mean_elapsed_time_omp_mmv mean_elapsed_time_proposed mean_elapsed_time_proposed_angles];
b = bar(x, y,'FaceColor',[0 0.4470 0.7410],'EdgeColor',[0 0 0],'LineWidth',2);
ylabel('Elapsed time (seconds)')