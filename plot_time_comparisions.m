clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt = 4;
Nr_range = [32 64];

total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;
snr = 5;
square_noise_variance = 10^(-snr/10);
subSamplingRatio = 0.75;
maxMCRealizations = 1;
T = 70;
Imax = 100;

%% Variables initialization
elapsed_time_proposed = zeros(length(Nr_range), 1);
elapsed_time_proposed_approximate = zeros(length(Nr_range), 1);
elapsed_time_svt = zeros(length(Nr_range), 1);
elapsed_time_vamp = zeros(length(Nr_range), 1);
elapsed_time_omp = zeros(length(Nr_range), 1);

%% Iterations for different SNRs, training length and MC realizations
for nr_indx = 1:length(Nr_range)
  Nr = Nr_range(nr_indx);
  Gr = Nr;
  Gt = Nt;  

  for sub_indx=1:length(subSamplingRatio)
   
     

    [H,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
    [Y_proposed_hbf, Y_conventional_hbf, W_tilde, Psi_bar, Omega, Lr] = wideband_hybBF_comm_system_training(H, T, square_noise_variance, subSamplingRatio, Gr);
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

    tic;
    [~, Y_proposed] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho, 'normal');
    elapsed_time_proposed(nr_indx) = toc;

    tic;
    [~, Y_proposed_approximate] = proposed_algorithm(Y_proposed_hbf, Omega, A, B, Imax, tau_X, tau_S, rho, 'approximate');
    elapsed_time_proposed_approximate(nr_indx) = toc;
  
%     disp('Running Two-stage-based Technique..');
    tic;
    Y_twostage = mc_svt(Y_proposed_hbf, Omega, Imax,  tau_X, 0.1);
    elapsed_time_svt(nr_indx) = toc;
    
%     disp('Running VAMP...');
    tic;
    Phi = kron(B.', A);
    y = vec(Y_conventional_hbf);
    s_vamp = vamp(y, Phi, square_noise_variance, numOfnz);
    elapsed_time_vamp(nr_indx) = toc;
       
    
%     disp('Running OMP...');
    tic;
    s_omp = OMP(Phi, y, numOfnz, square_noise_variance);
    elapsed_time_omp(nr_indx) = toc;
  

  end

end

elapsed_time_proposed
elapsed_time_proposed_approximate
elapsed_time_svt
elapsed_time_vamp
elapsed_time_omp