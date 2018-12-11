clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

%% Parameter initialization
Nt = 4;
Nr = 32;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 5;
snr_range = 15;
subSamplingRatio = 0.7;
maxMCRealizations = 1;
T_range = [40];
Imax = 50;

%% Variables initialization
error_proposed = zeros(maxMCRealizations,1);
error_omp = zeros(maxMCRealizations,1);
error_vamp = zeros(maxMCRealizations,1);
error_twostage = zeros(maxMCRealizations,1);
mean_error_proposed = zeros(length(T_range), length(snr_range));
mean_error_omp =  zeros(length(T_range), length(snr_range));
mean_error_vamp =  zeros(length(T_range), length(snr_range));
mean_error_twostage =  zeros(length(T_range), length(snr_range));

%% Iterations for different SNRs, training length and MC realizations
for snr_indx = 1:length(snr_range)
  snr = 10^(-snr_range(snr_indx)/10);
  snr_db = snr_range(snr_indx);
  
  for t_indx=1:length(T_range)
   T = T_range(t_indx);

   for r=1:maxMCRealizations
   disp(['realization: ', num2str(r)]);

    [H,Ar,At] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays);
    Gr = Nr;
    Gt = Nt;
    Dr = 1/sqrt(Nr)*exp(-1j*(0:Nr-1)'*2*pi*(0:Gr-1)/Gr);
    Dt = 1/sqrt(Nt)*exp(-1j*(0:Nt-1)'*2*pi*(0:Gt-1)/Gt);
    [Y, Abar, Zbar, W] = wideband_hybBF_comm_system_training(H, Dr, Dt, T, snr);
    Mr = size(W'*Dr, 2);
    Mt = size(Abar, 1);

    % Random sub-sampling
    Omega = zeros(Nr, T);
    for t = 1:T
        indices = randperm(Nr);
        sT = round(subSamplingRatio*Nr);
        indices_sub = indices(1:sT);
        Omega(indices_sub, t) = ones(sT, 1);
    end
    OY = Omega.*Y;
    sT2 = round(subSamplingRatio*T);
    Phi = kron(Abar(:, 1:sT2).', W'*Dr);
    y = vec(Y(:,1:sT2));

    
    % Proposed
    disp('Running proposed technique...');
    tau_X = 1/norm(OY, 'fro')^2;
    tau_S = tau_X/2;
    eigvalues = eigs(Y'*Y);
    rho = sqrt(min(eigvalues)*(tau_X+tau_S)/2);
    [~, Y_proposed] = proposed_algorithm(OY, Omega, W'*Dr, Abar, Imax, tau_X, tau_S, rho, Y, Zbar);
    S_proposed = pinv(W'*Dr)*Y_proposed*pinv(Abar);
    error_proposed(r) = norm(S_proposed-Zbar)^2/norm(Zbar)^2
    if(error_proposed(r)>1)
        error_proposed(r)=1;
    end
  
    % Two-stage scheme matrix completion and sparse recovery
    disp('Running Two-stage-based Technique..');
    Y_twostage = mc_svt(Y, OY, Omega, Imax,  tau_X, 0.1);
    S_twostage = pinv(W'*Dr)*Y_twostage*pinv(Abar);    
    error_twostage(r) = norm(S_twostage-Zbar)^2/norm(Zbar)^2
    if(error_twostage(r)>1)
        error_twostage(r) = 1;
    end
    
    % VAMP sparse recovery
    disp('Running VAMP...');
    s_vamp = vamp(y, Phi+1e-6*eye(size(Phi)), snr, Imax);
    S_vamp = reshape(s_vamp, Mr, Mt);
    error_vamp(r) = norm(S_vamp-Zbar)^2/norm(Zbar)^2
    if(error_vamp(r)>1)
        error_vamp(r) = 1;
    end
       
    
    % Sparse channel estimation
    disp('Running OMP...');
    s_omp = OMP(Phi, y, Imax, snr);
    S_omp = reshape(s_omp, Mr, Mt);
    error_omp(r) = norm(S_omp-Zbar)^2/norm(Zbar)^2
    if(error_omp(r)>1)
        error_omp(r)=1;
    end
  


   end

    mean_error_proposed(t_indx, snr_indx) = mean(error_proposed);
    mean_error_omp(t_indx, snr_indx) = mean(error_omp);
    mean_error_vamp(t_indx, snr_indx) = mean(error_vamp);
    mean_error_twostage(t_indx, snr_indx) = mean(error_twostage);

  end

end


figure;
p11 = semilogy(T_range, (mean_error_omp(:, 1)));hold on;
set(p11,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', '>', 'MarkerSize', 6, 'Color', 'Black');
p12 = semilogy(T_range, (mean_error_vamp(:, 1)));hold on;
set(p12,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Blue', 'MarkerFaceColor', 'Blue', 'Marker', 'o', 'MarkerSize', 6, 'Color', 'Blue');
p13 = semilogy(T_range, (mean_error_twostage(:, 1)));hold on;
set(p13,'LineWidth',2, 'LineStyle', '--', 'MarkerEdgeColor', 'Black', 'MarkerFaceColor', 'Black', 'Marker', 's', 'MarkerSize', 6, 'Color', 'Black');
p14 = semilogy(T_range, (mean_error_proposed(:, 1)));hold on;
set(p14,'LineWidth',2, 'LineStyle', '-', 'MarkerEdgeColor', 'Green', 'MarkerFaceColor', 'Green', 'Marker', 'h', 'MarkerSize', 6, 'Color', 'Green');
 
legend({'TD-OMP [11]', 'VAMP [23]', 'TSSR [15]', 'Proposed'}, 'FontSize', 12, 'Location', 'Best');


xlabel('number of training blocks');
ylabel('NMSE (dB)')
grid on;set(gca,'FontSize',12);
 
savefig('results/errorVStraining.fig')
% save('results/errorVStraining.mat')
