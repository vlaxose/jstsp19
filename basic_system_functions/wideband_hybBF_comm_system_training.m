function [Y_proposed_hbf, Y_conventional_hbf, W_tilde, Psi_bar, Omega, Lr] = wideband_hybBF_comm_system_training(H, T, snr, subSamplingRatio)

   %% Parameter initialization
   [Nr, Nt, L] = size(H);
   
   %% Variables initialization
   Psi_i = zeros(T, T, Nt);
   Psi_bar = zeros(Nt, T, L);
   W_tilde = 1/sqrt(Nr)*fft(eye(Nr));
   
   %% Additive white Gaussian noise
   N = sqrt(snr/2)*(randn(Nr, T) + 1j*randn(Nr, T));
   
   % Generate the training symbols
   for k=1:Nt
    Psi_i(:,:,k) =  toeplitz(randn(1, T)+1j*randn(1, T));
   end

   %% Wideband channel modeling
   R = N;
   for l=1:L
    for k=1:Nt
     Psi_bar(k,:,l) = Psi_i(l,:,k);
    end
    R = R + H(:,:,l)*Psi_bar(:,:,l);
   end
   
   %% Proposed HBF architecture
   Omega = zeros(Nr, T);  
   for t = 1:T
    indices = randperm(Nr);
    Lr = round(subSamplingRatio*Nr);
    indices_sub = indices(1:Lr);
    Omega(indices_sub, t) = ones(Lr, 1);
   end
   Y_proposed_hbf = Omega.*(W_tilde'*R);
   
   %% Conventional HBF architecture
   W = W_tilde(:, 1:Lr);
   Y_conventional_hbf = W'*R;
   
end
