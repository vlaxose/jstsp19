function [Y_proposed_hbf, Y_conventional_hbf, W_tilde, Psi_bar, Omega, Lr] = wideband_hybBF_comm_system_training(H, T, snr, subSamplingRatio, Gr)

   %% Parameter initialization
   [Nr, Nt, L] = size(H);
    
   %% Variables initialization
   Psi_i = zeros(T, T, Nt);
   Psi_bar = zeros(Nt, T, L);
   F = 1/sqrt(Nt)*fft(eye(Nt));
   W_tilde = 1/sqrt(Nr)*fft(eye(Nr));
%    W_tilde = 1/sqrt(Nr)*exp(1j*pi*cos(pi*[0:1:Nr-1]'/Nr)*[0:1:Nr-1]);

   %% Additive white Gaussian noise
   N = sqrt(snr/2)*(randn(Nr, T) + 1j*randn(Nr, T));
   
   % Generate the training symbols
   for k=1:Nt
    s = 1/sqrt(2)*(randn(1, T)+1j*randn(1, T));
    Psi_i(:,:,k) =  toeplitz(s);
   end

   %% Wideband channel modeling
   R = zeros(size(N));
   for l=1:L
    for k=1:Nt
     Psi_bar(k,:,l) = Psi_i(l,:,k);
    end
    R = R + H(:,:,l)*Psi_bar(:,:,l);
   end
   
   R = R + N;

% % %    % Alternative formulation
% % %    Y = N;
% % %    for k=1:Nt
% % %     Hl=[];
% % %     for l=1:L
% % %       Hl = [Hl H(:,k,l)];
% % %     end
% % %     Y = Y + Hl*Psi_i(1:L,:,k);
% % %    end
% % %    norm(Y-R)
   
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
   Y_conventional_hbf = W_tilde'*R;
   
end
