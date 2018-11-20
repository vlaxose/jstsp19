function [Y, Abar, Zbar, W] = wideband_hybBF_comm_system_training(H, Dr, Dt, T, snr)

   [Nr, Nt, L] = size(H);

   Y = zeros(Nr, T);

   Psi = zeros(T, T, Nt);
   for k=1:Nt
    Psi(:,:,k) =  toeplitz(randn(1, T)+1j*randn(1, T));

    Hl=[];
    for l=1:L
      Hl = [Hl H(:,k,l)];
    end
    Y = Y + Hl*Psi(1:L,:,k);

   end

   W = 1/sqrt(Nr)*fft(eye(Nr));

   Y = W'*Y + sqrt(snr/2)*(randn(Nr, T) + 1j*randn(Nr,T));
   Abar = [];
   Zbar = [];
   for l=1:L
     Psi_l = [];
     Zl = Dr'*H(:,:,l)*Dt;
     Zbar = [Zbar Zl];
     for k=1:Nt
       Psi_l = [Psi_l ; Psi(l, :, k)];
     end

     Abar = [Abar ; Dt'*Psi_l];
   end


end
