function [Y_proposed_hbf, W_e, Psi_bar, Omega, Y] = proposed_hbf(H, N, Psi_i, T, Lr_e, Lr, W)

   %% Parameter initialization
   [~, Nt, L] = size(H);

   
   %% Variables initialization
   Psi_bar = zeros(Nt, T, L);

   %% Wideband channel modeling
   W_e = W(:, 1:Lr_e);
   
   % Construct the received signal
   Y = zeros(size(N));
   for l=1:L
    for k=1:Nt
     Psi_bar(k,:,l) = Psi_i(l,:,k);
    end
    Y = Y + H(:,:,l)*Psi_bar(:,:,l);
   end
   
   R = Y + N;

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
   Omega = zeros(Lr_e, T);  
   for t = 1:T
    indices = randperm(Lr_e);
    indices_sub = indices(1:Lr);
    Omega(indices_sub, t) = ones(Lr, 1);
   end
   Y_proposed_hbf = Omega.*(W_e'*R);
   
end
