function [F,F_BB,F_RF,Fopt,W] = beamformer(H, At, Lt, Ns, type)

    %% Initialization
    [Nr, Nt] = size(H);
    F_RF  = eye(Nt, Lt);
    F_BB = zeros(Lt, Ns);
    
    %% Obtaion the optimal case, i.e., digital beamformer
    [U,~,V] = svd(H);
    Fopt = 1/sqrt(Ns)*V(:, 1:Ns);

    %%
    switch(type)
        case 'angular_codebook'
            D = At;
        case 'fft_codebook'
            D = 1/sqrt(Nt)*fft(eye(Nt));
    end

    maxLength = min(Lt, size(D,2));
    
    %%
    % TX
    if(Nt==Lt)
        F_RF = eye(Lt);
        F_BB = Fopt;
        F = F_RF*F_BB;
    else
        for indx=1:maxLength
          % Get the index with the maximum energy
          diff = Fopt - F_RF*F_BB;
          Psi = D'*diff/norm(diff, 'fro');
          C = diag(Psi*Psi');
          [~, I] = sort(C, 'descend');

          % Update the precoders
          F_RF(:, indx) = D(:,I(1));
          F_BB = F_RF\Fopt;
        end
        
        F = F_RF*F_BB;
        if(norm(Fopt - F)^2/norm(Fopt)^2>1e-1)
            warning(['A/D beamformer did not converge, with error:', num2str(norm(Fopt - F, 'fro'))]);
        end
    end
    
    % RX
    W =  1/sqrt(Ns)*U(:, 1:Ns);

        

end

