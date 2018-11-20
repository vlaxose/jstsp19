function [receivedSignal, M, OH, Omega] = get_measurements_at_RX(H, T, snr, B)

    % Varialbes initialization
    [Mr, Mt] = size(H);
    noise = sqrt(snr/2)*(randn(T, 1)+1j*randn(T, 1));
    receivedSignal = zeros(T, 1);
    M = zeros(T, Mr*Mt); % measurement matrix for all training instances
    OH = zeros(Mr,Mt);
    
    %% Conventional switching-based beamforming system
    for k=1:T
     
      % unit random training symbols
      s = 1;
      
      % uniform randomly generated values for the switches
      p = randsrc(Mt, 1, [-1 1 1j -1j]);
      q = randsrc(Mr, 1, [-1 1 1j -1j]);
      
      % based on the input/output model from the reference paper
      receivedSignal(k, :) = q'*H*p + noise(k);
    
      % collect the measurement vectors
      M(k, :) = kron(p.',q')*B;
    end


    %% Proposed random sub-sampling with OH=Omega.*H

    % Permute randomly the indicies of the matrix entries and select T of
    % them
    indices = randperm(Mr*Mt);
    indices_sub = indices(1:T);
    % Construct the \Omega sub-sampling matrix
    Omega = zeros(Mr, Mt);
    Omega(indices_sub) = ones(T, 1);
    
    % The following for loops represent the proposed training scheme, where
    % the TX and RX activate only one antenna element at each time.
    t=1;
    for k=1:Mr
      W = zeros(Mr,Mr);
      W(k,k) = 1;
      F = diag(Omega(k, :));
      for i=1:Mr
        for j=1:Mt
          if(abs(W(i,:)*H*F(:,j))>0)
              OH(i,j) = OH(i,j) + (W(i,:)*H*F(:,j) + noise(t));
              t = t + 1;
          end
        end
      end
    end

end