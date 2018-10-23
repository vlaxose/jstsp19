function [converged, rate, ee, Lt_opt, S_proposed, kappa, S_bf, rate_approximation_error] = systemModel(Nt, Nr, Lt, Ns, total_num_of_clusters, total_num_of_rays, transmit_snr, maxDBiters, check_rate_approximation_flag)

    %% Setup
    K = 6;
    rate = zeros(1, K);
    ee = zeros(1, K);
    power = zeros(1, K);
    p_dac = 0.1;
    Pcp = 10;
    Pps = 10*10^(-3);
    Pt = 100*10^(-3);
    noise_variance = sqrt(10^(-transmit_snr/10));
    
    %% mmWave channel model
    [H, ~, At] = parametric_mmwave_channel(Nr, Nt, total_num_of_clusters, total_num_of_rays);
    
    %% Optimal beamforming
    [~,F_BB,F_RF,Fopt,W] = beamformer(H, At, Lt, Ns, 'fft_codebook');

    %% Digital beamforming with high resolution DACs (8bits) and no power constraint
    rate(:, 1) = real(log2(det(eye(Ns) + 1/Ns* inv(noise_variance^2*(W'*W))*W'*H*(Fopt*Fopt')*H'*W)));
    power(:, 1) = (2*Nt*2^8*p_dac) + Nt*Pt + Nt^2*Pps + Pcp;
    ee(:, 1) = rate(:, 1) / power(:, 1);

    %% Hybrid beamforming

    max_bit=8;
    
    % 1-bit case
    Lt_opt(:, 2) = Lt;
    bit_vec = ones(Lt_opt(:, 2),1);
    Delta = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*bit_vec)));
    [ee(:,2), rate(:,2)] = ee_computation(eye(Lt_opt(:, 2)), noise_variance, bit_vec, Lt_opt(:, 2), Ns, W, H, F_RF, F_BB, Delta, p_dac, Pt, Pps, Pcp);

    % max-bit case
    Lt_opt(:, 3) = Lt;
    three_bits_vec = max_bit*ones(Lt_opt(:, 3),1);
    Delta = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*three_bits_vec)));
    [ee(:,3), rate(:,3)] = ee_computation(eye(Lt_opt(:, 3)), noise_variance, three_bits_vec, Lt_opt(:, 3), Ns, W, H, F_RF, F_BB, Delta,  p_dac, Pt, Pps, Pcp);


    % Brute-force
    ee_prop = 0;
    for bit=1:max_bit
        ee_prop_prev = ee_prop;
        bit_vec = bit*ones(Lt, 1);
        [S_prop, Lt_opt_prop, kappa, converged, rate_approximation_error] = proposed(H,W,F_RF,F_BB,Ns,Lt,maxDBiters, bit_vec, noise_variance, p_dac, Pt, Pcp, Pps, check_rate_approximation_flag);
        [ee_prop, rate_prop] = ee_computation(S_prop, noise_variance, bit_vec, Lt_opt_prop, Ns, W, H, F_RF, F_BB, Delta, p_dac, Pt, Pps, Pcp);
        if ee_prop>ee_prop_prev
          ee(:,4) = ee_prop;
          Lt_opt(:,4) = Lt_opt_prop;
          rate(:,4) = rate_prop;
          S_proposed = S_prop;
        end
    end

    lt_opt_per_bit = zeros(max_bit, 1);
    for bit=1:max_bit
      ee_bit_bf=0;
      for Lt_var = Ns+1:Lt
          ee_bit_bf_prev = ee_bit_bf;
          bit_vec = zeros(Lt_var, 1);
          bit_vec(1:Lt_var) = ones(Lt_var,1);
          S_bit_bf = diag(bit_vec);
          Delta = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*bit*bit_vec)));
          [~,F_BB,F_RF,Fopt,W] = beamformer(H, At, Lt_var, Ns, 'fft_codebook');
          ee_bit_bf = ee_computation(S_bit_bf, noise_variance, bit*bit_vec, Lt_var, Ns, W, H, F_RF, F_BB, Delta, p_dac, Pt, Pps, Pcp);
          if ee_bit_bf>ee_bit_bf_prev
              lt_opt_per_bit(bit) = Lt_var;
          end
      end
    end
    
    % Proposed
    ee_bit_bf=0;
    for bit=1:max_bit
          ee_bit_bf_prev = ee_bit_bf;
          bit_vec = zeros(Lt, 1);
          bit_vec(1:lt_opt_per_bit(bit)) = ones(lt_opt_per_bit(bit),1);
          S_bit_bf = diag(bit_vec);
          Delta = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*bit*bit_vec)));
          [ee_bit_bf, rate_bit_bf] = ee_computation(S_bit_bf, noise_variance, bit*bit_vec, lt_opt_per_bit(bit), Ns, W, H, F_RF, F_BB, Delta, p_dac, Pt, Pps, Pcp);
           if ee_bit_bf>ee_bit_bf_prev
             ee(:,5) = ee_bit_bf;
             Lt_opt(:,5) = lt_opt_per_bit(bit);
             rate(:,5) = rate_bit_bf;
             S_bf = S_bit_bf;
          end      
    end
    
    
    % Randomly selected RF chains given the optimal optimal number
    ee_bit_bf=0;
    for bit=1:max_bit
          ee_bit_bf_prev = ee_bit_bf;
          bit_vec = zeros(Lt, 1);
          bit_vec(1:lt_opt_per_bit(bit)) = ones(lt_opt_per_bit(bit),1);
          indx_rnd = randperm(Lt);
          indx = indx_rnd(1:Lt_opt(:, 5));
          s_random = zeros(Lt, 1);
          s_random(indx) = ones(Lt_opt(:,5), 1);
          S_random = diag(s_random);
          Delta = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*bit*bit_vec)));
          [ee_bit_bf, rate_bit_bf] = ee_computation(S_random, noise_variance, bit*bit_vec, lt_opt_per_bit(bit), Ns, W, H, F_RF, F_BB, Delta, p_dac, Pt, Pps, Pcp);
           if ee_bit_bf>ee_bit_bf_prev
             ee(:,6) = ee_bit_bf;
             Lt_opt(:,6) = lt_opt_per_bit(bit);
             rate(:,6) = rate_bit_bf;
             S_bf = S_bit_bf;
          end      
    end

    
end
