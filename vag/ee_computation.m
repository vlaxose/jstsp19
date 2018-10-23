function [ee, rate, power] = ee_computation(S, noise_variance, bits_vec, Lt_opt, Ns, W, H, F_RF, F_BB, Delta, p_dac, Pt, Pps, Pcp)

    [Nr, Nt] = size(H);
    
    % Covariance matrix of the additive quantization noise for the linear
    % approximation model
    C_epsilon = S*diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*S*bits_vec)).*sqrt(pi*sqrt(3)/2 .* 2.^(-2*S*bits_vec)));
    
    % Noise covariance matrix
    
    R_eta = real(((W'*H*F_RF*C_epsilon*(F_BB*F_BB')*C_epsilon*F_RF'*H'*W)+noise_variance^2*(W'*W)));
    
    % Achievable information rate
    rate = real(log2(det(eye(Ns) + 1/Ns*pinv(R_eta)*W'*H*F_RF*Delta*S*(F_BB*F_BB')*S*Delta'*F_RF'*H'*W)));
    
    % Consumed power
    power = 2*trace(S*diag(2.^bits_vec))*p_dac + Nt*Pt + Nt*Lt_opt*Pps + Pcp;
    
    % Energy efficiency computation
    ee = rate/power;
end

%     C_epsilon_random = Smu*diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*Smu*random_bits_vec)).*sqrt(pi*sqrt(3)/2 .* 2.^(-2*Smu*random_bits_vec)));
%     R_eta_random = ((W'*H*F_RF*C_epsilon_random*(F_BB*F_BB')*C_epsilon_random*F_RF'*H'*W)/Lt_opt(:, 5)+noise_variance^2*(W'*W));
%     [ee(:,5), rate(:,5), power(:,5)] = rate_power_estimation(Ns, R_eta_random, W, H, F_RF, F_BB, Delta, random_bits_vec, p_dac, Pt, Lt_opt(:, 5), Pps, Pcp);
%     rate(:, 5) = log2(real(det(eye(Ns) + 1/Ns*inv(R_eta_random )*W'*H*F_RF*Delta*Smu*(F_BB*F_BB')*Smu'*Delta'*F_RF'*H'*W)));
%     power(:, 5) = 2*trace(Smu*diag(2.^random_bits_vec))*p_dac + Nt*Pt + Nt*Lt_opt(:, 5)*Pps + Pcp;
%     ee(:, 5) = rate(:, 5)/power(:, 5);