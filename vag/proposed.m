function [S, Lt_opt, kappa, converged, rate_approximation_error] = proposed(H,W,F_RF,F_BB,Ns,Lt,maxDBiters, bits_vec, noise_variance, p_dac, Pt, Pcp, Pps, check_rate_approximation_flag)

  [Nr, Nt] = size(H);

  rate_approximation_error = 0;
    
  C_epsilon = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*bits_vec)).*sqrt(pi*sqrt(3)/2 .* 2.^(-2*bits_vec)));
  R_eta = (((W'*H*F_RF*C_epsilon*(F_BB*F_BB')*C_epsilon*F_RF'*H'*W)+noise_variance^2*(W'*W)));

  Delta = diag(sqrt(1-pi*sqrt(3)/2 * 2.^(-2*bits_vec)));
 
  [Ur,Er] = eig(R_eta);
  a = real(diag(1./diag(sqrt(Er))))*Ur*W'*H*F_RF*Delta;
  b = F_BB.';

  kappa = zeros(1, maxDBiters+1);
  indx = 1;
  % Dinkelbach iterations
  for iter=1:maxDBiters 

    % CVX solution
    cvx_begin quiet
      variable s(Lt);
      weighted_norm = 0;
      for i=1:Lt
        weighted_norm = weighted_norm + s(i)*sqrt(pi*sqrt(3)/(2*(1-Delta(i,i)^2)));
      end
      Q = zeros(Ns);
      for i=1:Lt
        Q = Q + s(i)*(a(:,i)'*a(:,i))*(b(:,i)*b(:,i)');
      end
      
      maximize(real(trace(Q)-kappa(iter)*weighted_norm))

      subject to
         sum(weighted_norm) <= 200;
         s >= 0;
         s <= 1;
         
    cvx_end

    % Thresholding to transform from real to binary
    s_indx = find(s/norm(s)>1e-6);
    Lt_opt = length(s_indx);
    S = zeros(Lt);
    S(s_indx(1:Lt_opt), s_indx(1:Lt_opt)) = eye(Lt_opt);

    % Check the solution
    if(isnan(cvx_optval) || isinf(cvx_optval) || sum(s<0) == length(s) || length(find(diag(S)>0)) == length(s))
      S = eye(Lt);
      converged = 0;
      Lt_opt = Lt;
    else
      converged = 1;
    end
    
    Q = zeros(Ns);
    for i=1:Lt
      Q = Q + S(i,i)*(a(:,i)'*a(:,i))*(b(:,i)*b(:,i)');
    end
    weighted_norm = 0;
    for i=1:Lt
      weighted_norm = weighted_norm + S(i,i)*sqrt(pi*sqrt(3)/(2*(1-Delta(i,i)^2)));

    end

    
    
%   WW = real(diag(1./diag(sqrt(Er))))*Ur*W'*H*F_RF*Delta*S*(F_BB*F_BB')*S*Delta'*F_RF'*H'*W*Ur'*real(diag(1./diag(sqrt(Er))));
%   WW2 = a*S*(F_BB*F_BB')*S*a';
%   norm(WW-WW2)
%   norm(WW-Q)
    
%          rate1(indx) = real(log2(1+1/Ns*trace(Q)))
         
%         log2(real(det(eye(Ns) + 1/Ns*WW)))
%         real(log2(1 + 1/Ns*trace(WW)))

%          rate2(indx) = real(log2(det(eye(Ns) + 1/Ns*WW)))
%       
%       power1(iter) = weighted_norm
%       power2(iter) = 2*trace(S*diag(2.^bits_vec))*p_dac + Nt*Pt + Nt*Lt_opt*Pps + Pcp
%       

    kappa(indx+1) = log2(1+1/Ns*trace(Q))/weighted_norm;

    indx = indx +1;
      
    % Validate Proposition 1
    if(check_rate_approximation_flag)
        Heff = Ur*W'*H*F_RF*Delta*S*F_BB;
        A = Ur*W'*H*F_RF*Delta;
        B = F_BB.';

    %     C1 = Heff;
    %     C2 = 0;
    %     for i=1:Lt
    %       C2 = C2 + S(i,i)*A(:,i)*B(:,i).';
    %     end
    %     norm(C1-C2)

        % Validate proposition
        B = F_BB.';
        Rate_original_expr = real(diag(1./diag(sqrt(Er))))*(Heff*Heff')*real(diag(1./diag(sqrt(Er))));

        A = real(diag(1./diag(sqrt(Er))))*Ur*W'*H*F_RF*Delta;        
        C2 = 0;
        C3 = 0;
        for i=1:Lt
            for j=1:Lt
    %             B(:,i).'*conj(B(:,j))
                C2 = C2 + S(i,i)*S(j,j)*A(:,i)*B(:,i).'*conj(B(:,j))*A(:,j)';
                if(i ~= j)
                    C3 = C3 + S(i,i)*S(j,j)*A(:,i)*B(:,i).'*conj(B(:,j))*A(:,j)';
                end
            end
        end

        % Rate based on Proposition 1
        Rate_proposition1 = 0;
        for i=1:Lt
            Rate_proposition1 = Rate_proposition1 + S(i,i)*A(:,i)*B(:,i).'*conj(B(:,i))*A(:,i)';
        end

        rate_approximation_error = norm(Rate_original_expr-Rate_proposition1, 'fro')^2/norm(Rate_original_expr, 'fro')^2;
     end
    
  end



end
