function [X] = genMRF3D (obj, N1, N2, N3, NGibbsIter, verbose)

    % Extract MRF parameters from MarkovField object
    betax = obj.betax;
    betay = obj.betay;
    betaz = obj.betaz;
    lambda = obj.alpha;

    % N1 N2 are dimensions of the image
    N1 = N1 + 2;
    N2 = N2 + 2;
    N3 = N3 + 2;
    
    X=rand(N1,N2,N3)<lambda;
    
    
%     index=find(X==0);
    X=double(X);
%     X(index)=-1;
    X(X == 0) = -1;
    
    X(1,:,:) = 0;%rand(1,Length)<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(end,:,:) = 0;%rand(1,Length)<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(:,1,:) = 0;%rand(1,Length).'<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(:,end,:) = 0;%rand(1,Length).'<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(:,:,1) = 0;
    X(:,:,end) = 0;    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    [r1, c1, d1] = meshgrid(2:2:N1-1,2:2:N2-1,2:2:N3-1);
    [r2, c2, d2] = meshgrid(3:2:N1-1,3:2:N2-1,2:2:N3-1);
    [r3, c3, d3] = meshgrid(2:2:N1-1,3:2:N2-1,2:2:N3-1);
    [r4, c4, d4] = meshgrid(3:2:N1-1,2:2:N2-1,2:2:N3-1);
    [r5, c5, d5] = meshgrid(2:2:N1-1,2:2:N2-1,3:2:N3-1);
    [r6, c6, d6] = meshgrid(3:2:N1-1,3:2:N2-1,3:2:N3-1);
    [r7, c7, d7] = meshgrid(2:2:N1-1,3:2:N2-1,3:2:N3-1);
    [r8, c8, d8] = meshgrid(3:2:N1-1,2:2:N2-1,3:2:N3-1);    
    
    ind1 = r1(:) + (c1(:) - 1)*N1 + (d1(:) - 1)*N1*N2;
    ind2 = r2(:) + (c2(:) - 1)*N1 + (d2(:) - 1)*N1*N2;
    ind3 = r3(:) + (c3(:) - 1)*N1 + (d3(:) - 1)*N1*N2;
    ind4 = r4(:) + (c4(:) - 1)*N1 + (d4(:) - 1)*N1*N2;
    ind5 = r5(:) + (c5(:) - 1)*N1 + (d5(:) - 1)*N1*N2;
    ind6 = r6(:) + (c6(:) - 1)*N1 + (d6(:) - 1)*N1*N2;
    ind7 = r7(:) + (c7(:) - 1)*N1 + (d7(:) - 1)*N1*N2;
    ind8 = r8(:) + (c8(:) - 1)*N1 + (d8(:) - 1)*N1*N2;
    
    ind1_cm = ind1 - N1;
    ind2_cm = ind2 - N1;
    ind3_cm = ind3 - N1;
    ind4_cm = ind4 - N1;
    ind5_cm = ind5 - N1;
    ind6_cm = ind6 - N1;
    ind7_cm = ind7 - N1;
    ind8_cm = ind8 - N1;
    
    ind1_rm = ind1 - 1;
    ind2_rm = ind2 - 1;
    ind3_rm = ind3 - 1;
    ind4_rm = ind4 - 1;
    ind5_rm = ind5 - 1;
    ind6_rm = ind6 - 1;
    ind7_rm = ind7 - 1;
    ind8_rm = ind8 - 1;
    
    ind1_dm = ind1 - N1*N2;
    ind2_dm = ind2 - N1*N2;
    ind3_dm = ind3 - N1*N2;
    ind4_dm = ind4 - N1*N2;
    ind5_dm = ind5 - N1*N2;
    ind6_dm = ind6 - N1*N2;
    ind7_dm = ind7 - N1*N2;
    ind8_dm = ind8 - N1*N2;
    
    ind1_cp = ind1 + N1;
    ind2_cp = ind2 + N1;
    ind3_cp = ind3 + N1;
    ind4_cp = ind4 + N1;
    ind5_cp = ind5 + N1;
    ind6_cp = ind6 + N1;
    ind7_cp = ind7 + N1;
    ind8_cp = ind8 + N1;
    
    ind1_rp = ind1 + 1;
    ind2_rp = ind2 + 1;
    ind3_rp = ind3 + 1;
    ind4_rp = ind4 + 1;
    ind5_rp = ind5 + 1;
    ind6_rp = ind6 + 1;
    ind7_rp = ind7 + 1;
    ind8_rp = ind8 + 1;
    
    ind1_dp = ind1 + N1*N2;
    ind2_dp = ind2 + N1*N2;
    ind3_dp = ind3 + N1*N2;
    ind4_dp = ind4 + N1*N2;
    ind5_dp = ind5 + N1*N2;
    ind6_dp = ind6 + N1*N2;
    ind7_dp = ind7 + N1*N2;
    ind8_dp = ind8 + N1*N2;
    
    for n=1:NGibbsIter
        
        cpx2 = exp(-betax*(X(ind1_rm) + X(ind1_rp)) - ...
            betay*(X(ind1_cm) + X(ind1_cp)) - ...
            betaz*(X(ind1_dm) + X(ind1_dp)) - lambda);
        cpx1 = exp(betax*(X(ind1_rm) + X(ind1_rp)) + ...
            betay*(X(ind1_cm) + X(ind1_cp)) + ...
            betaz*(X(ind1_dm) + X(ind1_dp)) - lambda);
        X(ind1) = 2*(rand(size(ind1))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind2_rm) + X(ind2_rp)) - ...
            betay*(X(ind2_cm) + X(ind2_cp)) - ...
            betaz*(X(ind2_dm) + X(ind2_dp)) - lambda);
        cpx1 = exp(betax*(X(ind2_rm) + X(ind2_rp)) + ...
            betay*(X(ind2_cm) + X(ind2_cp)) + ...
            betaz*(X(ind2_dm) + X(ind2_dp)) - lambda);
        X(ind2) = 2*(rand(size(ind2))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind3_rm) + X(ind3_rp)) - ...
            betay*(X(ind3_cm) + X(ind3_cp)) - ...
            betaz*(X(ind3_dm) + X(ind3_dp)) - lambda);
        cpx1 = exp(betax*(X(ind3_rm) + X(ind3_rp)) + ...
            betay*(X(ind3_cm) + X(ind3_cp)) + ...
            betaz*(X(ind3_dm) + X(ind3_dp)) - lambda);
        X(ind3) = 2*(rand(size(ind3))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind4_rm) + X(ind4_rp)) - ...
            betay*(X(ind4_cm) + X(ind4_cp)) - ...
            betaz*(X(ind4_dm) + X(ind4_dp)) - lambda);
        cpx1 = exp(betax*(X(ind4_rm) + X(ind4_rp)) + ...
            betay*(X(ind4_cm) + X(ind4_cp)) + ...
            betaz*(X(ind4_dm) + X(ind4_dp)) - lambda);
        X(ind4) = 2*(rand(size(ind4))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind5_rm) + X(ind5_rp)) - ...
            betay*(X(ind5_cm) + X(ind5_cp)) - ...
            betaz*(X(ind5_dm) + X(ind5_dp)) - lambda);
        cpx1 = exp(betax*(X(ind5_rm) + X(ind5_rp)) + ...
            betay*(X(ind5_cm) + X(ind5_cp)) + ...
            betaz*(X(ind5_dm) + X(ind5_dp)) - lambda);
        X(ind5) = 2*(rand(size(ind5))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind6_rm) + X(ind6_rp)) - ...
            betay*(X(ind6_cm) + X(ind6_cp)) - ...
            betaz*(X(ind6_dm) + X(ind6_dp)) - lambda);
        cpx1 = exp(betax*(X(ind6_rm) + X(ind6_rp)) + ...
            betay*(X(ind6_cm) + X(ind6_cp)) + ...
            betaz*(X(ind6_dm) + X(ind6_dp)) - lambda);
        X(ind6) = 2*(rand(size(ind6))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind7_rm) + X(ind7_rp)) - ...
            betay*(X(ind7_cm) + X(ind7_cp)) - ...
            betaz*(X(ind7_dm) + X(ind7_dp)) - lambda);
        cpx1 = exp(betax*(X(ind7_rm) + X(ind7_rp)) + ...
            betay*(X(ind7_cm) + X(ind7_cp)) + ...
            betaz*(X(ind7_dm) + X(ind7_dp)) - lambda);
        X(ind7) = 2*(rand(size(ind7))  < cpx1./(cpx1 + cpx2)) - 1;
        
        cpx2 = exp(-betax*(X(ind8_rm) + X(ind8_rp)) - ...
            betay*(X(ind8_cm) + X(ind8_cp)) - ...
            betaz*(X(ind8_dm) + X(ind8_dp)) - lambda);
        cpx1 = exp(betax*(X(ind8_rm) + X(ind8_rp)) + ...
            betay*(X(ind8_cm) + X(ind8_cp)) + ...
            betaz*(X(ind8_dm) + X(ind8_dp)) - lambda);
        X(ind8) = 2*(rand(size(ind8))  < cpx1./(cpx1 + cpx2)) - 1;
        
        if verbose
            fprintf('Sparsity: %1.2f%%\n', 100*sum(X(:) == 1)/(N1*N2*N3));
        end
        
    end
    
    X = double(X > 0);
    
    X = X(2:end-1,2:end-1,2:end-1);
    
    X = X(:);
end
