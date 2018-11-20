function [Uout] = active_subspace_exp(grad, ortho_UX_i, k, svdMode, svdApprox)

if isempty(ortho_UX_i)   
    switch svdMode
        case 'propack'
            if (svdApprox)
                Ak = LowRankApproximation(grad, k, svdApprox);
                [Uout, ~, ~] = lansvd(grad, k, 'L'); 
            else
                [Uout, ~, ~] = lansvd(grad, k, 'L');            
            end;
        case 'svds'
            if (svdApprox)
                Ak = LowRankApproximation(grad, k, svdApprox);
                [Uout, ~, ~] = svd(Ak);
            else
                [Uout, ~, ~] = svds(grad, k, 'L');
            end;
        otherwise
            if (svdApprox)
                Ak = LowRankApproximation(grad, k, svdApprox);
                [Uout, ~, ~] = svd(Ak);
            else
                [U, ~, ~] = svd(grad);
                Uout = U(:,1:k);
            end;
    end    
else
    switch svdMode
        case 'propack'
            if (svdApprox)
                Ak = LowRankApproximation(ortho_UX_i*grad, k, svdApprox);
                [Uout, ~, ~] = lansvd(ortho_UX_i*grad, k, 'L');     
            else
                [Uout, ~, ~] = lansvd(ortho_UX_i*grad, k, 'L');            
            end;
        case 'svds'
            if (svdApprox)
                Ak = LowRankApproximation(ortho_UX_i*grad, k, svdApprox);
                [Uout, ~, ~] = svd(Ak);
            else
                [Uout, ~, ~] = svds(ortho_UX_i*grad, k, 'L');
            end;
        otherwise
            if (svdApprox)
                Ak = LowRankApproximation(ortho_UX_i*grad, k, svdApprox);
                [Uout, ~, ~] = svd(Ak);
            else
                [U, ~, ~] = svd(ortho_UX_i*grad);
                Uout = U(:,1:k);
            end;
    end  
end;