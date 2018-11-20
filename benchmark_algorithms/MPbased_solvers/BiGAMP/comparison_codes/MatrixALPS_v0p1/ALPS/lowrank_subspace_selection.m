function [Wi, UWi, VWi] = lowrank_subspace_selection(Vi, k, svdMode, svdApprox)

switch svdMode
    case 'propack'
        if (svdApprox)
            Ak = LowRankApproximation(Vi, k, svdApprox);
            [UWi, SWi, VWi] = lansvd(Ak, k, 'L');
            Wi = UWi*SWi*VWi';
        else
            [UWi, SWi, VWi] = lansvd(Vi, k, 'L');            
            Wi = UWi*SWi*VWi';
        end;
    case 'svds'
        if (svdApprox)
            Ak = LowRankApproximation(Vi, k, svdApprox);
            [UWi, SWi, VWi] = svd(Ak);
            Wi = UWi*SWi*VWi';
        else
            [UWi, SWi, VWi] = svds(Vi, k, 'L');
            Wi = UWi*SWi*VWi';
        end;
    otherwise
        if (svdApprox)
            Ak = LowRankApproximation(Vi, k, svdApprox);
            [UWi, SWi, VWi] = svd(Ak);
            Wi = UWi*SWi*VWi';
        else
            [U, S, V] = svd(Vi);
            UWi = U(:,1:k);
            VWi = V(:,1:k);
            Wi = UWi*S(1:k, 1:k)*VWi';        
        end;
end  