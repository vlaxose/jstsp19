function [G_TRUE] = genMRF3D (obj, TBobj, N, NGibbsIter)

    % Extract MRF parameters from MarkovField object
    T = N;      % # timesteps = spatial dimension
    PRIOR_VAR = TBobj.resize(obj.marginal_var, N^2, 1, T);
    BETA = TBobj.resize(obj.beta, N^2, 1, T);
    GAMMA = TBobj.resize(obj.gamma, N^2, 1, T);
    
    % Expand certain matrix indexing arrays along the third
    % (temporal) dimension, accounting for shifts in linearly
    % indexed quantities accordingly
%     NbrListRep = NaN(N,maxDeg,T);
%     RevNbrListRep = NaN(N,maxDeg,T);
%     for t = 1:T
%         RevNbrListRep(:,:,t) = (t - 1)*N*maxDeg + obj.RevNbrList;
%         NbrListRep(:,:,t) = (t - 1)*N + NbrList;
%     end
    dummyMaskRep = repmat(obj.dummyMask, [1, 1, T]);
    
    % Initialize diagonal elements of the precision matrix
    ALPHA = (1 - (BETA .* sum(dummyMaskRep, 2) + ...
        2*GAMMA) .* PRIOR_VAR) ./ PRIOR_VAR;
    ALPHA(:,1,1) = ALPHA(:,1,1) + GAMMA(:,1,1);     % Correction for t = 1
    ALPHA(:,1,T) = ALPHA(:,1,T) + GAMMA(:,1,T);     % Correction for t = T
    assert(all(ALPHA(:) >= 0), ['Invalid parameterization. ' ...
        'Decrease beta, gamma, or maginal_var'])
    DiagPrecision = ALPHA + BETA .* sum(dummyMaskRep, 2) + ...
        2 * GAMMA;
    DiagPrecision(:,1,1) = DiagPrecision(:,1,1) - GAMMA(:,1,1);
    DiagPrecision(:,1,T) = DiagPrecision(:,1,T) - GAMMA(:,1,T);
    
    % Now, construct a sparse precision matrix for the vectorized
    % spatio-temporal GMRF amplitude matrix, Q
    SpatialPrecision = -obj.beta * obj.AdjMtx;
%     SPcell = num2cell(repmat(SpatialPrecision, [1, 1, T]), [1, 2]);
%     Q = blkdiag(SPcell{:});     % Populate duplicates along T blockdiagonals
    Q = spalloc(N^3, N^3, sum(obj.AdjMtx(:)) + 3*N^3);
%     rowIdx = repmat(1:N^2, 1, N^2)';
%     colIdx = reshape(repmat(1:N^2, N^2, 1), N^4, 1);
    for t = 1:T
        Q((1:N^2)+(t-1)*N^2,(1:N^2)+(t-1)*N^2) = SpatialPrecision;
    end
    % Replace main diagonal with DiagPrecision
    Q(1:N^3+1:N^6) = DiagPrecision(:);
    Q = Q + spdiags(-obj.gamma*ones(N^2*T,1), N^2+1, N^3, N^3) + ...
        spdiags(-obj.gamma*ones(N^2*T,1), -N^2-1, N^3, N^3);  % Add temporal
    
    % Lastly, create realization
    L = chol(Q);
    G_TRUE = L' \ randn(N^3, 1);
end
