function [S] = AdaptiveSampling(A, E0, S0, s, t)

c = 1;
S = [];
Sj = [];
E = E0;
[~, n] = size(E0);

for j = 1:t
    P = [zeros(n, 1) (1:n)'];
    Frob_E = norm(E, 'fro')^2;
    for i = 1:n
        P(i,1) = c*norm(E(:,i),2)^2/Frob_E;
    end;
    sortP = sortrows(P, 1);
    cumsumP = cumsum(sortP(:,1));
    sortP = [sortP cumsumP];
    
    for k = 1:s(j)        
        p = rand; 
        pick = p > sortP(:,3);
        iCol = min(find(pick == 0));
        Sj = [Sj round(sortP(iCol, 2))];
    end;
    
    S = union(S, Sj);
    
    [U, Sv, ~] = svd(A(:, union(S0, S)));
    Sv = diag(Sv); Sv = Sv(Sv ~= 0);
    ortho_S = U(:,length(Sv)+1:end)*U(:,length(Sv)+1:end)';
    E = ortho_S*E;
    Sj = [];
end;