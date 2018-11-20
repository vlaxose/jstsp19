function [S0] = ApproximateVolumeSampling(A, k)

S = [];
E = A;
c = 1;

[~, n] = size(E);

for j = 1:k    
    P = [zeros(n, 1) (1:n)'];
    Frob_E = norm(E, 'fro')^2;
    for i = 1:n
        P(i,1) = c*norm(E(:,i),2)^2/Frob_E;
    end;
    sortP = sortrows(P, 1);
    cumsumP = cumsum(sortP(:,1));
    sortP = [sortP cumsumP];
    
    p = rand;
    pick = p > sortP(:,3);
    iCol = min(find(pick == 0));
    S = [S round(sortP(iCol, 2))];
    
    [U, Sv, ~] = svd(A(:, S));
    Sv = diag(Sv); Sv = Sv(Sv ~= 0);
    ortho_S = U(:,length(Sv)+1:end)*U(:,length(Sv)+1:end)';
    E = ortho_S*E;
end;

S0 = S;