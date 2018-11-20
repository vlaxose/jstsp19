function [errVal,X] = coding_error(Y,A,K,useTST)

%coding_error uses OMP (or TST) to code Y as Y = AX with at most K
%non-zeros per column. errVal is the normalized coding error. The
%dictionary is normalized to have unit norm columns prior to coding.

if nargin < 4
    useTST = 0;
end


%Get sizes
[~,N] = size(A);
[~,L] = size(Y);

%Ensure dictionary is normalized
%Normalize the columns
A = A*diag(1 ./ sqrt(abs(diag(A'*A))));

if ~useTST
    
    %%%%%Use OMP
    X = omp(A'*Y,A'*A,K);
    
else
    %%%%%%Use TST
    %Optionally can use TST to perform the coding. This is much slower, but
    %often performs better than OMP for a fixed dictionary, particularly if
    %K is large
    X = zeros(N,L);
    for ll = 1:L
        X(:,ll) = RecommendedTST(A,Y(:,ll), 300,1e-5,zeros(N,1),K/N);
    end
    
end

%Compute the error
errVal = norm(A*X - Y,'fro')/norm(Y,'fro');

