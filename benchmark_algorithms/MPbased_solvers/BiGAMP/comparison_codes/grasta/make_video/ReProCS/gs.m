function V = gs(A)

% Gram-Schmidt orthogonalization of vectors stored in 
% columns of the matrix A. Orthonormalized vectors are 
% stored in columns of the matrix V

V = zeros(size(A));
n = size(A,2); 
for k=1:n 
    V(:,k) = A(:,k); 
    for j=1:k-1 
        R(j,k) = V(:,j)'*A(:,k); 
        V(:,k) = V(:,k) - R(j,k)*V(:,j);
    end
    R(k,k) = norm(V(:,k));
    V(:,k) = V(:,k)/R(k,k);
end