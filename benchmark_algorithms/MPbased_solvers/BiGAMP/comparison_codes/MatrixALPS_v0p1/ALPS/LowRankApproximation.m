function [Ak] = LowRankApproximation(A, k, e)

S0 = ApproximateVolumeSampling(A, k);
[US0, SS0, ~] = svd(A(:,S0));
SS0 = diag(SS0); SS0 = SS0(SS0 ~= 0);
ortho_S0 = US0(:,length(SS0)+1:end)*US0(:,length(SS0)+1:end)';
E0 = ortho_S0*A;

t = ceil((k+1)*log2(k+1));
s = 2*k*ones(1,t);
s(end) = 16*k/e;

Si = AdaptiveSampling(A, E0, S0, s, t);

S = union(Si, S0);

size(S)

[U, ~, ~] = svd(A(:, S));
Ak = U(:,1:k)*U(:,1:k)'*A;