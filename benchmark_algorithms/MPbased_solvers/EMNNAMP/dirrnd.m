function theta = dirrnd(alpha,N)

k = length(alpha);
theta = zeros(N, k);
scale = 1; % arbitrary?
for i=1:k
  theta(:,i) = gamrnd(alpha(i), scale, N, 1);
  %theta(:,i) = sample_gamma(alpha(i), scale, N, 1);
end
%theta = mk_stochastic(theta);
S = sum(theta,2); 
theta = theta ./ repmat(S, 1, k);

return