function [X, convergence_error] = mc_svt(H, OH, Omega, Imax)

 [Mr, Mt] = size(H);
 convergence_error = zeros(Imax,1);

  Y = zeros(Mr, Mt);
  subRatio = length(find(Omega))/(Mr*Mt);
  rho = 3*subRatio;
  tau = rho*norm(OH);

  for i=1:Imax
    X = svt(Y, tau);
    Y = Y + rho*(OH-Omega.*X);
    convergence_error(i) = norm(X-H)^2/norm(H)^2;
  end

end


