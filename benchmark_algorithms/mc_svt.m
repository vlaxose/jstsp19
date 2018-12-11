function [X, convergence_error] = mc_svt(H, OH, Omega, Imax, tau, rho)

 [Mr, Mt] = size(H);
 convergence_error = zeros(Imax,1);

  Y = zeros(Mr, Mt);

  for i=1:Imax
    X = svt(Y, tau/rho);
    Y = Y + rho*(OH-Omega.*X);
    convergence_error(i) = norm(X-H)^2/norm(H)^2;
%     convergence_error(i)
  end

end


