function [X] = mc_svt(OH, Omega, Imax, tau, rho)

 [Mr, Mt] = size(OH);

  Y = zeros(Mr, Mt);

  for i=1:Imax
    X = svt(Y, tau/rho);
    Y = Y + rho*(OH-Omega.*X);
  end

end


