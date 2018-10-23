function [X, convergence_error] = mc_admm(Htrue, OH, Omega, Imax, tau, rho)

  [Mr, Mt] = size(OH);
  convergence_error = zeros(Imax,1);
  
  X = zeros(Mr, Mt);
  Y = zeros(Mr, Mt);
  Z = zeros(Mr, Mt);


  A = zeros(Mr*Mt);
  for i=1:Mr
    Eii = zeros(Mr);
    Eii(i,i) = 1;
    A = A + kron(diag(Omega(i, :))', Eii);
  end
  A = A + rho*eye(Mr*Mt);
    
  
  for i=1:Imax

    X = svt(Y-1/rho *Z, tau/rho);

    y = A\(vec(OH) + vec(Z) + rho * vec(X));
    Y = reshape(y, Mr, Mt);
    Z = Z + rho*(X-Y);

    convergence_error(i) = norm(X-Htrue)^2/norm(Htrue)^2;
    

  end
  

end