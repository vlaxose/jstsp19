function [S, convergence_error] = sparse_admm(Htrue, OH, Dr, Dt, Imax)

  [Mr, Mt] = size(OH);
  Gr = size(Dr, 2);
  Gt = size(Dt, 2);
  convergence_error = zeros(Imax,1);
  
  Z = zeros(Mr, Mt);
  R = zeros(Gr, Gt);

%   subRatio = length(find(Omega))/(Mr*Mt);
  rho = 0.01;%subRatio/norm(Htrue, 'fro')^2;
  tau_s = 0.0001;%0.001;%rho*norm(OH);

  A = kron(conj(Dt), Dr);
  B = A'*A - rho*eye(Mr*Mt);
  
  for i=1:Imax

    % sub 1
    v = vec(R + Z/rho);
    s = max(abs(real(v))-tau_s/(rho),0).*sign(real(v)) +1j* max(abs(imag(v))-tau_s/(rho),0).*sign(imag(v));
    S = reshape(s, Mr, Mt);
    
    % sub 2    
    r = B \ (vec(Z) - rho* s + A'*vec(OH));
    R = reshape(r, Mr, Mt);
    
    % dual update
    Z = Z + rho*(R-S);

    convergence_error(i) = norm(Dr*S*Dt'-Htrue)^2/norm(Htrue)^2;
  end
  

end
