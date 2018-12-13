function [S, Y, convergence_error] = proposed_algorithm(subY, Omega, A, B, Imax, tau, tau_S, rho)

  [N, M] = size(subY);
  Gr = size(A, 2);
  Gt = size(B, 1);
  convergence_error = zeros(Imax,2);
 
  X = zeros(N, M);
  V1 = zeros(N, M);
  V2 = zeros(N, M);
  C = zeros(N, M);
  s = zeros(Gr*Gt, 1);

  K1 = zeros(N*M);
  for i=1:N
    Eii = zeros(N);
    Eii(i,i) = 1;
    K1 = K1 + kron(diag(Omega(i, :))', Eii);
  end
  
  K2 = kron(B.', A);

  [L,U] = lu(K2);

  for i=1:Imax

    % sub 1
    Y = svt(X-1/rho*V1, tau/rho);
    
    % sub 2
    x = (K1+2*rho*eye(N*M))\(vec(V1) + rho*vec(Y) + vec(subY) + vec(V2) + rho*vec(C) + rho*K2*s);
    X = reshape(x, N, M);
    
    % sub 3
    k = (vec(X)-1/rho*vec(V2)-vec(C));
    v = U\(L\k);
    s = max(abs(real(v))-tau_S/rho,0).*sign(real(v)) +1j* max(abs(imag(v))-tau_S/rho,0).*sign(imag(v));
    S = reshape(s, Gr, Gt);
    Xs = A*S*B;

    % sub 4    
    C = rho/(rho+1)*(X - Xs - V2/rho);
     
    % dual update
    V1 = V1 + rho*(Y-X);
    V2 = V2 + rho*(C - X + Xs);

    convergence_error(i, 1) = norm(V1)^2;
    convergence_error(i, 2) = norm(V2)^2;
  end


end
