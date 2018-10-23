function [S, Y, convergence_error] = mcsi_admm(subY, Omega, A, B, Imax, tau, tau_S, rho, Yopt, Zbar)

  [N, M] = size(subY);
  Gr = size(A, 2);
  Gt = size(B, 1);
  convergence_error = zeros(Imax,1);
 
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

  for i=1:Imax

    % sub 1
    Y = svt(X-1/rho*V1, tau/rho);
%       norm(Y-Yopt, 'fro')^2/norm(Yopt, 'fro')^2
    
    % sub 2
    x = (K1+2*rho*eye(N*M))\(vec(V1) + rho*vec(Y) + vec(subY) + vec(V2) + rho*vec(C) + rho*K2*s);
    X = reshape(x, N, M);
    
    % sub 3
    k = (vec(X)-1/rho*vec(V2)-vec(C));
    K2_real = [real(K2) -imag(K2) ; imag(K2) real(K2)];
    k_real  = [real(k) ; imag(k)];

    cvx_begin quiet
      variable s_real(2*Gr*Gt)
      minimize(rho/2*norm(K2_real*s_real - k_real) + tau_S*norm(s_real,1))
    cvx_end

    s = s_real(1:Gr*Gt) + 1j*s_real(Gr*Gt+1:end);

    S = reshape(s, Gr, Gt);

    Xs = A*S*B;

    % sub 4    
    C = rho/(rho+1)*(X - Xs - V2/rho);
     
    % dual update
    V1 = V1 + rho*(Y-X);
    V2 = V2 + rho*(C - X + Xs);

     S_mcsi = pinv(A)*Y*pinv(B);
     norm(S_mcsi-Zbar)^2/norm(Zbar)^2
  end


end
