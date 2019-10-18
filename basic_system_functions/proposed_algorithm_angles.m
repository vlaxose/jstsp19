function [S, Y, convergence_error] = proposed_algorithm_angles(subY, Omega, indx_S, A, B, Imax, tau_Y, tau_S, rho, type, greedy_nnz)

  [N, M] = size(subY);
  Gr = size(A, 2);
  Gt = size(B, 1);
  convergence_error = zeros(Imax,3);
 
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
  iK1 = sparse(diag(1./diag(K1+2*rho*eye(N*M))));
  
  K2 = kron(B.', A);
  switch(type)
      case 'approximate'
        R = K2'*K2;
        L = size(R, 2);
        v = zeros(L, 1);
      otherwise
          [L,U] = lu(K2);
  end

  Omega_S = zeros(Gr, Gt);

  for i=1:Imax
  
      Omega_S(indx_S(1:min(10+5*i, Gt*Gr))) = 1;
      K3 = zeros(Gr*Gt);
      for ii=1:size(Omega_S, 1)
        Eii = zeros(size(Omega_S, 1));
        Eii(ii,ii) = 1;
        K3 = K3 + kron(diag(Omega_S(ii, :))', Eii);
      end
      K3 = sparse(K3);

    % sub 1
    Y = svt(X-1/rho*V1, tau_Y/rho);
    
    % sub 2
    b=(vec(V1) + rho*vec(Y) + vec(subY) + vec(V2) + rho*vec(C) + rho*K2*s);
    x = iK1*b;
    X = reshape(x, N, M);
    
    % sub 3
    k = (vec(X)-1/rho*vec(V2)-vec(C));
    
    switch(type)
        case 'approximate'
            res = K2'*k - R*v;
            alpha = res'*res/(res'*R*res);
            prev_v = v;            
            v = v + alpha*res;
            convergence_error(i, 3) = norm(prev_v-v)^2/norm(prev_v)^2;
        otherwise
            v = U\(L\k);
    end
    
    s = max(abs(real(v))-tau_S/rho,0).*sign(real(v)) +1j* max(abs(imag(v))-tau_S/rho,0).*sign(imag(v));
    s = K3*s;
    S = reshape(s, Gr, Gt);
    Xs = A*S*B;

    % sub 4    
    C = rho/(rho+1)*(X - Xs - V2/rho);
     
    % dual update
    V1 = V1 + rho*(Y-X);
    V2 = V2 + rho*(C - X + Xs);

    convergence_error(i, 1) = norm(V1)^2/norm(X)^2;
%     convergence_error(i, 1)
    convergence_error(i, 2) = norm(V2)^2/norm(X)^2;
  end


end
