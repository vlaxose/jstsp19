function xhat = grpLasso(y,A,gam,grpInd,opt)
% grpLasso:  Performs the group Lasso via Newton's method.
%
% The group lasso minimizes
%    min_x (1/2)*||y-A*x||^2_2 + gam* \sum norm(x(Igroup(k)))
% where Igroup(k) is the elements in the k-th group.

% Get dimensions
[m,n] = size(A);

% Get options
nit = opt.nit; % number of iterations
prt = opt.prt; % verbose

Jiter = zeros(nit,2);
a = 4;
xhat = zeros(n,1);
P = A'*A;
Anorm = max(svd(A,'econ'));
P = P + 1e-4*Anorm^2*eye(n);
step = 0.1;
update = 1;

for it = 1:nit
    
    % Compute value, gradient and Hessian
    if (update)
        % Compute the residual
        e = y-A*xhat;
        
        % Evaluate the regularization term and its derivitives
        [f0,f0act,f1,f2] = regEval(xhat,a,grpInd);
        J0 = 0.5*sum(abs(e).^2) + gam*f0;
        J0act  = 0.5*sum(abs(e).^2) + gam*f0act;
        Jgrad = -A'*e + gam*f1;
        H = P + gam*f2;
        delx = H \ Jgrad;
    end
    
    % Record current value
    Jiter(it,:) = [J0act J0];
    
    % Compute step 
    dx = -step*delx;
    x1 = xhat + dx;
    dJest = Jgrad'*dx;
    
    % Evaluate new point
    e1 = y-A*x1;
    f0new = regEval(x1,a,grpInd);
    J1 = 0.5*sum(abs(e1).^2) + gam*f0new;
    
    % Check if condition passes
    if (J1-J0 < 0.5*dJest)
        if (prt)
            fprintf(1,'%d J=%12.4e %12.4e a=%5.1f step=%12.4e pass\n', it, Jiter(it,:), a, step);
        end
        step = min(1, step*2);
        xhat = x1;
        update = 1;
        
        % Test if a should be increased
        if (J1 > J0 - 0.01)
            a = min(64,2*a);
        end
        
    else
        if (prt)
            fprintf(1,'%d J=%12.4e %12.4e a=%5.1f step=%12.4e fail\n', it, Jiter(it,:), a, step);
        end
        step = step/2;
        update = 0;
    end
    
    
end

return;

% Compute change in cost function by removing component
xhat1 = xhat;
nit1 = 10;
J1 = zeros(nit1,1);
Asq = sum(abs(A).^2)';
Jnzold = zeros(n,1);

for it1 = 1:nit1
    % Measure cost function and save if current is lower than new
    J1(it1) = 0.5*sum(abs(y-A*xhat1).^2) + gam*sum(abs(xhat1));
    if (J1(it1) < J0act)
        xhat = xhat1;
        J0act = J1(it1);
    end
    
    % Exit if cost function is growing
    if (J1(it1) > 2*J0act)
        if (prt)
            fprintf(1,'%d J=%12.4e Optimal solution not found\n', it1, J1(it1));     
        end
        break;
    end
    
    % Generate new test point
    z = A'*e + xhat1.*Asq;
    Jnz = (abs(z) > gam);
    if (sum(Jnz) >= m-1)
        break;
    end
    
    if (all(Jnz == Jnzold))
        if (prt)
            fprintf(1,'%d J=%12.4e Optimal solution found\n', it1, J1(it1)); 
        end
        break;
    else
        Jnzold = Jnz;
    end
    I = find(Jnz);
    s = sign(z(I));
    AI = A(:,I);
    xhat1 = zeros(n,1);
    xhat1(I) = (AI'*AI) \ (AI'*y - s*gam);    
end


function [f0, f0act, f1, f2] = regEval(x,a,grpInd)
% Computes the regression term:  
%
%   f0(x)    = sum_{igrp} (1/a)*log(cosh(a*rsqrt(igrp))
%   f0act(x) = sum_{igrp} rsqrt(igrp)
%
% where rsqrt(igrp) = norm( x.*(grpInd == igrp) ), the norm of the 
% components of the vector x with group index igrp.
%
% The method returns f0, the function value, f1 the gradient and f2 the
% Hessian.

% Get dimensions and initialize vectors
ngrp = max(grpInd);
nx = length(x);
f0 = 0;
f0act = 0;
if (nargout >= 3)
    f1 = zeros(nx,1);
    f2 = zeros(nx);
end

% Loop over groups
for igrp = 1:ngrp
    
    % Compute the norm in the group
    I = find(grpInd == igrp);
    xgrp = x(I);
    n1 = length(I);
    r = max(1e-6,xgrp'*xgrp);
    rsqrt = sqrt(r);
    
    % Compute the value
    f0 = f0 + (1/a)*log(cosh(a*rsqrt));
    f0act = f0act + rsqrt;
    
    if (nargout >= 3)
    
        % First and second derivitives of f with respect to r
        fr1 = tanh(a*rsqrt);    
        fr2 = 4*a./cosh(a*rsqrt);

        % First and second derivatives of f with respect to x
        f1(I) = fr1/rsqrt*xgrp;
        f2(I,I) = (fr2/r - fr1/(rsqrt^3))*(xgrp*xgrp') + fr1/rsqrt*eye(n1);
    end
        
end
    

