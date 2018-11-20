function rex = BAMP_GM_simple(y, A, sigmaL, sigmaS, sigw, lam, tol, ampiter)
% BAMP decoder for the two-state Gaussian mixture data 
% it's a clear and simplified version of myAMP_GM.m
% by Chunli 12/11/2013

[M, N]=size(A);
alpha = M/N;
mu=zeros(N,1);
x_hat_old = mu;
z=y;
c=100*(lam*sigmaL+(1-lam)*sigmaS+sigw);
% c=1;
k=0;

while(1)
    theta = A'*z + mu;
    mu = F_GM(theta, c, sigmaL, sigmaS, lam);
    x_hat = mu;
    
   
    if (length(find(isnan(x_hat))) ~= 0 )
        x_hat = 0.1*ones(N,1);
    end
    
    mmse =c*sum(dF_GM(theta, c, sigmaL, sigmaS, lam))/N;
    
    z = y - A*mu + z*mmse/c/alpha;
    c = sigw + mmse/alpha;
    
    k=k+1;
    diff = sum((x_hat_old-x_hat).^2)/N;
    x_hat_old = x_hat;
    
    if(diff<tol)
        break;
    end
    if(k>ampiter)
        break;
    end
    % disp([k, diff]);
end
rex=x_hat;    
 