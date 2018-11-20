function r = spud_subproblem( Y, w ) 

%%%
%
% solve the optimization problem 
%
%    minimize    || e ||_1 
%    subject to  e = Y' * r
%                1 = w' * r
%
%  This is equivalent to 
% 
%    min || r' * Y ||_1  st   w' * r = 1,
%
%  the problem advertised in our paper. 
%
%  John Wright, Summer '12. johnwright@ee.columbia.edu
%
%%%

% NOTE: this file uses an ADMM approach, which should be much more
% efficient than the previously taken generic approach. 

[m,n] = size(Y);

VERBOSE = 0;
MU      = 10;
TAU     = 1.025;
MAX_ITERATIONS = 1000;
EPS=1e-5;

% primal variables
r = w / norm(w,2);
e = Y' * r;

% dual variables
lambda = zeros(n,1);

rho  = 1.25; 
done = false;

% precomputations for r updates
G = inv( Y * Y' );
Y_dag = G * Y;
Gw = G * w;
eta = w' * Gw; 

iteration = 0; 

while ~done, 
    
    % update e 
    e = shrink( Y' * r - (1/rho) * lambda , 1/rho );
    
    % update r
    z = e + lambda / rho;
    r_prev = r; 
    q = Y_dag * z;
    
    xi   = w' * q; 
    zeta = (xi - 1) / eta;
    
    r = q - zeta * Gw;
        
    % update lagrange multiplier lambda
    lambda = lambda + rho * ( e - Y' * r);
    
    iteration = iteration + 1; 
    
    % update rho based
    s = Y' * ( rho * ( r - r_prev ) ); 
    norm_s = norm(s);
    
    infeas = Y' * r - e;
    norm_infeas = norm(infeas);
     
    if norm_infeas > MU * norm_s,
         rho = rho * TAU; 
    elseif norm_infeas < norm_s / MU, 
         rho = rho / TAU;
    end
    %rho = 1;
           
    if VERBOSE && mod(iteration,100) == 0,        
        disp(['Iteration ' num2str(iteration) '   obj ' num2str(sum(abs(e))) '  infeas ' num2str(norm_infeas) '  norm_s ' num2str(norm_s) '  rho ' num2str(rho) ])
        pause(.001);
    end    
    
    if iteration > MAX_ITERATIONS, 
        done = true;
    end
        
    if (norm_infeas<EPS)&&(norm_s<EPS)
        done=true;
    end
end