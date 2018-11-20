function [x_hat, totalIterations, relativeResidual] = minimize_l1_lc(A,b,lambda,solver)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Solves the optimization problem
%
%     minimize ||x||_1  subject  Ax = b    (1)
%
%   Currently assumes that Ay = b, so the feasible set is nonempty. Uses
%   Bregman iteration / ALM together with an iterative thresholding
%   algorithm for solving the inner iterations. 
%
%  THIS VERSION ASSUMES THE OPERATOR A IS SQUARE AND IDEMPOTENT, I.E.,
%    A IS THE ORTHOPROJECTOR ONTO SOME SUBSPACE S <= Re^m  
%
%   Inputs:
%      A      - m x n matrix, or function handle
%      At     - function handle implementing the action of A', or [] if A
%                is explicit
%      b      - m x 1 observation vector
%      n      - number of unknowns
%      lambda - optional scalar parameter. The Bregman iteration solves 
%                a sequence of subproblems of the form 
%  
%                minimize  lambda ||x||_1 + \lambda || Ax - r ||^2 / 2  (2)
%
%      solver - Technique used to solve the subproblem (2). Current options are
%
%                'FISTA' - default, accelerated proximal point algorithm of Beck and Teboulle                
%                'ISTA'  - iterative soft thresholding (proximal point algorithm)
%
%   Outputs:
%      x_hat     - solution
%
%   Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE                         = 0;
MAX_ITERATIONS                  = 2500;
DEFAULT_LAMBDA                  = 50; 
MIN_INNER_IT_TO_DECREASE_LAMBDA = 400;
TERMINATE_IF_NONMONOTONE        = 1; 
RELATIVE_RESIDUAL_THRESHOLD     = 0;

if nargin < 6,
    lambda = DEFAULT_LAMBDA;
end

if nargin < 7,
    solver = 'FISTA';
end

implicitMode = isa(A,'function_handle');

m = length(b);
n = m;
r = zeros(m,1);
x = zeros(n,1);

converged = false;
prevResidual = inf; 
minInnerItReached = false;
numIterations = 0;
totalIterations = 0;

if implicitMode, 
    delta = b - A(x);
else
    delta = b - A * x;
end
normb = norm(b);

while ~converged,
    numIterations = numIterations + 1;
    
    if implicitMode,
        r = r + delta;
    else
        r = r + delta;
    end
 
     if strcmp(solver,'FISTA'),
         [xNew, it] = minimize_l1_l2_FISTA(A,x,r,lambda);
     elseif strcmp(solver,'ISTA'),
         error('SORRY, ISTA UNSUPPORTED IN THIS VERSION');
     else
         error(['Unknown solver in minimize_l1_lc_qc.m! : User specifed ' solver]);
     end
            
    totalIterations = totalIterations + it;
    
    stepSize = norm(xNew-x);
    x = xNew;
    
    if implicitMode,
        delta = b - A(x);
    else
        delta = A' * (b - A*x);
    end    
    
    residual = norm(delta);
    
    if VERBOSE,
        disp(['Bregman iteration ' num2str(numIterations) '  Total it: ' num2str(totalIterations) '  Objective ' num2str(sum(abs(x))) '  Residual ' num2str(residual) ]);
    end
    
    if totalIterations >= MAX_ITERATIONS || residual < 1e-10, % || (relativeResidual < RELATIVE_RESIDUAL_THRESHOLD) || (residual >= prevResidual && TERMINATE_IF_NONMONOTONE)      
        converged = true;
    end
    
    %pause;
    
    prevResidual = residual;
end

x_hat = x;
