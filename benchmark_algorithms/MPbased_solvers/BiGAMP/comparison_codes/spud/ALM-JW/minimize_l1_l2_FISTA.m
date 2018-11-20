function [x_hat, numIterations] = minimize_l1_l2_FISTA(A,x0,b,lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Solves the optimization problem
% 
%    min          lambda ||x||_1 + ||Ax - b||_2^2/2
%
%  Using the FISTA algorithm of Beck and Teboulle. 
%
%  THIS VERSION ASSUMES THE OPERATOR A IS SQUARE AND IDEMPOTENT, I.E.,
%    A IS THE ORTHOPROJECTOR ONTO SOME SUBSPACE S <= Re^m  
%
%  Inputs:
%    w  - m x 1 nonnegative vector (can include zeros), defining the
%                weighted L1 norm
%    A  - m x m matrix, or function handle
%    x0 - m x 1 vector, initial guess
%    b  - m x 1 vector, observation
%
%  Outputs:
%    x_hat - n x 1 vector, optimal solution
%
%  Spring 2010, John Wright. Questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VERBOSE = 0;
MAX_ITERATIONS = 1000;
MIN_STEP = 5e-5;
delta    = 1;          % (1/delta = \bar{L} in Beck and Teboulle's notation)

implicitMode = isa(A,'function_handle');

m = length(b);
n = length(x0);

converged = false;
numIterations = 0;

x = x0;
xPrev = x0;
h = x0; 
tCur  = 1;
tPrev = 1;

clear('x0');

% functionValues = []; 

while ~converged,
    
    numIterations = numIterations + 1;
    
    % gradient step
    if implicitMode,
        g = A( h - b );
    else
        g = A * ( h - b );
    end
    
    % shrinkage step
    %
    %   minimize     lambda ||x||_1 + ( 1 / 2 ) ||x - (h - g)||^2
    xPrev = x;
    x = shrink( h - g, 1/(2*lambda) ); 
    
    stepSize = norm(x-xPrev);    
    tPrev = tCur;    
    tCur = (1 + sqrt(1+4*tCur^2)) / 2;
    h    = x + ((tPrev - 1)/tCur) * ( x - xPrev ); 
    
    % output
    if VERBOSE,
        disp(['   Iteration ' num2str(numIterations) '  ||x||_1 ' num2str(w' * abs(x)) '  step ' num2str(stepSize) '  ||x-y|| ' num2str(norm(x-y))]);
    end
    
    % check convergence
    if numIterations >= MAX_ITERATIONS || stepSize < MIN_STEP
        converged = true;
    end
end

x_hat = x;