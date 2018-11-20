function [x,estHist] = fistaEst(x0,b,A,options)
%FISTA Fast Iterative Soft Thresholding
%   Code to implement FISTA as described in "A Fast Iterative
%   Shrinkage-Thresholding Algorithm for Linear Inverse Problems" by Beck
%   and Teboulle. Implements the FISTA algorithm to solve the convex
%   program argmin_x ||Ax - b||_2^2 + |Lam .* x|. The implementation
%   support complex valued data.
%
%   Code inputs:
%   -x0 is the initial guess at the solution
%   -b is the measured data, i.e. b = Ax + w.
%   -A either a matrix or a linear operator defined by the LinTrans class.
%   -options a set of options of the FistaOpt class
%
%   Code Outputs:
%   -x is the estimate of the cost function's argmin.
%   -estHist history of the algorithm progress
%       estHist.xhat each iteration for the estimate


%% Input checking

% Get options
if (nargin < 4)
    options = FistaOpt();
elseif (isempty(options))
    options = FistaOpt();
end

%Check to see if history desired
saveHist = nargout > 1;


% If A is a double matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Use the power iteration to estimate the Lipschitz constant if not provided
if isempty(options.lip)
    
    %Initial guess
    q = randn(length(x0),1);
    q = q / norm(q);
    
    thresh = 1e-5;
    err = inf;
    uest = inf;
    while err > thresh
        q = A.multTr(A.mult(q));
        
        %Progress
        unew = norm(q);
        err = abs(log(uest / unew));
        
        %Save the estimate
        uest = unew;
        q = q / norm(q);
    end
    
    %Use a little more than twice the eigenvalue estimate
    options.lip = 2.05*uest;
    
end


%Convert options.lam to a vector
if length(options.lam) == 1
    Lam = options.lam * ones(size(x0));
else
    Lam = options.lam;
end



%% Iteration


%Initialize variables
y = x0;
x = x0;
t = 1;
stop = 0;
it = 0;


%Preallocate storage if estimHist is requested
if saveHist
    estHist.xhat = zeros(length(x0),options.nit);
end

%Walk through iterations
while ~stop
    
    %Counter
    it = it + 1;
    
    %Gradient step
    alphares = y - (1 / options.lip)*(2*A.multTr(A.mult(y) - b));
    
    %Soft threshold
    xnew = max(0,abs(alphares)-Lam/options.lip) .*...
        (alphares ./ abs(alphares));
    
    
    %Compute the new t
    tnew = (1 + sqrt(1 + 4*t^2))/2;
    
    %Compute the new y
    y = xnew + (t - 1)/tnew*(xnew - x);
    
    
    
    %Check max iterations
    if it > options.nit
        stop = 1;
    else 
        
        %Check error tolerance
        cnew = norm(x - xnew) / norm(xnew);
        if cnew < options.tol && options.tol > 0
            stop = 1;
        end
        
    end
    
    %Record history if desired
    if saveHist
        estHist.xhat(:,it) = xnew;
    end
    
    %Update on progress
    if options.verbose
        
        disp(['FISTA on iteration '...
            num2str(it)...
            ' Most Recent threshold check was ' num2str(cnew)])
        
    end
    
    
    %Update x and t
    x = xnew;
    t = tnew;
    
end

%Trim the outputs if early termination occurred
if saveHist && (it < options.nit)
    estHist.xhat = estHist.xhat(:,1:it);
end


%Inform the user of final result
if options.verbose
    disp(['Total FISTA Iterations completed: '...
        num2str(it) ' out of '...
        num2str(options.nit) ' permitted'])
end



















