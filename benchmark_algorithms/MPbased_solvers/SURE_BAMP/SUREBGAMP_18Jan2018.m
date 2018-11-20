
function [z, convergence_error] = SUREBGAMP_18Jan2018(A, optSURE, optGAMP, up, down, sigma, z_opt)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
% [M,~] = A.size();
% if size(Y, 1) ~= M
%     error('Dimension mismatch betweeen Y and A')
% end

% %Set default GAMP options if unspecified
% if nargin <= 2
%     optSURE = [];
% end
% 
% if nargin <= 3
%     optGAMP = [];
% end

%Check inputs and initializations
[optGAMP, optSURE] = check_opts(optGAMP, optSURE);

optSURE.L = 1;
%optSURE.heavy_tailed = false;

% histFlag = false;

% if nargout >=3 
%     histFlag = true;
% end;

maxEMiters = 100;
convergence_error = zeros(maxEMiters, 1);

[m,n] = size(A);

z = zeros(n,1);
b = zeros(m,1);

for em_step=1:maxEMiters

    Xz = A*z;

    for i=1:m
       quant1 = (exp(-(down(i) - Xz(i))^2/(2*sigma^2)) - exp(-(up(i) - Xz(i))^2/(2*sigma^2)));
       quant2 = (erf((-down(i) + Xz(i))/(sqrt(2)*sigma)) - erf((-up(i) + Xz(i))/(sqrt(2)*sigma)));
      
      b(i) = quant1/quant2;


    end
    b = sigma/sqrt(2*pi)*b;

%     % Sparse solvers
%     [~, indx] = sort(abs(z_opt), 'descend');
%     orcl = indx(1:2*L);
%     z_orcl = zeros(n, 1);
%     z_orcl(orcl) = X(:, orcl)\(Xz+b);


    Y = A'*(A*z+b);
% if histFlag
    z = SUREGMAMP(Y, A, optSURE, optGAMP);
% else
%    [Xhat, SUREfin] = SUREGMAMP(Y, A, optSURE, optGAMP);
% end
    convergence_error(em_step) = norm(z-z_opt)^2/norm(z_opt)^2;

end

return;
