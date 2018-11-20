
function [Xhat, SUREfin, estHist, optSUREfin] = SUREBGAMP(Y, A, optSURE, optGAMP)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
[M,~] = A.size();
if size(Y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end

%Set default GAMP options if unspecified
if nargin <= 2
    optSURE = [];
end

if nargin <= 3
    optGAMP = [];
end

%Check inputs and initializations
[optGAMP, optSURE] = check_opts(optGAMP, optSURE);

optSURE.L = 1;
%optSURE.heavy_tailed = false;

% histFlag = false;

if nargout >=3 ;
    histFlag = true;
end;

% if histFlag
    [Xhat, SUREfin, estHist, optSUREfin] = SUREGMAMP(Y, A, optSURE, optGAMP);
% else
%    [Xhat, SUREfin] = SUREGMAMP(Y, A, optSURE, optGAMP);
% end

return;
