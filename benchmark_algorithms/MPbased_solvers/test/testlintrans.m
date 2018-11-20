function A = testlintrans(lt)
% Test consistency of a LinTrans object
% 
%   A = testlintrans(lt)
%
%  Tests performed:
%   * dimensionality matches lt.size()
%   * lt.mult and lt.multTr are adjoint operators (detect scaling problems)
%   * multSq and mulSqTr are consistent with the Frobenius norm of the matrix operator
%   * if nargout > 0, the full matrix for lt is returned
%
%   Note that recovering the full matrix operator can be slow for large transforms
%
% Mark Borgerding 2013-05-17

[m,n] = lt.size();

%%%%
% test dimensionality is correct
y = lt.mult( randn(n,1) );
if ~all(size(y)==[m 1])
    error 'wrong size'
else
    fprintf('dimensionality %dx%d OK\n',m,n)
end

%%%%
%% test that mult and multTr are truly adjoint operators 
x = zeros(n,1);
x(1) = 1;
y = lt.mult(x);
ny = norm(y);
if ny==0
    error('does not support zero-valued columns');
end
x = lt.multTr( y );  
ratioOfNorms =  x(1) / ny^2 ;  % this will be 1.0 if mult,multTr are adjoint
if abs(1-ratioOfNorms) > 1e-9
    fprintf(2,'ratioOfNorms=1%+g \n',1-ratioOfNorms)
else
    fprintf('adjointness OK\n')
end

fwdOpFroNorm2 = sum( lt.multSq( ones(n,1) ) );
adjOpFroNorm2 = sum( lt.multSqTr( ones(m,1) ) );
if nargout>0
    fprintf('recovering the matrix operator (this may take a while)\n');
    A=zeros(m,n);
    for i=1:n
        x = zeros(n,1);
        x(i) = 1;
        A(:,i) = lt.mult(x);
    end
    FroNorm2 = norm(A,'fro')^2;
    tol = 1e-6;
else
    fprintf('estimating the matrix Frobenius norm \n');
    FroNorm2=0;
    trialVecs = min(n,100);
    for i=1:trialVecs
        if trialVecs==n
            x = zeros(n,1);
            x(i) = 1;
            inNorm = 1;
        else
            x = randn(n,1);
            inNorm = norm(x);
        end
        FroNorm2 = FroNorm2 + norm( lt.mult(x) /inNorm )^2;
    end
    FroNorm2 = FroNorm2 / trialVecs * n;
    A=[];
    tol=1e-2;
end

if abs( 1- fwdOpFroNorm2/adjOpFroNorm2) > 1e-6 
    fprintf(2,'FrobeniusNorm^2 using multSq=%g,using multSqTr=%g (delta=%g)\n' , fwdOpFroNorm2,adjOpFroNorm2,fwdOpFroNorm2-adjOpFroNorm2);
elseif abs( 1- FroNorm2/fwdOpFroNorm2) > tol
    fprintf(2,'FrobeniusNorm^2 long way= %g, using multSq&multSqTr=%g (rel delta=%g)\n' , FroNorm2,fwdOpFroNorm2,1- FroNorm2/fwdOpFroNorm2)
else
    fprintf('recovered Forbenius norm matches that of multSq,multSqTr\n');
end
