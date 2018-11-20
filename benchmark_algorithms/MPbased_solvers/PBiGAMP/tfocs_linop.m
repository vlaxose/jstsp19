%TFOCS linear operator to match the convention used 
%by LowRank_Matrix_Recovery_ParametricZ
function y = tfocs_linop( Mq, Nq, M, Phi, x, mode )
switch mode,
    case 0,
        y = { [Mq,Nq], [M,1] };
    case 1,
        y = Phi'*x(:);
    case 2,
        y = reshape(Phi*x,Mq,Nq);
end


