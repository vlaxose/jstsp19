function x=l1Opt_fista(Y,r,vv,p)

%solve the dictionary learning problem:
% min_x |Y'x|_1, s.t., r'x=1
%input:
% r: constraint vector
% h=Y* (YY*)^{-1} r
% vv=V*V';
% Y is m by p ......

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% q=VV'h/qc; Precomputation needed
% qc=|VV'h|_2
% where h=Y* (YY*)^{-1} r
% where Y=USV'

hc=inv(Y*Y')*Y;

h=hc'*r;

q=vv*h;
qc=norm(q);
q=q/qc;
qq=q*q';

At=@(z)(qq*z+z-vv*z);


b=q(:,1)/qc;


%tic                
z = minimize_l1_lc(At,b,10,'FISTA');
%tCur = toc;
        

x=hc*z;

