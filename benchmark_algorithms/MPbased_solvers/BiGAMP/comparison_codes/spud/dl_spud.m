function [A,X]=dl_spud(Y)

%% Input
% Y: m-by-p sample matrix,
% n: the number of atoms in the dictionary (currently we assume n=m)
% IterNum: number of iterations. 
%output:
% A : m-by-n dictionary
% X : n-by-p coefficients

%% Based on John Wright's ADMM optimization toolbox.

VERBOSE = 1;
epsilon=1e-5;
[m,p]=size (Y);
n=m;


%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preconditioning
[U,S,V]=svd(Y,'econ');
Ypre=V';
%vv=V*V';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=[];
for i=1:m
    if i~=1
       P=B*pinv (B'*B)*B';          
    else
       P=0;
    end
 
    smNNZ=inf;
    for j=1:p
        r=Ypre(:,j);
        r=r-P*r;  
       
%        b=l1Opt_fista(Ypre,r,vv,p);            

        b = spud_subproblem(Ypre, r); 
        t=b'*Ypre;
        curNNZ=sum (sum ( abs (t) > epsilon ));

        if curNNZ<smNNZ
            smNNZ=curNNZ;
            X(i,:)=t;
            B(:,i)=b;
        end           
     end
     if VERBOSE,
        disp ([ num2str(i) ' row(s) of X recovered, nonzero # in the new row:'  num2str(smNNZ)]);
     end
end
    

A = (Y * Y') * inv(X*Y');
A = normalize_columns (A);
X = inv(A)*Y;

function Y = normalize_columns (X)

stds=sqrt (diag (X'*X));
Y=X./(repmat (stds',[size(X,1),1]));




