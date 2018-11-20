function [A,X]=dl_spud_gd(Y)

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
[n,p]=size (Y);
IterNum=p;


%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preconditioning
[U,S,V]=svd(Y,'econ');
Ypre=V';
%vv=V*V';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%boostthres=0;% round (p/6);           %% you may set the threshold to make
%it faster

for j=1:IterNum
    r=Ypre(:,j);

 %   b=l1Opt_fista(Ypre,r,vv,p);      
    b = spud_subproblem(Ypre, r); 

%     if sum (isnan (b))~=0
%       continue;
%    end
    X_all(j,:)=b'*Ypre;
    curNNZ=sum (sum ( abs (X_all(j,:)) > epsilon ));
%     if curNNZ<boostthres
%        X(i,:)=t;
%        break; 
%     end
       
    if VERBOSE,
        disp ([ num2str(j) ' candidate row(s) of X recovered, nonzero # in the new row:'  num2str(curNNZ)]);
    end

end
    


% [X,Ilist]=DictSel(X_all',n);
% X=X';

X=rowSel(X_all,n);


A = (Y * Y') * inv(X*Y');
A = normalize_columns (A);
X = inv(A)*Y;




function X=rowSel(X_all,n)
epsilon=1e-5;
tol=0.0001;
p=size(X_all,2);

t=abs(X_all)>epsilon;
tn=sum(t,2);
[C,I]=sort(tn,'ascend');

js=1;
J_all=[];
for i=1:n
    for j=js:p
        X(i,:)=X_all(I(j),:);
        js=j+1;
        if(rank(X,tol)==i)
           J_all=[J_all,j];
           break;
        end
    end
end

%%%The rest is only to fix the case when we do not get n independent rows.
%%%If so, we simply add some sparse candidates back to return an X, but in
%%%this case X is ill-conditioned.

Jc=I;
Jc(J_all)=[];
sx=size(X,1);
if(sx<n)
    X((sx+1):n,:)=X_all(Jc(1:(n-sx)),:);
end

function Y = normalize_columns (X)

stds=sqrt (diag (X'*X));
Y=X./(repmat (stds',[size(X,1),1]));


function [A,Ilist]=DictSel(atoms,n)

[d,N]=size(atoms);

atoms=normalize_columns(atoms);
Ilist(1)=randsample(N,1);
%Ilist(1)=1;
A(:,1)=atoms(:,Ilist(1));
for i=2:n
    top=0;
    topi=1;
    if i<d
        for j=1:N
            Ae=[A,atoms(:,j)];
            t=det(Ae'*Ae);
            if t>top
               topi=j;
               top=t;
            end
        end
    else
        for j=1:N
            Ae=[A,atoms(:,j)];
            t=det(Ae*Ae');
            if t>=top
               topi=j;
               top=t;
            end
        end        
    end
    
    A(:,i)=atoms(:,topi);
    Ilist(i)=topi;
end




