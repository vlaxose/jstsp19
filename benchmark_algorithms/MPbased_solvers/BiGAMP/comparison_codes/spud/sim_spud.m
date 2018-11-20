addpath(genpath('./ALM-JW'))

n=10;
k=3;

m=n;
p=round(5*n*log(n));

A=randn(m,n);
X=randn(n,p);
for i=1:p
    rp = randperm(n);
    idx = rp(1:n-k);
    X(idx,i)=0;
 end


Y=A*X;
            
 
% tic
% [A_spud,X_spud]=dl_spud(Y);     
% toc
            
tic
[A_spud,X_spud]=dl_spud_gd(Y);     
toc
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Evaluation
           
[relativeErrorA,relativeErrorX] = verify_dictionary(A,X,A_spud,X_spud)
            
        


