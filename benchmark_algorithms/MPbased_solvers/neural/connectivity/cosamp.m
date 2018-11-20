function [Sest,d]=cosamp(Phi,u,K,opt)

% Cosamp algorithm
%   Input
%       K : sparsity of Sest
%       Phi : measurement matrix
%       u: measured vector
%       tol1 : tolerance for approximation between successive solutions. 
%   Output
%       Sest: Solution found by the algorithm
%       d   : success index (d=1 is success, d = 0 is no convergence)
%
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% 
% This implementation was written by David Mary
%
% This script/program is released under the Commons Creative Licence
% with Attribution Non-commercial Share Alike (by-nc-sa)
% http://creativecommons.org/licenses/by-nc-sa/3.0/
% Short Disclaimer: this script is for educational purpose only.
% Longer Disclaimer see  http://igorcarron.googlepages.com/disclaimer

% Get options
tol1 = opt.tol1;
prt = opt.prt;

% Initialization
Sest=zeros(size(Phi,2),1);
utrue = Sest;
v=u;
t=1; T2=[];
while t < 101 
[k,z]=sort(abs(Phi'*v));k=flipud(k);z=flipud(z);
Omega=z(1:2*K);
T=sort(union(Omega,T2));phit=Phi(:,T);
% The next step is the one that can be improved with a Conjugate Gradient
% algorithm
b=abs(pinv(phit)*u);
[k3,z3]=sort((b));k3=flipud(k3);z3=flipud(z3);
Sest=zeros(size(utrue));
Sest(T(z3(1:K)))=abs(b(z3(1:K)));
[k2,z2]=sort(abs(Sest));k2=flipud(k2);z2=flipud(z2);
T2=z2(1:K);
v=u-Phi*Sest;
d=0;n2=norm(abs(v),'inf');
if n2 < tol1
    d=1;t=1e10;
    if (prt)
        disp('CoSaMP: success');
    end
end
 t=t+1;
 end
 if d==0
     if (prt)
        disp('CoSaMP: failed')
     end
 end
 