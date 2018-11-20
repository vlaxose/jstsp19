%Script to verify the multiple snapshot setup for P-BiG-AMP

%% Problem setup

clear all %#ok<CLSCR>
clc

%Control uniformVariance
uniformVariance = 1;

%Specify sizes
K = 30;
N = 45;
L = 8;
Nb = 6;

%Derived sizes
Nc = N*L;
M = K*L;


%Build vectors
b = complex(randn(Nb,1),randn(Nb,1));
c = complex(randn(Nc,1),randn(Nc,1));
nub = 3*rand(Nb,1);
nuc = 5*rand(Nc,1);

%Make shat and nus
shat = 2*complex(randn(M,1),randn(M,1));
nus = 4*rand(M,1);


%Uniform variance
if uniformVariance
    nub = mean(nub(:));
    nuc = mean(nuc(:));
    nus = mean(nus(:));
end

%Build random matrices
A0 = complex(randn(K,N),randn(K,N));
Ai = complex(randn(K,N,Nb),randn(K,N,Nb));

%Build the true z
C = reshape(c,N,L);
Z = A0*C;
for ii = 1:Nb
    Z = Z + b(ii)*(Ai(:,:,ii)*C);
end
z = vec(Z);



%% Use Affine version to construct truth

%Notify
disp('Building sparse tensor')

%Tensor prep horribly inefficient direct method to ensure it is correct
zij = tenzeros(M,Nb,Nc);
for ii = 1:Nb
    zij(:,ii,:) = kron(eye(L),Ai(:,:,ii));
end
zij = sptensor(zij);


%Build it
pzAffine = Affine_ParametricZ(zij,[],sptensor(kron(eye(L),A0)));


%% Run checks

%Options
opt.varThresh = 1e5;
opt.uniformVariance = uniformVariance;

tic
%Compute P
[Zcheck,pvarBar,pvar] = pzAffine.pComputation(opt,b,nub,c,nuc);

%Compute RQ checks
[rhat,rvar,qhat,qvar] = ...
    pzAffine.rqComputation(opt,b,nub,c,nuc,shat,nus);
toc


%Make sure we get the right Z this way
Z_check = pzAffine.computeZ(b,c);
norm(Z - reshape(Z_check,K,L),'fro')

%% Now, test the ParametricZ object

%Notify
disp('Starting ParametricZ tests')

%Build the needed Aop
Aop = Matrix_PLinTrans(reshape(Ai,K*N,Nb),A0,K,N);

%Build the ParametricZ objet
pz = Multiple_Snapshot_ParametricZ(Aop,L);

%Start timing
tic

%Check P computations
[Z_pComp,pvarBar_pComp,pvar_pComp] = pz.pComputation(opt,b,nub,c,nuc);

%Check RQ computations
[rhat_rqComp,rvar_rqComp,qhat_rqComp,qvar_rqComp] = ...
    pz.rqComputation(opt,b,nub,c,nuc,shat,nus);

%Stop timing
toc

%Check Z
Z_computeZ = pz.computeZ(b,c);
norm(vec(Z) - Z_computeZ,'fro')



%Checks
norm(Z_pComp - vec(Z),'fro')
norm(pvarBar_pComp - pvarBar,'fro')
norm(pvar_pComp - pvar,'fro')
norm(rhat_rqComp - rhat,'fro')
norm(rvar_rqComp - rvar,'fro')
norm(qhat_rqComp - qhat,'fro')
norm(qvar_rqComp - qvar,'fro')



