%Script to verify the Generalized RPCA setup for P-BiG-AMP

%% Problem setup

clear all %#ok<CLSCR>
clc

%Control uniformVariance
uniformVariance = 1;


%Specify the rank of the matrix
R = 5;

%Specify sizes of the two matrix factors
Nq = 50;
Mq = 47;

%Specify number of measurements
M = 45;

%Derived sizes
Nb = Mq*R;
Nc = Nq*R + Mq*Nq;

%Build the two matrix factors
B = complex(randn(R,Mq),randn(R,Mq));
C1 = complex(randn(R,Nq),randn(R,Nq));

%Build the sparse (here dense for checking) matrix
C2 = complex(randn(Mq,Nq),randn(Mq,Nq));

%Create variances for use when testing ParametricZ
Nub = 3*rand(R,Mq);
Nuc1 = 5*rand(R,Nq);
Nuc2 = 2*rand(Mq,Nq);

%Make shat and nus
shat = 2*complex(randn(M,1),randn(M,1));
nus = 4*rand(M,1);

%Compute Q1
Q1 = B.'*C1;

%Build Q
Q = Q1 + C2;

%Compute vectorized versions
q = vec(Q);
b = vec(B);
c = [vec(C1);vec(C2)];
nub = vec(Nub);
nuc = [vec(Nuc1);vec(Nuc2)];


%Uniform variance
if uniformVariance
    nub = mean(nub(:));
    nuc = mean(nuc(:));
    nus = mean(nus(:));
end

%Build the random measurement matrices
Phi = complex(randn(Mq,Nq,M),randn(Mq,Nq,M));

%Make Phi structually sparse
for m = 1:M
    Phi(:,:,m) = Phi(:,:,m) .* (rand(Mq,Nq) > 0.8);
end

%Build Z
Z = zeros(M,1);
for m = 1:M
    Z(m) = trace(Phi(:,:,m)'*Q);
end

%Build full Phi
Phi_full = zeros(Mq*Nq,M);
for m = 1:M
    Phi_full(:,m) = vec(Phi(:,:,m));
end


%% Use Affine version to construct truth

%Notify
disp('Building sparse tensor')

%Tensor prep horribly inefficient direct method to ensure it is correct
zij = tenzeros(M,Nb,Nc);
for m = 1:M
    zij(m,:,1:(Nq*R)) = kron(conj(Phi(:,:,m)),eye(R));
end
zij = sptensor(zij);

%Build z0j
z0j = zeros(M,Nc);
z0j(:,(Nq*R+1):end) = Phi_full';
z0j = sptensor(z0j);

%Build it
pzAffine = Affine_ParametricZ(zij,[],z0j);


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
Z_pzLR = pzAffine.computeZ(b,c);
norm(Z - Zcheck);

%% Now, test the ParametricZ object

%Notify
disp('Starting ParametricZ tests')


%Build it
pz = LowRank_Plus_Sparse_Matrix_Recovery_ParametricZ(Mq,Nq,R,sparse(Phi_full));

%Expand nuc to 2x1 for uniform variance case
if uniformVariance
    nuc = nuc*ones(2,1);
end

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
norm(Z - Z_computeZ,'fro')


%Check Q1
Q1_pz = pz.computeQ1(b,c);
norm(Q1_pz - Q1,'fro')

%Checks
norm(Z_pComp - Z,'fro')
norm(pvarBar_pComp - pvarBar,'fro')
norm(pvar_pComp - pvar,'fro')
norm(rhat_rqComp - rhat,'fro')
norm(rvar_rqComp - rvar,'fro')
norm(qhat_rqComp - qhat,'fro')
norm(qvar_rqComp - qvar,'fro')

%Notify user about limitation of
if uniformVariance
    disp('In the uniform variance case, rvar and rhat will be slightly off')
end



