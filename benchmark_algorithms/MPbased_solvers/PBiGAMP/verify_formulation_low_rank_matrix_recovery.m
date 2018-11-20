
%Script to verify the Generalized Matrix Recovery setup for P-BiG-AMP

%% Problem setup

clear all %#ok<CLSCR>
clc

%Specify the rank of the matrix
R = 8;

%Specify sizes of the two matrix factors
Nq = 40;
Mq = 41;

%Specify number of measurements
M = 30;


%Derived sizes
Nb = Mq*R;
Nc = Nq*R;

%Build the two matrix factors
B = complex(randn(R,Mq),randn(R,Mq));
C = complex(randn(R,Nq),randn(R,Nq));

%Create variances for use when testing ParametricZ
Nub = 3*rand(R,Mq);
Nuc = 5*rand(R,Nq);

%Make shat and nus
shat = 2*complex(randn(M,1),randn(M,1));
nus = 4*rand(M,1);

%Compute Q
Q = B.'*C;

%Compute vectorized versions
q = vec(Q);
b = vec(B);
c = vec(C);
nub = vec(Nub);
nuc = vec(Nuc);


%Build the random measurement matrices
Phi = complex(randn(Nb/R,Nc/R,M),randn(Nb/R,Nc/R,M));

%Make Phi structually sparse
for m = 1:M
    Phi(:,:,m) = Phi(:,:,m) .* (rand(Nb/R,Nc/R) > 0.8);
end

%Build Z
Z = zeros(M,1);
for m = 1:M
    Z(m) = trace(Phi(:,:,m)'*Q);
end


%% Derivation checks

%Notify
disp('Starting derivation tests')

%Verify other method for computing Z
Phi_full = zeros(Nb*Nc/R^2,M);
Phi_fullT = zeros(Nb*Nc/R^2,M);
for m = 1:M
    Phi_full(:,m) = vec(Phi(:,:,m));
    Phi_fullT(:,m) = vec(Phi(:,:,m).');
end
Z_alt = Phi_full'*q;
norm(Z - Z_alt,'fro')

%Verify alternate q expressions
q1 = kron(eye(Nc/R),B.')*c;
q2 = kron(C.',eye(Nb/R))*vec(B.');
norm(q - q1,'fro')
norm(q - q2,'fro')

%Now, verify alternative method for computing Z
%First, build A
A = zeros(M,Nc);
A_otherForm = A;
for m = 1:M
    A(m,:) = b.' * kron(conj(Phi(:,:,m)),eye(R));
    A_otherForm(m,:) = vec(B*conj(Phi(:,:,m))).';
end
norm(A - A_otherForm,'fro')

%Compute new Z
Ztest = A*c;

%Check it
norm(Z - Ztest,'fro')

%% Check Z derivatives

%Notify
disp('Starting Z derivative tests')

%Build Zij
Zij = zeros(Nb,Nc,M);
Z_fromZij = zeros(M,1);
for m = 1:M
    Zij(:,:,m) = kron(conj(Phi(:,:,m)),eye(R));
    Z_fromZij(m) = b.'*Zij(:,:,m)*c;
end
norm(Z - Z_fromZij,'fro')

%Build Z0j- Notice that this is also A(b)!
Z0j = zeros(M,Nc);
for m = 1:M
    for jj = 1:Nc
        Z0j(m,jj) = b.'*Zij(:,jj,m);
    end
end
norm(A - Z0j,'fro')

%Test first of the Z0j expressions
Z0j_opt1 = Phi_full'*kron(eye(Nq),B.');
norm(Z0j - Z0j_opt1,'fro')

%Test second Z0j expression
Z0j_opt2 = zeros(M,Nc);
for m = 1:M
    Z0j_opt2(m,:) = vec(B*conj(Phi(:,:,m)));
end
norm(Z0j_opt2 - Z0j,'fro')


%Build Zi0
Zi0 = zeros(M,Nb);
for m = 1:M
    for ii = 1:Nb
        Zi0(m,ii) = Zij(ii,:,m)*c;
    end
end

%Test first Zi0 expression
Zi0_opt1 = Phi_fullT'*kron(eye(Mq),C.');
norm(Zi0 - Zi0_opt1,'fro')

%Try this another way. This let's us compute this without having to use the
%tranposed versions of the Phi matrices
coords = vec(reshape(1:numel(Q),Nq,Mq)');
temp = kron(eye(Mq),C.');
Zi0_opt1_alt = Phi_full'*temp(coords,:);
norm(Zi0 - Zi0_opt1_alt,'fro')

%Test second Zi0 expression
Zi0_opt2 = zeros(M,Nb);
for m = 1:M
    Zi0_opt2(m,:) = vec(C*Phi(:,:,m)');
end
norm(Zi0 - Zi0_opt2,'fro')

%Consistency check between the two
norm(Zi0*b - Z0j*c,'fro')

%Let's build the Aii matrices
Aii = zeros(M,Nc,Nb);
for ii = 1:Nb
    Bloc = zeros(size(B));
    Bloc(ii) = 1;
    Aii(:,:,ii) = Phi_full'*kron(eye(Nq),Bloc.');
end

%Test them by computing Zi0 with the Aii matrices
Zi0_Aii = zeros(size(Zi0));
for ii = 1:Nb
    Zi0_Aii(:,ii) = Aii(:,:,ii)*c;
end
norm(Zi0_Aii - Zi0,'fro')

%% pvar checks

%Notify
disp('Starting pvar tests')

%Compute pvarBar
pvarBar = abs(Zi0).^2 * nub;
pvarBar = pvarBar + abs(Z0j).^2 * nuc;


%Now compute the quadratic term in pvar
pvarQ = zeros(M,1);
for m = 1:M
    pvarQ(m) = nub'*(abs(Zij(:,:,m)).^2)*nuc;
end

%Test fast method
pvarQ2 = abs(Phi_full').^2 * vec(Nub.'*Nuc);
norm(pvarQ - pvarQ2,'fro')

%Compute pvar
pvar = pvarBar + pvarQ;


%% RQ checks

%Notify
disp('Starting RQ tests')

%Compute rvar
rvar = sum(repmat(nus,1,Nc).*abs(Z0j).^2);
rvar = 1 ./ rvar.';

%Going to compute rhat brute force for comparison sanity
rhat = zeros(Nc,1);
for jj = 1:Nc
    rhat(jj) = rhat(jj) + c(jj);
    for m = 1:M
        rhat(jj) = rhat(jj) + ...
            rvar(jj)*shat(m)*conj(Z0j(m,jj));
        
        for ii = 1:Nb
            %Compute appropriate double sum term
            rhat(jj) = rhat(jj) - c(jj)*rvar(jj)* ...
                nus(m)*nub(ii)*abs(Zij(ii,jj,m)).^2;
            
        end
    end
end

%Compute qvar
qvar = sum(repmat(nus,1,Nb).*abs(Zi0).^2);
qvar = 1 ./ qvar.';

%Going to compute qhat brute force for comparison sanity
qhat = zeros(Nb,1);
for ii = 1:Nb
    qhat(ii) = qhat(ii) + b(ii);
    for m = 1:M
        qhat(ii) = qhat(ii) + ...
            qvar(ii)*shat(m)*conj(Zi0(m,ii));
        for jj = 1:Nc
            qhat(ii) = qhat(ii) - ...
                b(ii)*qvar(ii)*nus(m)*nuc(jj)*abs(Zij(ii,jj,m)).^2;
            
        end
    end
end


%% Now, test the ParametricZ object

%Notify
disp('Starting ParametricZ tests')

%Tensor prep
zij = tenzeros(M,Nb,Nc);
for m = 1:M
    zij(m,:,:) = kron(conj(Phi(:,:,m)),eye(R));
end
zij = sptensor(zij);





%% Run checks

%Check RQ computations
opt.varThresh = inf;
opt.uniformVariance = false;

%Pick pz to test
pz = LowRank_Matrix_Recovery_ParametricZ(Mq,Nq,R,sparse(Phi_full));
%pz = Affine_ParametricZ(zij,[],[],[]);

%Check AX
Z_computeZ = pz.computeZ(b,c);
norm(Z - Z_computeZ,'fro')

tic
%Check P computations
[AX_pComp,pvarBar_pComp,pvar_pComp] = pz.pComputation(opt,b,nub,c,nuc);
[rhat_rqComp,rvar_rqComp,qhat_rqComp,qvar_rqComp] = ...
    pz.rqComputation(opt,b,nub,c,nuc,shat,nus);
toc

norm(AX_pComp - Z,'fro')
norm(pvarBar_pComp - pvarBar,'fro')
norm(pvar_pComp - pvar,'fro')
norm(rhat_rqComp - rhat,'fro')
norm(rvar_rqComp - rvar,'fro')
norm(qhat_rqComp - qhat,'fro')
norm(qvar_rqComp - qvar,'fro')






