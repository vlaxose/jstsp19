%Script to verify the Generalized Matrix Recovery setup for P-BiG-AMP

%% Problem setup

%Clean slate
clear all %#ok<CLSCR>
clc

%Control uniformVariance
uniformVariance = 1;

%Specify the rank of the matrix
R = 8;

%Specify sizes of the two matrix factors
Nq = 40;
Mq = 41;

%Specify number of measurements
M = 70;


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


%Uniform variance
if uniformVariance
    Nub = mean(Nub(:));
    Nuc = mean(Nuc(:));
    nus = mean(nus(:));
end

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


%First, build A
A = zeros(M,Nc);
for m = 1:M
    A(m,:) = b.' * kron(conj(Phi(:,:,m)),eye(R));
end

%Build Phi_full
Phi_full = zeros(Nb*Nc/R^2,M);
for m = 1:M
    Phi_full(:,m) = vec(Phi(:,:,m));
end


%% Check Z derivatives



%Build Zij
Zij = zeros(Nb,Nc,M);
for m = 1:M
    Zij(:,:,m) = kron(conj(Phi(:,:,m)),eye(R));
end

%Build Z0j- Notice that this is also A(b)!
Z0j = zeros(M,Nc);
for m = 1:M
    for jj = 1:Nc
        Z0j(m,jj) = b.'*Zij(:,jj,m);
    end
end


%Build Zi0
Zi0 = zeros(M,Nb);
for m = 1:M
    for ii = 1:Nb
        Zi0(m,ii) = Zij(ii,:,m)*c;
    end
end

%Compute needed sums over the z's
Z0j_norm2 = sum(abs(Z0j(:)).^2);
Zi0_norm2 = sum(abs(Zi0(:)).^2);
zij_norm2 = sum(abs(Zij(:)).^2);
zij_sumM = sum(abs(Zij).^2,3);

%% pvar checks

%Notify
disp('Starting pvar tests')


if ~uniformVariance
    %Compute pvarBar
    pvarBar = abs(Zi0).^2 * nub;
    pvarBar = pvarBar + abs(Z0j).^2 * nuc;
    
    
    %Now compute the quadratic term in pvar
    pvarQ = zeros(M,1);
    for m = 1:M
        pvarQ(m) = nub'*(abs(Zij(:,:,m)).^2)*nuc;
    end
    
    
    %Compute pvar
    pvar = pvarBar + pvarQ;
    
else
    
    %Compute pvarBar
    pvarBar = (nub*Zi0_norm2 + nuc*Z0j_norm2)/M;
    
    %Compute pvar
    pvar = pvarBar + nub*nuc/M*zij_norm2;
    
end


%% RQ checks

%Notify
disp('Starting RQ tests')

if ~uniformVariance
    
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
    
else
    
    %Compute rvar and qvar
    rvar = 1 / (nus/Nc*Z0j_norm2);
    qvar = 1 / (nus/Nb*Zi0_norm2);
    
    %Compute rhat
    rhat = c + rvar*(Z0j'*shat) - (rvar*nus*nub)*(c .* vec(sum(zij_sumM,1)));
    
    %Compute qhat
    qhat = b + qvar*(Zi0'*shat) - (qvar*nus*nuc)*(b .* vec(sum(zij_sumM,2)));
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
opt.uniformVariance = uniformVariance;

%Pick pz to test
pz = LowRank_Matrix_Recovery_ParametricZ(Mq,Nq,R,sparse(Phi_full));
%pz = Affine_ParametricZ(zij,[],[],[]);

%Check AX
Z_computeZ = pz.computeZ(b,c);
norm(Z - Z_computeZ,'fro')

tic
%Check computations
[z_pComp,pvarBar_pComp,pvar_pComp] = pz.pComputation(opt,b,nub,c,nuc);
[rhat_rqComp,rvar_rqComp,qhat_rqComp,qvar_rqComp] = ...
    pz.rqComputation(opt,b,nub,c,nuc,shat,nus);
toc

norm(z_pComp - Z,'fro')
norm(pvarBar_pComp - pvarBar,'fro')
norm(pvar_pComp - pvar,'fro')
norm(rhat_rqComp - rhat,'fro')
norm(rvar_rqComp - rvar,'fro')
norm(qhat_rqComp - qhat,'fro')
norm(qvar_rqComp - qvar,'fro')




