function [X] = genMRF (obj, N1, N2, NGibbsIter)

    % Extract MRF parameters from MarkovField object
    beta1 = obj.betaH;
    beta2 = obj.betaV;
    lambda = obj.alpha;

    % N1 N2 are dimensions of the image
    N1=N1+2;
    N2=N2+2;
    
    X=rand(N1,N2)<0.5;
    
    
    index=find(X==0);
    X=double(X);
    X(index)=-1;
    
    X(1,:)=0;%rand(1,Length)<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(end,:)=0;%rand(1,Length)<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(:,1)=0;%rand(1,Length).'<exp(-lambda)/(exp(-lambda)+exp(lambda));
    X(:,end)=0;%rand(1,Length).'<exp(-lambda)/(exp(-lambda)+exp(lambda));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    [r1 c1]=meshgrid(2:2:N1-1,2:2:N2-1);
    [r2 c2]=meshgrid(3:2:N1-1,3:2:N2-1);
    [r3 c3]=meshgrid(2:2:N1-1,3:2:N2-1);
    [r4 c4]=meshgrid(3:2:N1-1,2:2:N2-1);
    
    ind1=r1(:)+(c1(:)-1)*N1;
    ind2=r2(:)+(c2(:)-1)*N1;
    ind3=r3(:)+(c3(:)-1)*N1;
    ind4=r4(:)+(c4(:)-1)*N1;
    
    ind1_cm=ind1-N1;
    ind2_cm=ind2-N1;
    ind3_cm=ind3-N1;
    ind4_cm=ind4-N1;
    
    ind1_rm=ind1-1;
    ind2_rm=ind2-1;
    ind3_rm=ind3-1;
    ind4_rm=ind4-1;
    
    ind1_cp=ind1+N1;
    ind2_cp=ind2+N1;
    ind3_cp=ind3+N1;
    ind4_cp=ind4+N1;
    
    ind1_rp=ind1+1;
    ind2_rp=ind2+1;
    ind3_rp=ind3+1;
    ind4_rp=ind4+1;
    
    for n=1:NGibbsIter
    
        cpx2=exp(-beta1*(X(ind1_cm)+X(ind1_cp)) - beta2*(X(ind1_rm)+X(ind1_rp)) + lambda);
        cpx1=exp(beta1*(X(ind1_cm)+X(ind1_cp)) + beta2*(X(ind1_rm)+X(ind1_rp)) - lambda);
        X(ind1)=2*(rand(size(ind1))  < cpx1./(cpx1+cpx2))-1;
    
        cpx2=exp(-beta1*(X(ind2_cm)+X(ind2_cp)) - beta2*(X(ind2_rm)+X(ind2_rp)) + lambda);
        cpx1=exp(beta1*(X(ind2_cm)+X(ind2_cp)) + beta2*(X(ind2_rm)+X(ind2_rp)) - lambda);
        X(ind2)=2*(rand(size(ind2))  < cpx1./(cpx1+cpx2))-1;
    
        cpx2=exp(-beta1*(X(ind3_cm)+X(ind3_cp)) - beta2*(X(ind3_rm)+X(ind3_rp)) + lambda);
        cpx1=exp(beta1*(X(ind3_cm)+X(ind3_cp)) + beta2*(X(ind3_rm)+X(ind3_rp)) - lambda);
        X(ind3)=2*(rand(size(ind3))  < cpx1./(cpx1+cpx2))-1;
    
        cpx2=exp(-beta1*(X(ind4_cm)+X(ind4_cp)) - beta2*(X(ind4_rm)+X(ind4_rp)) + lambda);
        cpx1=exp(beta1*(X(ind4_cm)+X(ind4_cp)) + beta2*(X(ind4_rm)+X(ind4_rp)) - lambda);
        X(ind4)=2*(rand(size(ind4))  < cpx1./(cpx1+cpx2))-1;
        
    end

X=(X>0);

X=X(2:end-1,2:end-1);
    
end
