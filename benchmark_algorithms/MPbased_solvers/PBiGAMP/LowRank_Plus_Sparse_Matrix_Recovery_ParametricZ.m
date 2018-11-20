classdef LowRank_Plus_Sparse_Matrix_Recovery_ParametricZ < ParametricZ
    %This class implements the ParametricZ functinality for low rank plus
    %sparse matrix recovery. This problem is more general than RPCA, since the
    %measurements are arbitrary linear functions of the entries of the entire
    %matrix.
    %The full matrix is Q = Q1 + C2. All 3 matrices are Mq x Nq sized,
    %with Q1 low rank and C2 sparse.
    %The low rank matrix is Q1 = B.'C1.
    %The matrix is rank R.
    %B is R x Mq=Nb/R
    %C1 is R x Nq=(Nc - Mq*Nq)/R
    %where Nb and Nc are the sizes of the two parameters.
    %The parameter vector b = vec(B)
    %The parameter vector c = [vec(C1);vec(C2)].
    %The model returns M linear funcions of the entries of Q.
    %Each measurement corresponds to a measurement matrix Phi_m. We thus have:
    %z_m = trace(Phi_m' * Q) for m = 1....M.
    %Phi should be provided as a sparse matrix if it is structually sparse.
    %We can write this as
    %z_m = vec(Phi_m)'*kron(speye(obj.Nq),B.')*vec(C1) + vec(Phi_m)'*vec(C2)
    %The current implementation is straightforward. Performance
    %improvements leveraging sparsity and/or other structure are possible.
    
    properties (SetAccess=protected)
        Mq; Nq; R; %Dimensions and rank of low rank matrix
        %Provide Phi as sparse if it has a sparse structure
        Phi; %NqMq by M matrix. Column m is vec(Phi_m) so that Z = Phi'*vec(Q)
        Phi2; %Elementwise squared version of Phi. Stored to avoid repeated computation
        
        %We also store and compute the vectorized tranposes. Notice that
        %these are NOT the transposes of Phi and Phi2. They are
        %permutations that correspond to transposing the Phi_m matrices
        %BEFORE they are vectorized.
        PhiT;
        PhiT2;
        
        %Quantities used by sparse mode
        zij_sumSquared; %sum of the squared entries of zij
        zij_sumMi; %sum of squared entries over m and i (Nc x 1)
        zij_sumMj %sum of squared entries over m and j (Nb x 1)
        Phi2_sum; %sum of Phi2
        G1; %Sum of Phi_m^H*Phi_m
        G2; %Sum of Phi_m*Phi_m^H
    end
    
    
    
    methods
        
        %% default constructor
        function obj = LowRank_Plus_Sparse_Matrix_Recovery_ParametricZ(...
                Mq,Nq,R,Phi)
            
            if nargin < 4
                error('Require 4 arguments')
            end
            
            %Check
            if (size(Phi,1) ~= Nq*Mq) || (R > min(Nq,Mq))
                error('Sizes inconsistent')
            end
            
            %Assign
            obj.Nq = Nq;
            obj.Mq = Mq;
            obj.R = R;
            obj.Phi = Phi;
            obj.Phi2 = abs(Phi).^2;
            
            %Build the tranposed versions
            coordHelper = reshape(1:(Mq*Nq),Mq,Nq)';
            coordHelper = coordHelper(:);
            obj.PhiT = Phi(coordHelper,:);
            obj.PhiT2 = abs(obj.PhiT).^2;
            
            %Builds some needed quantities for sparse mode
            obj.zij_sumSquared = full(obj.R*sum(obj.Phi2(:)));
            
            %Pre-compute sums over dimensions of zij
            sumM = reshape(sum(obj.Phi2,2),Mq,Nq);
            obj.zij_sumMi = full(kron(vec(sum(sumM,1)),ones(obj.R,1)));
            obj.zij_sumMj = full(kron(vec(sum(sumM,2)),ones(obj.R,1)));
            obj.Phi2_sum = full(sum(obj.Phi2(:)));
            
            %Build Phi matrices
            obj.G1 = zeros(Nq,Nq);
            obj.G2 = zeros(Mq,Mq);
            for m = 1:size(Phi,2)
                Phim = reshape(Phi(:,m),Mq,Nq);
                obj.G1 = obj.G1 +  Phim'*Phim;
                obj.G2 = obj.G2 + Phim*Phim';
            end
        end
        
        %% Function to compute Q from b and c
        function Q = computeQ(obj,bhat,chat)
            %Just a multiplication after reshape followed by adding the
            %sparse component
            Q = reshape(bhat,obj.R,obj.Mq).'*reshape(chat(1:obj.R*obj.Nq),obj.R,obj.Nq);
            Q = Q + reshape(chat(obj.R*obj.Nq+1:end),obj.Mq,obj.Nq);
        end
        
        %% Function to compute Q1 from b and c
        function Q1 = computeQ1(obj,bhat,chat)
            %Just a multiplication after reshape
            Q1 = reshape(bhat,obj.R,obj.Mq).'*reshape(chat(1:obj.R*obj.Nq),obj.R,obj.Nq);
            
        end
        
        
        
        %% returnSizes
        %Return all problem dimensions
        function [M,Nb,Nc] = returnSizes(obj)
            
            %Obtain sizes
            M = size(obj.Phi,2);
            Nc = obj.Nq*obj.R + obj.Nq*obj.Mq;
            Nb = obj.Mq*obj.R;
            
        end
        
        
        
        
        %% computeZ
        
        %Compute z(bhat,chat)
        function z = computeZ(obj,bhat,chat)
            
            %Compute Q
            Q = obj.computeQ(bhat,chat);
            
            %Compute the result
            z = obj.Phi' * Q(:);
            
        end
        
        
        
        %% pComputation
        %Method computes A(bhat)*X(chat), pvar, and pvarBar based on the
        %P-BiG-AMP derivation given the specified inputs. opt is an object
        %of class PBiGAMPOpt
        %Notice that for uniformVariance, nuc should be 2x1
        function [z,pvarBar,pvar] = pComputation(obj,opt,bhat,nub,chat,nuc)
            
            %Get sizes
            M = obj.returnSizes();
            
            %Grab last index of the low rank part of c
            loc = obj.Nq*obj.R;
            
            
            %Computations depend on uniformVariance flag
            if ~opt.uniformVariance
                
                %Compute Z0j- This is also part of A(b)
                Z0j= obj.Phi'*kron(speye(obj.Nq),reshape(bhat,obj.R,obj.Mq).');
                
                %Compute Zi0
                Zi0= obj.PhiT'*kron(speye(obj.Mq),reshape(chat(1:loc),obj.R,obj.Nq).');
                
                %Compute AX
                z = Z0j*chat(1:loc) + obj.Phi'*chat(loc+1:end);
                
                %Compute pvarBar
                pvarBar = abs(Zi0).^2 * nub + abs(Z0j).^2 * nuc(1:loc) + ...
                    obj.Phi2'*nuc(loc+1:end);
                
                %Compute pvar
                varProd = reshape(nub,obj.R,obj.Mq)'*reshape(nuc(1:loc),obj.R,obj.Nq);
                pvar = obj.Phi2' * varProd(:);
                pvar = pvar + pvarBar;
                
            else
                
                %Compute z
                z = obj.computeZ(bhat,chat);
                
                %Get sums- note that abs() is just to eliminate imaginary
                %numerical error
                sumZi0 = abs(trace(obj.G1* ...
                    (reshape(chat(1:loc),obj.R,obj.Nq)'*reshape(chat(1:loc),obj.R,obj.Nq))));
                
                sumZ0j = abs(trace(obj.G2* ...
                    (reshape(bhat,obj.R,obj.Mq).'*reshape(conj(bhat),obj.R,obj.Mq))));
                
                %Compute pvarBar
                pvarBar = (nub*sumZi0 + nuc(1)*sumZ0j+ nuc(2)*obj.Phi2_sum)/M;
                
                %Compute pvar
                pvar = pvarBar + nub*nuc(1)/M*obj.zij_sumSquared;
                
            end
            
            
        end
        
        
        %% rqComputation
        
        %Method computes Q and R based on the P-BiG-AMP derivation
        %given the specified inputs. opt is an object of class PBiGAMPOpt
        %Notice that for uniformVariance, nuc should be 2x1
        function [rhat,rvar,qhat,qvar] = rqComputation(...
                obj,opt,bhat,nub,chat,nuc,shat,nus)
            
            %Get sizes
            [~,Nb,Nc] = obj.returnSizes();
            
            %Grab last index of the low rank part of c
            loc = obj.Nq*obj.R;
            
            %Uniform variance
            if ~opt.uniformVariance
                
                %Compute Z0j- This is also part of A(b)
                Z0j= obj.Phi'*kron(speye(obj.Nq),reshape(bhat,obj.R,obj.Mq).');
                
                %Compute Zi0- go ahead and complex conjugate
                Zi0= (obj.PhiT'*kron(speye(obj.Mq),reshape(chat(1:loc),obj.R,obj.Nq).'))';
                
                %Directly compute single sum rhat term
                rhat = [Z0j'*shat; obj.Phi*shat];
                
                %Single sum term in qhat
                qhat = Zi0*shat;
                
                %Directly compute rvar (currently inverted)
                rvar = [abs(Z0j').^2*nus; obj.Phi2*nus];
                
                %Compute qvar (currently inverted)
                qvar = abs(Zi0).^2*nus;
                
                %Handle double sum components
                rhat(1:loc) = rhat(1:loc) - chat(1:loc) .* ...
                    vec(reshape(nub,obj.R,obj.Mq)* ...
                    reshape(obj.Phi2*nus,obj.Mq,obj.Nq));
                
                qhat = qhat - bhat .* ...
                    vec(reshape(nuc(1:loc),obj.R,obj.Nq)* ...
                    reshape(obj.Phi2*nus,obj.Mq,obj.Nq)');
                
            else
                
                %Get sums- note that abs() is just to eliminate imaginary
                %numerical error
                sumZi0 = abs(trace(obj.G1* ...
                    (reshape(chat(1:loc),obj.R,obj.Nq)'*reshape(chat(1:loc),obj.R,obj.Nq))));
                
                sumZ0j = abs(trace(obj.G2* ...
                    (reshape(bhat,obj.R,obj.Mq).'*reshape(conj(bhat),obj.R,obj.Mq))));
                
                %Compute rhat and qhat single sum terms
                rhat = zeros(Nc,1);
                rhat(1:loc) = vec(reshape(conj(bhat),obj.R,obj.Mq)* ...
                    reshape(obj.Phi*shat,obj.Mq,obj.Nq));
                rhat((loc+1):Nc) = obj.Phi*shat;
                
                qhat = vec(reshape(conj(chat(1:loc)),obj.R,obj.Nq)* ...
                    reshape(obj.PhiT*shat,obj.Nq,obj.Mq));
                
                %Compute rvar (currently inverted)
                rvar(1) = nus/(loc)*sumZ0j;
                rvar(2) = nus/(Nc-loc)*obj.Phi2_sum;
                
                %Compute qvar (currently inverted)
                qvar = nus/Nb*sumZi0;
                
                %Double sum term
                rhat(1:loc) = rhat(1:loc) - ...
                    nus*nub*(chat(1:loc) .* obj.zij_sumMi);
                qhat = qhat - nus*nuc(1)*(bhat .* obj.zij_sumMj);
                
            end
            
            %Invert the variance computations
            rvar = 1 ./ (rvar + realmin);
            qvar = 1 ./ (qvar + realmin);
            
            %Enforce variance limits
            rvar(rvar > opt.varThresh) = opt.varThresh;
            qvar(qvar > opt.varThresh) = opt.varThresh;
            
            
            %Scale the rhat and qhat terms by the variances and then add in
            %the chat and bhat terms
            if ~opt.uniformVariance
                rhat = rhat .* rvar;
            else
                rhat(1:loc) = rhat(1:loc)*rvar(1);
                rhat((loc+1):end) = rhat((loc+1):end)*rvar(2);
            end
            rhat = rhat + chat;
            qhat = qhat .* qvar;
            qhat = qhat + bhat;
            
        end
        
    end
    
end
