classdef LowRank_Matrix_Recovery_ParametricZ < ParametricZ
    %This class implements the ParametricZ functinality for low rank matrix
    %recovery. This problem is more general than matrix recovery, since the
    %measurements are arbitrary linear functions of the entries of the entire
    %low rank matrix.
    %The low rank matrix is Q = B.'C.
    %The matrix is rank R. Q is Mq x Nq
    %B is R x Mq=Nb/R
    %C is R x Nq=Nc/R
    %where Nb and Nc are the sizes of the two parameters.
    %The parameter vector b = vec(B)
    %The parameter vector c = vec(C).
    %The model returns M linear funcions of the entries of Q. Each measurement
    %corresponds to a measurement matrix Phi_m. We thus have:
    %z_m = trace(Phi_m' * Q) for m = 1....M.
    %Phi should be provided as a sparse matrix if it is structually sparse.
    %We can write this as
    %z_m = vec(Phi_m)'vec(Q) = vec(Phi_m)'*kron(speye(obj.Nq),B.')*c
    %Further optimizations for speed improvement are likely possible.
    
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
        G1; %Sum of Phi_m^H*Phi_m
        G2; %Sum of Phi_m*Phi_m^H
    end
    
    
    
    methods
        
        %% default constructor
        function obj = LowRank_Matrix_Recovery_ParametricZ(Mq,Nq,R,Phi)
            
            if nargin < 4
                error('Require 4 arguments')
            end
            
            %Check
            if (size(Phi,1) ~= Nq*Mq)% || (R > min(Nq,Mq))
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
            
            %Build Phi matrices
            obj.G1 = zeros(Nq,Nq);
            obj.G2 = zeros(Mq,Mq);
            for m = 1:size(Phi,2)
                Phim = reshape(Phi(:,m),Mq,Nq);
                obj.G1 = obj.G1 +  Phim'*Phim;
                obj.G2 = obj.G2 + Phim*Phim';
            end
        end
        
        %% Function to compute low rank matrix from factors
        function Q = computeQ(obj,bhat,chat)
            %Just a multiplication after reshape
            Q = reshape(bhat,obj.R,obj.Mq).'*reshape(chat,obj.R,obj.Nq);
        end
        
        
        %% returnSizes
        %Return all problem dimensions
        function [M,Nb,Nc] = returnSizes(obj)
            
            %Obtain sizes
            M = size(obj.Phi,2);
            Nb = obj.Mq*obj.R;
            Nc = obj.Nq*obj.R;
            
        end
        
        
        %% The getter for A
        
        %This setup can be cast in the form Z = A(b)*C, where Z is MqxNq
        %and C is reshaped to R x Nq. This function returns the A()
        %operator that corresponds to this formulation of the problem.
        function Aop = getAOperator(obj,bhat)
            
            %Build the matrix- this could be made more efficient
            Amat = obj.Phi'*(kron(speye(obj.Nq),reshape(bhat,obj.R,obj.Mq).'));
            
            
            %Return a LinTrans operator
            Aop = MatrixLinTrans(Amat);
            
        end
        
        
        
        
        %% computeZ
        %Compute z = vec(B.'*C), where B and C are appropriately reshaped
        function z = computeZ(obj,bhat,chat)
            
            %Compute the product
            Q = reshape(bhat,obj.R,obj.Mq).' * reshape(chat,obj.R,obj.Nq);
            
            %Compute the result
            z = vec(obj.Phi' * Q(:));
            
        end
        
        
        
        %% pComputation
        %Method computes z(bhat,chat), pvar, and pvarBar based on the
        %P-BiG-AMP derivation given the specified inputs. opt is an object
        %of class PBiGAMPOpt
        function [z,pvarBar,pvar] = pComputation(obj,opt,bhat,nub,chat,nuc)
            
            %Get sizes
            M = obj.returnSizes();
            
            %Computations depend on uniformVariance flag
            if ~opt.uniformVariance
                
                %Compute Z0j- This is Also A(b)
                Z0j= obj.Phi'*kron(speye(obj.Nq),reshape(bhat,obj.R,obj.Mq).');
                
                
                %Compute Zi0
                Zi0= obj.PhiT'*kron(speye(obj.Mq),reshape(chat,obj.R,obj.Nq).');
                
                %Compute z
                z = Z0j*chat;
                
                %Compute pvarBar
                pvarBar = abs(Zi0).^2 * nub + abs(Z0j).^2 * nuc;
                
                %Compute pvar
                varProd = reshape(nub,obj.R,obj.Mq)'*reshape(nuc,obj.R,obj.Nq);
                pvar = obj.Phi2' * varProd(:);
                pvar = pvar + pvarBar;
            else
                
                %Compute z
                z = obj.computeZ(bhat,chat);
                
                %Get sums- note that abs() is just to eliminate imaginary
                %numerical error
                sumZi0 = abs(trace(obj.G1* ...
                    (reshape(chat,obj.R,obj.Nq)'*reshape(chat,obj.R,obj.Nq))));
                
                sumZ0j = abs(trace(obj.G2* ...
                    (reshape(bhat,obj.R,obj.Mq).'*reshape(conj(bhat),obj.R,obj.Mq))));
                
                %Compute pvarBar
                pvarBar = (nub*sumZi0 + nuc*sumZ0j)/M;
                
                %Compute pvar
                pvar = pvarBar + nub*nuc/M*obj.zij_sumSquared;
                
            end
            
        end
        
        
        %% rqComputation
        
        %Method computes Q and R based on the P-BiG-AMP derivation
        %given the specified inputs. opt is an object of class PBiGAMPOpt
        function [rhat,rvar,qhat,qvar] = rqComputation(...
                obj,opt,bhat,nub,chat,nuc,shat,nus)
            
            %Get sizes
            [~,Nb,Nc] = obj.returnSizes();
            
            %Handle uniform variance
            if ~opt.uniformVariance
                
                %Compute Z0j- This is Also A(b)
                Z0j = obj.Phi'*kron(speye(obj.Nq),reshape(bhat,obj.R,obj.Mq).');
                
                %Compute Zi0- go ahead and complex conjugate
                Zi0 = (obj.PhiT'*kron(speye(obj.Mq),reshape(chat,obj.R,obj.Nq).'))';
                
                %Compute rhat and qhat single sum terms
                rhat = Z0j'*shat;
                qhat = Zi0*shat;
                
                %Directly compute rvar (currently inverted)
                rvar = abs(Z0j').^2*nus;
                
                %Compute qvar (currently inverted)
                qvar = abs(Zi0).^2*nus;
                
                %Handle double sum terms
                rhat = rhat - chat .* ...
                    vec(reshape(nub,obj.R,obj.Mq)*reshape(obj.Phi2*nus,obj.Mq,obj.Nq));
                
                qhat = qhat - bhat .* ...
                    vec(reshape(nuc,obj.R,obj.Nq)*reshape(obj.Phi2*nus,obj.Mq,obj.Nq)');
                
            else
                
                %Compute rhat and qhat single sum terms
                rhat = vec(reshape(conj(bhat),obj.R,obj.Mq)* ...
                    reshape(obj.Phi*shat,obj.Mq,obj.Nq));
                
                qhat = vec(reshape(conj(chat),obj.R,obj.Nq)* ...
                    reshape(obj.PhiT*shat,obj.Nq,obj.Mq));
                
                %Get sums- note that abs() is just to eliminate imaginary
                %numerical error
                sumZi0 = abs(trace(obj.G1* ...
                    (reshape(chat,obj.R,obj.Nq)'*reshape(chat,obj.R,obj.Nq))));
                
                sumZ0j = abs(trace(obj.G2* ...
                    (reshape(bhat,obj.R,obj.Mq).'*reshape(conj(bhat),obj.R,obj.Mq))));
                
                %Compute rvar (currently inverted)
                rvar = nus/Nc*sumZ0j;
                
                %Compute qvar (currently inverted)
                qvar = nus/Nb*sumZi0;
                
                %Double sum term
                rhat = rhat - nus*nub*(chat .* obj.zij_sumMi);
                qhat = qhat - nus*nuc*(bhat .* obj.zij_sumMj);
                
            end
            
            %Invert the variance computations
            rvar = 1 ./ (rvar + realmin);
            qvar = 1 ./ (qvar + realmin);
            
            %Enforce variance limits
            rvar(rvar > opt.varThresh) = opt.varThresh;
            qvar(qvar > opt.varThresh) = opt.varThresh;
            
            
            %Scale the rhat and qhat terms by the variances and then add in
            %the chat and bhat terms
            rhat = rhat .* rvar;
            rhat = rhat + chat;
            qhat = qhat .* qvar;
            qhat = qhat + bhat;
            
        end
        
    end
    
end
