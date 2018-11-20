classdef Matrix_PLinTrans < PLinTrans
    %Implements A(b) = A0 + \sum_i bi*A_i (i = 1...Nb)
    %for the case where A0 and all the A_i are available as (possibly
    %sparse) matrices. A0 and all the A_i are KxN
    
    properties
        
        %The offset matrix. Set to empty to ignore
        A0;
        
        %The A_i matrices. These are stored in a single matrix Ai
        %that is (K*N) x Nb. The ith column is vec(A_i). The result is that
        %A(b) = A0 + reshape(Ai*b,K,N)
        Ai;
        Ai2=[]; %elementwise squared
        
        %Storage of computed matrices A(b) and abs(A(b)).^2
        A=[];
        A2=[];
        
        %Storage of computed Frobenius norm
        Afrob_holder = [];
        
        %Storage of G matrix
        G = [];
        
        
    end
    
    
    methods
        
        %Default constructor
        function obj = Matrix_PLinTrans(Ai,A0,K,N)
            
            %Assign
            obj.A0 = A0;
            obj.Ai = Ai;
            obj.K = K;
            obj.N = N;
            obj.b = zeros(size(Ai,2),1);
            
        end
        
        
        %Override set method
        function setParameter(obj,b)
            
            %Check if the b changed
            if any(obj.b ~= b)
                
                %Assign the new b
                obj.b = b;
                
                %Reset computed quantities
                obj.A = [];
                obj.A2 = [];
                obj.Afrob_holder = [];
                
            end
            
        end
        
        %Method to compute A(b)
        function computeA(obj)
            
            %If needed, compute
            if isempty(obj.A)
                
                if isempty(obj.A0)
                    obj.A = reshape(obj.Ai*obj.b,obj.K,obj.N);
                else
                    obj.A = obj.A0 + reshape(obj.Ai*obj.b,obj.K,obj.N);
                end
            end
            
            
            
        end
        
        
        %Method to compute abs(A(b)).^2
        function computeA2(obj)
            
            %First, need to compute A
            obj.computeA();
            
            %Now, compute A2 if needed
            if isempty(obj.A2)
                obj.A2 = abs(obj.A).^2;
            end
            
            
        end
        
        %Method to compute abs(Ai).^2
        function computeAi2(obj)
            
            %Square Ai if needed
            if isempty(obj.Ai2)
                obj.Ai2 = abs(obj.Ai).^2;
            end
            
        end
        
        
        %Implements A(b)*C
        function Z = mult(obj,C)
            
            %Compute A
            obj.computeA();
            
            %Do it
            Z = obj.A*C;
            
        end
        
        %Implements A(b)'*Z
        function C = multTr(obj,Z)
            
            %Compute A
            obj.computeA();
            
            %Do it
            C = obj.A'*Z;
            
        end
        
        %Implements abs(A(b)).^2*C
        function Z = multSq(obj,C)
            
            %Compute A2
            obj.computeA2();
            
            %Do it
            Z = obj.A2*C;
            
        end
        
        %Implements (abs(A(b)).^2)'*Z
        function C = multSqTr(obj,Z)
            
            %Compute A2
            obj.computeA2();
            
            %Do it
            C = obj.A2'*Z;
            
        end
        
        %This method should multiply the matrix C (NxL) with each of the Nb
        %matrices A_i. The result Zi is (K*L)xNb, where the ith column is
        %given by: vec(A_i*C)
        function Zi = multWithDerivatives(obj,C)
            
            %Use a sparse kron
            Zi = kron(C.',speye(obj.K))*obj.Ai;
            
            
            
        end
        
        
        
        %This method should multiply the matrix C (NxL) with each of the Nb
        %element-wise squared matrices abs(A_i).^2.
        %The result Zi2 is (K*L)xNb, where the ith column is
        %given by: vec(abs(A_i).^2*C)
        function Zi2 = multWithSqrDerivatives(obj,C)
            
            %Compute Ai2
            obj.computeAi2();
            
            %Compute with sparse kron
            Zi2 = kron(C.',speye(obj.K))*obj.Ai2;
            
            
        end
        
        %Method to compute the Frobenius norm of A(b)
        function Afrob = computeFrobNorm(obj)
            
            %Check for norm
            if isempty(obj.Afrob_holder)
                %First, compute A
                obj.computeA();
                
                %Now the norm
                obj.Afrob_holder = norm(obj.A,'fro');
                
            end
            
            %Use it
            Afrob =  obj.Afrob_holder;
            
            
        end
        
        %Method to compute \sum_i norm(A_i*C,'fro')^2
        function frobSum = computeFrobSum(obj,C)
            
            %Compute G
            obj.computeG();
            
            %Compute the sum
            frobSum = abs(trace(obj.G*(C*C')));
            
            
        end
        
        %Method to compute G
        function computeG(obj)
            
            %If not already done
            if isempty(obj.G)
                
                obj.G = 0;
                for ii = 1:size(obj.Ai,2)
                    Aiimat = reshape(obj.Ai(:,ii),obj.K,obj.N);
                    obj.G = obj.G +  Aiimat'*Aiimat;
                end
                
            end
            
        end
        
        %Method to compute Frobenius norms of parts of {Ai}.
        %Ainorms Should be a vector of length Nb, where
        %AiNorms(ii) = norm(A_i,'fro')
        %AnNorms is a vector of length K where
        %AnNorms(k)^2 = \sum_i sum(abs(A_i(:,k)).^2)
        function [AiNorms,AnNorms] = computeAFrobNorms(obj)
            
            %Preallocate
            AiNorms = zeros(obj.parameterDimension(),1);
            
            %Compute them
            for ii = 1:length(AiNorms)
                AiNorms(ii) = norm(obj.Ai(:,ii),'fro');
            end
            
            %Compute AnNorms
            AnNorms = zeros(obj.N,1);
            for n = 1:obj.N
                %Get indices
                locs = (1:obj.K) + obj.K*(n-1);
                AnNorms(n) = norm(obj.Ai(locs,:),'fro');
            end
            
            
        end
        
        %Method to compute Ab = \sum_i nub(i)*abs(A_i).^2
        function Ab = computeWeightedSum(obj,nub)
            
            %Prep Ai2
            obj.computeAi2();
            
            %Compute it
            Ab = reshape(obj.Ai2*nub,obj.K,obj.N);
            
            
        end
        
    end
    
    
    
    
    
    
    
    
end