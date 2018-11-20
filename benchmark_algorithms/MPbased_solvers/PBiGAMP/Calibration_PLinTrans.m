classdef Calibration_PLinTrans < PLinTrans
    %Implements A(b) = diag(H*b)*A0
    %This code could be easily generalized to the case where A0 is a fixed
    %implicit operator, such as an object of the GAMPMATLAB LinTrans
    %class, but this version assumes that both H and A0 are available as
    %(possibly sparse) explicit matrices.
    
    properties
        
        %The underlying operator
        A0;
        
        %The calibration subspace
        H;
        
        %Useful stored values
        A02; %Elementwise squared A0
        H2; %elementwise squared
        Hb; %Product of H*b
        Hb2; %elementwise squared
        A02_sum; %A02*ones(N,1)
        G=[];
    end
    
    
    methods
        
        %Default constructor
        function obj = Calibration_PLinTrans(A0,H)
            
            %Check sizes
            if size(H,1) ~= size(A0,1)
                error('Sizes are inconsistent')
            end
            
            %Assign
            obj.A0 = A0;
            obj.H = H;
            [obj.K,obj.N] = size(A0);
            
            %Compute
            obj.A02 = abs(A0).^2;
            obj.H2 = abs(H).^2;
            obj.A02_sum = obj.A02*ones(obj.N,1);
            
            %Set param to 0
            obj.b = zeros(size(H,2),1);
            obj.Hb = zeros(size(H,1),1);
            obj.Hb2 = zeros(size(H,1),1);
        end
        
        %Override set method
        function setParameter(obj,b)
            
            
            %Assign the new b
            obj.b = b;
            
            %Recompute H*b
            obj.Hb = obj.H * obj.b;
            obj.Hb2 = abs(obj.Hb).^2;
            
        end
        
        %Implements A(b)*C
        function Z = mult(obj,C)
            
            %Do it
            Z = bsxfun(@times,obj.Hb,(obj.A0*C));
            
        end
        
        %Implements A(b)'*Z
        function C = multTr(obj,Z)
            
            %Do it
            C = obj.A0'*bsxfun(@times,conj(obj.Hb),Z);
            
        end
        
        %Implements abs(A(b)).^2*C
        function Z = multSq(obj,C)
            
            %Do it
            Z = bsxfun(@times,obj.Hb2,(obj.A02*C));
            
        end
        
        %Implements (abs(A(b)).^2)'*Z
        function C = multSqTr(obj,Z)
            
            %Do it
            C = obj.A02'*bsxfun(@times,obj.Hb2,Z);
            
        end
        
        %This method should multiply the matrix C (NxL) with each of the Nb
        %matrices A_i. The result Zi is (K*L)xNb, where the ith column is
        %given by: vec(A_i*C)
        function Zi = multWithDerivatives(obj,C)
            
            %First, compute A0*C
            A0C = vec(obj.A0*C);
            
            %Do it
            Zi = bsxfun(@times,A0C,repmat(obj.H,size(C,2),1));
            
        end
        
        
        
        %This method should multiply the matrix C (NxL) with each of the Nb
        %element-wise squared matrices abs(A_i).^2.
        %The result Zi2 is (K*L)xNb, where the ith column is
        %given by: vec(abs(A_i).^2*C)
        function Zi2 = multWithSqrDerivatives(obj,C)
            
            %First, compute A0*C
            A0C = vec(obj.A02*C);
            
            %Do it
            Zi2 = bsxfun(@times,A0C,repmat(obj.H2,size(C,2),1));
            
            
        end
        
        %Method to compute the Frobenius norm of A(b)
        function Afrob = computeFrobNorm(obj)
            
            %Use it
            Afrob =  sqrt(sum(obj.A02_sum .* obj.Hb2));
            
        end
        
        %Method to compute \sum_i norm(A_i*C,'fro')^2
        function frobSum = computeFrobSum(obj,C)
            
            %This also works and does not require G. May be useful with an
            %implicit A0 version.
            %frobSum = sum(sum(abs(obj.multWithDerivatives(C)).^2));
            
            %Compute G
            obj.computeG();
            
            %Compute the sum
            frobSum = abs(trace(obj.G*(C*C')));
        end
        
        %Method to compute G
        function computeG(obj)
            
            %If not already done
            if isempty(obj.G)
                obj.G = obj.A0'*diag(sum(obj.H2,2))*obj.A0;
            end
            
        end
        
        %Method to compute Frobenius norms of parts of {Ai}.
        %Ainorms Should be a vector of length Nb, where
        %AiNorms(ii) = norm(A_i,'fro')
        %AnNorms is a vector of length K where
        %AnNorms(k)^2 = \sum_i sum(abs(A_i(:,k)).^2)
        function [AiNorms,AnNorms] = computeAFrobNorms(obj)
            
            %Build it
            AiNorms = sqrt(vec(sum(bsxfun(@times,obj.A02_sum,obj.H2),1)));
            
            %Compute AnNorms
            AnNorms = sqrt(vec(sum(bsxfun(@times,sum(obj.H2,2),obj.A02),1)));
            
        end
        
        %Method to compute Ab = \sum_i nub(i)*abs(A_i).^2
        function Ab = computeWeightedSum(obj,nub)
            
            %Compute it
            Ab = diag(obj.H2*nub)*obj.A02;
            
            
        end
        
    end
    
    
    
    
    
    
    
    
end