classdef PLinTrans < hgsetget
    % PLinTrans: Abstract class to specify the methods and properties
    % required for a Parametric linear transformation. The object
    % implements various operations with A(b) = A0 + \sum_i bi*A_i
    
    properties (Access=protected)
        b;  % Length Nb parameter vector. Cannot be changed or accessed
        K;N; %Matrix is KxN
    end
    
    methods (Abstract)
        
        %Implements A(b)*C
        Z = mult(obj,C);
        
        %Implements A(b)'*Z
        C = multTr(obj,Z);
        
        %Implements abs(A(b)).^2*C
        Z = multSq(obj,C);
        
        %Implements (abs(A(b)).^2)'*Z
        C = multSqTr(obj,Z);
        
        %This method should multiply the matrix C (NxL) with each of the Nb
        %matrices A_i. The result Zi is (K*L)xNb, where the ith column is
        %given by: vec(A_i*C)
        Zi = multWithDerivatives(obj,C);
        
        %This method should multiply the matrix C (NxL) with each of the Nb
        %element-wise squared matrices abs(A_i).^2.
        %The result Zi2 is (K*L)xNb, where the ith column is
        %given by: vec(abs(A_i).^2*C)
        Zi2 = multWithSqrDerivatives(obj,C);
        
        %Method to compute Ab = \sum_i nub(i)*abs(A_i).^2
        Ab = computeWeightedSum(obj,nub);
        
        %Method to compute the Frobenius norm of A(b)
        Afrob = computeFrobNorm(obj);
        
        %Method to compute \sum_i norm(A_i*C,'fro')^2
        frobSum = computeFrobSum(obj,C);
        
        %Method to compute Frobenius norms of parts of {Ai}.
        %Ainorms Should be a vector of length Nb, where
        %AiNorms(ii) = norm(A_i,'fro')
        %AnNorms is a vector of length N where
        %AnNorms(n)^2 = \sum_i sum(abs(A_i(:,n)).^2)
        [AiNorms,AnNorms] = computeAFrobNorms(obj);
        
    end
    
    methods
        
        %Return the size of the parameter vector
        function Nb = parameterDimension(obj)
            Nb = numel(obj.b);
        end
        
        %Set b
        function setParameter(obj,b)
            obj.b = b;
        end
        
        %Get b
        function b = getParameter(obj)
            b = obj.b;
        end
        
        %Getter for size
        function [K,N] = getSizes(obj)
            K = obj.K;
            N = obj.N;
        end
        
        %This method verifies that (A(b)*v)'*u = v'*(A(b)'*u) for the
        %current parameter b with randomly drawn u and v. This is a
        %standard test to ensure that the adjoint is correctly implemented
        function consCheck(obj)
            
            %Get sizes
            m = obj.K;
            n = obj.N;
            
            %Start with complex check
            try
                
                %Random
                u = randn(m,1) + 1j*randn(m,1);
                v = randn(n,1) + 1j*randn(n,1);
                
                %Check
                v1 = v'*obj.multTr(u);
                v2 = obj.mult(v)'*u;
                errorResult = abs(v1 - v2);
                if errorResult > 1e-6
                    error('the forward and adjoint operators seem mismatched for complex data')
                end
                
            catch %#ok<CTCH>
                
                %If the complex fails, try the real check
                warning('check failed for complex data, attempting check with real data') %#ok<WNTAG>
                %Random
                u = randn(m,1);
                v = randn(n,1);
                
                %Check
                v1 = v'*obj.multTr(u);
                v2 = obj.mult(v)'*u;
                errorResult = abs(v1 - v2);
                if errorResult > 1e-6
                    error('the forward and adjoint operators also seem mismatched for real data')
                else
                    fprintf('check was succesful for real data\n') %#ok<WNTAG>
                end
            end
        end
    end
    
    
end