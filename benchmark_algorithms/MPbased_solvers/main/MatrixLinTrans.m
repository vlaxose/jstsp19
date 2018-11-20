classdef MatrixLinTrans < LinTrans
    % LinTrans:  Linear transform class with a matrix
    
    properties
        A;      % Matrix A
        Asq;    % Asq = (A.^2)
        Avar;   % Variance of A entries.
    end
    
    methods
        
        % Constructor
        function obj = MatrixLinTrans(A,Avar)
            obj = obj@LinTrans;
            obj.A = A;
            obj.Asq = (abs(A).^2);
            
            %Assign matrix entry-wise variances if provided
            if nargin < 2
                obj.Avar = 0;
            else
                obj.Avar = Avar;
            end
        end
        
        % size method ( deals with optional dimension argin  ; nargout={0,1,2} )
        function [m,n] = size(obj,dim)
            if nargin>1 % a specific dimension was requested
                if dim>2
                    m=1;
                else
                    m=size(obj.A,dim);
                end
            elseif nargout<2  % all dims in one output vector
                m=size(obj.A);
            else % individual outputs for the dimensions
                [m,n] = size(obj.A);
            end
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            y = obj.A*x;
        end
        % Matrix multiply transpose
        function y = multTr(obj,x)
            y = obj.A'*x;
        end
        % Matrix multiply with square
        function y = multSq(obj,x)
            y = obj.Asq*x;
        end
        % Matrix multiply transpose
        function y = multSqTr(obj,x)
            y = obj.Asq'*x;
        end
        
        %Include matrix uncertainty
        function pvar = includeMatrixUncertainty(obj,pvarBar,xhat,xvar)
            
            if isequal(obj.Avar,0) %Do nothing if the uncertainty is zero
                pvar = pvarBar;
            elseif isscalar(obj.Avar) %fast implementation for scalar Avar
                
                %Just a sum scaled by the Avar value
                pvar = pvarBar +...
                    obj.Avar*sum(xvar + abs(xhat).^2)*ones(size(pvarBar));
            else
                %Otherwise we need the full matrix multiply
                pvar = pvarBar + obj.Avar*(xvar + abs(xhat).^2);
            end
            
        end
        
        
    end
end