classdef LinTrans < hgsetget
    % LinTrans:  Abstract base class for specifying a linear transform in
    % the GAMP algorithm.  Linear transforms are specified by deriving from
    % this base class and implementing the methods listed below.  These
    % methods include operations to multiply by the matrix, its transpose
    % and the componentwise square of the matrix.
    %
    %    methods (Abstract)
    %
    %        % Matrix multiply:  z = A*x
    %        y = mult(obj,x)
    %
    %        % Matrix multiply transpose:  x = A'*z
    %        x = multTr(obj,z)
    %
    %        % The subclass should also supply either
    %        %       FrobNorm: The Frobenius Norm
    %        % or
    %        %       multSq and multSqTr
    %
    %         Nominally, the multSq,multSqTr methods implement
    %           z = abs(A).^2 * x
    %          and
    %           y = abs(A').^2 * x
    %         respectively.
    %
    %         However a rank-1 approximation is common since it is fast,
    %         easy, and often does exactly the right thing.  If this is not the desired behavior,
    %         then the subclass must provide the multSq,multSqTr methods.
    %
    %         This approximation matrix is ones(m,n) times the
    %         Frobenius norm squared divided by the product of the dimensions
    %         That is what is implemented here.  It implicitly assumes either
    %           abs(A_ij) is constant (e.g. DFT , Rademacher )
    %         or
    %           the input vector x is a constant value (i.e. uniform variance) and every
    %         row,column of A has the same L2 norm as the other rows,columns of A
    %         (e.g. unitary transforms , square or stacked into frames of orthobases)
    %    end
    properties
        dims;  % to be supplied by subclass constructor if needed for size method
        FrobNorm = -1; % matrix Frobenius Norm: non-positive value triggers estimation , zero value triggers *quiet* estimation
    end
    
    methods
        % Constructor
        function obj = LinTrans(m,n,FrobNorm) % if the FrobNorm is not given, it will be estimated before multSq or multSqTr
            switch nargin
                case 0
                    obj.dims=[1,1]; % allow no-argument constructor
                case 1
                    obj.dims=[m,m]; %square
                case 2
                    obj.dims=[m,n]; % rectangular m-by-n
                case 3
                    obj.dims=[m,n]; %
                    obj.FrobNorm = FrobNorm;
            end
        end
        
        % size method ( deals with optional dimension argin  ; nargout={0,1,2} )
        function [m,n] = size(obj,dim)
            if nargin>1 % a specific dimension was requested
                if dim>2
                    m=1;
                else
                    m=obj.dims(dim);
                end
            elseif nargout<2  % all dims in one output vector
                m=obj.dims;
            else % individual outputs for the dimensions
                m=obj.dims(1);
                n=obj.dims(2);
            end
        end
        
        
        % rank-1 approximation of elementwise Matrix multiply
        function z = multSq(obj,x)
            if obj.FrobNorm <= 0
                obj.consCheck();
                obj.estFrob();
            end
            [m,n] = obj.size();
            z = ones(m,1) * (  (obj.FrobNorm^2/(m*n))*sum(x) );
        end
        
        % rank-1 approximation of elementwise Matrix multiply transpose
        function y = multSqTr(obj,x)
            if obj.FrobNorm <= 0
                obj.consCheck();
                obj.estFrob();
            end
            [m,n] = obj.size();
            y = ones(n,1) * (  (obj.FrobNorm^2/(m*n))*sum(x) );
        end

        % returns the Frobenius norm squared
        function fn2 = squaredNorm(obj)
            [m,n] = obj.size();
            fn2 = sum( obj.multSq( ones(n,1) ) );
            % TODO: we could optionally get the induced l2 norm by power iteration (if there was a need)
        end

        %Optional method, not included in required interface. This method
        %returns an updated version of pvar based on matrix uncertainty in
        %the operator. gampEst checks using ismethod to see if this is
        %defined. Was not added to this template specifically to avoid
        %breaking existing code.
        %Include matrix uncertainty
        %function pvar = includeMatrixUncertainty(obj,pvarBar,xhat,xvar)
        
        % Multiplication operator
        function z = mtimes(obj,x)
            z = obj.mult(x);
        end
        
        % Conjugate transpose. Returns an operator wrapper class with the
        % adjoint flag set.
        function obj1 = ctranspose(obj)
            obj1 = LTOperator(obj,true,false);
        end
        
        % Absolute value squared.  Returns an operator wrapper class with
        % the transpose flag set.
        function obj1 = absSq(obj)
            obj1 = LTOperator(obj,false,true);
        end
        
        % estimate the Frobenius norm of the operator.  This is used in multSq, multSqTr rank-one approximations
        function estFrob(obj,qual)
            [~,n] = obj.size();
            % approximate the squared Frobenius norm
            if nargin < 2
                qual = min(50,n);      % increase for a better approximation
            end
            if obj.FrobNorm == -1
                fprintf('Estimating Frobenius Norm...\n Note to developer: set LinTrans.FrobNorm to either \n a.  The Frobenius norm for your operator or \n b. zero to estimate quietly \n');
            end
            tmp=0;
            for p=1:qual,   % using a "for" loop with vectors since subclass mult may not support matrices
                if qual == n  % no reason to do more vectors than we have columns, in that case we can compute the Frob Norm exactly
                    x=zeros(n,1);  % do exhaustive search
                    x(p) = 1;
                    normX = 1;
                else
                    x = randn(n,1);  % random sampling of dimensions
                    normX = norm(x);
                end
                y = obj.mult( x );
                tmp = tmp + ( norm(y) / normX ).^2;
            end
            obj.FrobNorm = sqrt( tmp * ( n / qual) );
        end
        
        % consistency check for adjointness of mult and multTr operators
        function consCheck(obj)
            
            %Get sizes
            [m,n] = obj.size();
            
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
